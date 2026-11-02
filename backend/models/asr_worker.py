"""Windows CUDA转写的受控进程：原生调用超时只回收自己的worker。"""
from dataclasses import asdict
import logging
import multiprocessing
import threading
from types import SimpleNamespace

logger = logging.getLogger('zhiying.models.asr')


class AsrWorkerError(RuntimeError):
    pass


def _serve(connection, spec):
    """只加载Whisper；不导入Torch、不共用网页进程的CUDA上下文。"""
    threading.current_thread().name = 'whisper-worker'
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s][%(levelname)s][%(threadName)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    try:
        from backend.models._runtime import ensure_cuda_dlls
        ensure_cuda_dlls()
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        model = WhisperModel(spec['path'], device=spec['device'], compute_type=spec['compute_type'],
                             cpu_threads=spec['cpu_threads'], num_workers=spec['num_workers'], local_files_only=True)
        connection.send({'type': 'ready', 'device': model.model.device, 'compute_type': model.model.compute_type})
        while True:
            command = connection.recv()
            if command['type'] == 'stop':
                return
            try:
                options = command['options']
                batch = command['batch']
                if batch > 1:
                    segments, info = BatchedInferencePipeline(model=model).transcribe(
                        command['audio'], batch_size=batch, **options)
                else:
                    segments, info = model.transcribe(command['audio'], **options)
                connection.send({'type': 'prepared', 'info': {
                    key: getattr(info, key, 0 if key != 'language' else '')
                    for key in ('duration', 'language', 'language_probability')}})
                for segment in segments:
                    connection.send({'type': 'segment', 'data': asdict(segment)})
                connection.send({'type': 'done'})
            except Exception as exc:
                from backend.models.batching import is_oom
                connection.send({'type': 'error', 'error': f'{type(exc).__name__}: {exc}',
                                 'recoverable': is_oom(exc)})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({'type': 'error', 'error': f'{type(exc).__name__}: {exc}'})
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class AsrWorker:
    def __init__(self, spec, *, timeout=90, start_timeout=60, _entry=_serve):
        self.timeout = timeout
        self.process = None
        self.connection = None
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_entry, args=(child, spec), name='videolens-whisper', daemon=True)
        try:
            self.process.start()
            child.close()
            ready = self._receive(start_timeout)
            if ready['type'] != 'ready':
                raise AsrWorkerError('Whisper进程未完成模型加载')
            self.device, self.compute_type = ready['device'], ready['compute_type']
            if self.device != spec['device'] or self.compute_type != spec['compute_type']:
                raise AsrWorkerError('Whisper实际设备或精度与配置不一致，禁止降级')
        except BaseException:
            child.close()
            self.close()
            raise

    @property
    def alive(self):
        return self.process is not None and self.process.is_alive()

    def _receive(self, timeout=None):
        try:
            if not self.connection.poll(self.timeout if timeout is None else timeout):
                raise AsrWorkerError('Whisper原生调用超过无进展时限，已回收转写子进程；本次分析失败，未自动重试或切换CPU')
            message = self.connection.recv()
        except (EOFError, OSError) as exc:
            self.close()
            raise AsrWorkerError('Whisper转写进程异常退出，本次分析失败，未生成替代转写') from exc
        except AsrWorkerError:
            self.close()
            raise
        if message['type'] == 'error':
            if not message.get('recoverable'):
                self.close()
            raise AsrWorkerError(message['error'])
        return message

    def transcribe(self, audio, batch, options):
        if not self.alive:
            raise AsrWorkerError('Whisper进程未加载，请重新提交任务')
        try:
            self.connection.send({'type': 'transcribe', 'audio': str(audio), 'batch': batch, 'options': options})
        except (OSError, EOFError) as exc:
            self.close()
            raise AsrWorkerError('无法向Whisper进程提交转写') from exc
        prepared = self._receive()
        if prepared['type'] != 'prepared':
            self.close()
            raise AsrWorkerError('Whisper转写协议错误')

        def segments():
            while True:
                message = self._receive()
                if message['type'] == 'done':
                    return
                if message['type'] != 'segment':
                    self.close()
                    raise AsrWorkerError('Whisper片段协议错误')
                data = message['data']
                data['words'] = [SimpleNamespace(**word) for word in data.get('words') or []]
                yield SimpleNamespace(**data)
        return segments(), SimpleNamespace(**prepared['info'])

    def close(self):
        process = self.process
        if process is not None and process.pid is not None:
            if process.is_alive():
                try:
                    self.connection.send({'type': 'stop'})
                except (OSError, EOFError):
                    pass
                process.join(timeout=.5)
            if process.is_alive():
                # multiprocessing对象仅代表本对象创建的子进程，不按系统进程名查杀。
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
        if self.connection is not None:
            self.connection.close()
