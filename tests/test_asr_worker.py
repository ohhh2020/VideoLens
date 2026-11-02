"""使用真实spawn和内存管道测试，不加载模型或访问GPU。"""
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import load_app_config
from backend.models.asr import AsrConfig
from backend.models.asr_worker import AsrWorker, AsrWorkerError


def fake_worker(connection, spec):
    connection.send({'type': 'ready', 'device': spec['device'], 'compute_type': spec['compute_type']})
    try:
        while True:
            message = connection.recv()
            if message['type'] == 'stop':return
            if message['audio'] == 'hang':
                time.sleep(30)
            elif message['audio'] == 'crash':
                return
            elif message['audio'] == 'oom':
                connection.send({'type': 'error', 'error': 'CUDA out of memory', 'recoverable': True})
            elif message['audio'] == 'bad':
                connection.send({'type': 'error', 'error': 'CUDA illegal address'})
            else:
                connection.send({'type': 'prepared', 'info': {'language': 'zh', 'duration': 2, 'language_probability': 1}})
                connection.send({'type': 'segment', 'data': {'start': 0, 'end': 2, 'text': '完整尾句',
                                 'words': [{'start': 0, 'end': 2, 'word': '完整尾句', 'probability': 1}]}})
                connection.send({'type': 'done'})
    finally:
        connection.close()


class AsrWorkerTests(unittest.TestCase):
    def create(self):
        worker = AsrWorker({'device': 'cuda', 'compute_type': 'float16'}, timeout=.15,
                           start_timeout=10, _entry=fake_worker)
        self.addCleanup(worker.close)
        return worker

    def test_segments_keep_words_and_worker_is_reused(self):
        worker = self.create()
        pid = worker.process.pid
        for _ in range(2):
            rows, info = worker.transcribe('normal', 4, {'beam_size': 5})
            rows = list(rows)
            self.assertEqual(info.language, 'zh')
            self.assertEqual(rows[-1].text, '完整尾句')
            self.assertEqual(rows[-1].words[-1].end, 2)
            self.assertEqual(worker.process.pid, pid)
        worker.close()
        self.assertFalse(worker.alive)

    def test_native_timeout_terminates_only_owned_worker(self):
        worker = self.create()
        start = time.monotonic()
        with self.assertRaisesRegex(AsrWorkerError, '未自动重试或切换CPU'):
            worker.transcribe('hang', 4, {})
        self.assertFalse(worker.alive)
        self.assertLess(time.monotonic() - start, 5)
        fresh = self.create()
        self.assertNotEqual(fresh.process.pid, worker.process.pid)
        rows, _ = fresh.transcribe('normal', 4, {})
        self.assertEqual(len(list(rows)), 1)

    def test_child_crash_returns_error_without_empty_transcript(self):
        worker = self.create()
        with self.assertRaisesRegex(AsrWorkerError, '异常退出'):
            worker.transcribe('crash', 4, {})
        self.assertFalse(worker.alive)

    def test_oom_can_shrink_but_other_cuda_failure_recycles_context(self):
        worker = self.create()
        with self.assertRaisesRegex(AsrWorkerError, 'out of memory'):
            worker.transcribe('oom', 4, {})
        self.assertTrue(worker.alive)
        rows, _ = worker.transcribe('normal', 1, {})
        self.assertEqual(len(list(rows)), 1)
        with self.assertRaisesRegex(AsrWorkerError, 'illegal address'):
            worker.transcribe('bad', 4, {})
        self.assertFalse(worker.alive)

    def test_config_wiring_and_timeout_validation(self):
        config = load_app_config(ignore_user_settings=True, overrides={
            'asr.isolate_cuda': True, 'asr.worker_timeout_s': 75, 'asr.worker_start_timeout_s': 40})
        actual = AsrConfig.from_config(config)
        self.assertTrue(actual.isolate_cuda)
        self.assertEqual((actual.worker_timeout_s, actual.worker_start_timeout_s), (75, 40))
        for value in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):AsrConfig(worker_timeout_s=value).validate()


if __name__ == '__main__':unittest.main(verbosity=2)
