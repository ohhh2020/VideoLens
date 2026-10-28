"""前台控制台：保留终端输入，子服务通过专用 stdin 管道正常关闭。"""

from __future__ import annotations

import argparse
import codecs
import ctypes
import json
import logging
import os
import queue
import secrets
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path
from multiprocessing.connection import Client, Listener
from urllib.request import urlopen

from backend.logging_setup import ANSI_ESCAPE, MANAGED_LOG_ENV, open_session_log

logger = logging.getLogger("zhiying.console")
CONTROL_ENV = "VIDEOLENS_CONSOLE_PIPE"
COMMANDS = {"help", "status", "stop", "restart", "clear"}


class ConsoleOutput:
    """终端提示符/清屏与日志落盘分离，父子进程统一由此输出。"""

    def __init__(self, stream, *, interactive=False, color=False):
        self.stream, self.interactive, self.color = stream, interactive, color
        self.lock = threading.RLock()
        self.file = None
        self.input_open = self.prompt_visible = False

    def start_session(self, directory):
        with self.lock:
            if self.file is not None:
                self.file.close()
            self.file = open_session_log(directory)

    def _prompt(self):
        if self.interactive and self.input_open and not self.prompt_visible:
            self.stream.write("\033[33m<< \033[0m" if self.color else "<< ")
            self.prompt_visible = True

    def prompt(self):
        with self.lock:
            self.input_open = True
            self._prompt()
            self.stream.flush()

    def input_finished(self):
        with self.lock:
            self.input_open = self.prompt_visible = False

    def write(self, text):
        with self.lock:
            if self.file is not None:
                self.file.write(ANSI_ESCAPE.sub("", text))
                self.file.flush()
            if self.prompt_visible:
                # 日志另起一行，不接在黄色输入标记后，也不清掉正在键入的内容。
                self.stream.write("\n")
                self.prompt_visible = False
            self.stream.write(text)
            if text.endswith("\n"):
                self._prompt()
            self.stream.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def clear(self):
        with self.lock:
            # 直接写终端，既不调用 logger，也不把控制码/提示符写入日志。
            self.stream.write("\033[2J\033[3J\033[H")
            self.prompt_visible = False
            self._prompt()
            self.stream.flush()

    def close(self):
        with self.lock:
            if self.prompt_visible:
                self.stream.write("\n")
            self.input_open = self.prompt_visible = False
            if self.file is not None:
                self.file.close()
                self.file = None


def input_lines(stream):
    """Windows 管道先检查可读字节，再读取，避免原生库初始化互锁。

    NumPy/OpenBLAS 初始化会检查 stdin 管道。同步管道上悬挂的阻塞读取会
    让它的 PeekNamedPipe 也等待；因此既不能阻塞 readline，也不能阻塞 ReadFile。
    """
    if os.name != "nt":
        yield from stream
        return
    try:
        import msvcrt

        handle = msvcrt.get_osfhandle(stream.fileno())
    except (AttributeError, OSError, ValueError):
        yield from stream
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    read = kernel.ReadFile
    read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
    read.restype = ctypes.c_int
    kind = kernel.GetFileType
    kind.argtypes = [ctypes.c_void_p]
    kind.restype = ctypes.c_ulong
    is_pipe = kind(handle) == 3
    peek = kernel.PeekNamedPipe
    peek.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
    peek.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    size = ctypes.c_ulong()
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    while True:
        available = ctypes.c_ulong()
        if is_pipe:
            if not peek(handle, None, 0, None, ctypes.byref(available), None):
                error = ctypes.get_last_error()
                if error in {109, 232, 995}:
                    break
                raise ctypes.WinError(error)
            if not available.value:
                time.sleep(0.05)
                continue
        count = min(len(buffer), available.value) if is_pipe else len(buffer)
        if not read(handle, buffer, count, ctypes.byref(size), None):
            error = ctypes.get_last_error()
            if error in {109, 232, 995}:
                break
            raise ctypes.WinError(error)
        if not size.value:
            break
        pending += decoder.decode(buffer.raw[:size.value])
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            yield line.rstrip("\r")
    pending += decoder.decode(b"", final=True)
    if pending:
        yield pending.rstrip("\r")


def send_console_command(command, *, state_path=None, port=8000):
    """同一用户的本机自动化入口，不新增 HTTP 管理接口或启动参数。"""
    if command not in COMMANDS:
        raise ValueError("仅支持 help/status/stop/restart/clear")
    if state_path is None:
        from backend.runtime import runtime_path

        state_path = runtime_path(f"state/console-{port}.json")
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    with Client(state["address"], family=state["family"], authkey=bytes.fromhex(state["token"])) as connection:
        connection.send(command)
        if not connection.poll(5):
            raise TimeoutError("控制台未及时响应")
        return connection.recv()


def web_parser():
    parser = argparse.ArgumentParser(description="在前台启动知影网页；运行时输入 help 查看指令")
    parser.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发时监视前后端文件并正常重启")
    return parser


def start_control_reader(server, stream=None):
    """只在受控子服务中读取管道；管道断开也正常退出，避免遗留孤儿服务。"""
    stream = stream if stream is not None else sys.stdin

    def read():
        for line in input_lines(stream):
            try:
                message = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(message, dict) and message.get("command") == "stop":
                logger.info("收到控制台关闭请求，等待当前请求和分析任务结束")
                server.should_exit = True
                return
        logger.info("控制台连接已关闭，正在正常关闭服务")
        server.should_exit = True

    thread = threading.Thread(target=read, name="console-control", daemon=True)
    thread.start()
    return thread


class ConsoleSupervisor:
    """父进程始终在当前终端前台；restart 等旧服务完全退出后再启动新进程。"""

    def __init__(self, command, *, host="127.0.0.1", port=8000, stream=None, watch_paths=(), control_state_path=None, output=None, log_directory=None):
        self.command = list(command)
        self.host, self.port = host, port
        self.stream = stream if stream is not None else sys.stdin
        self.watch_paths = tuple(Path(p) for p in watch_paths)
        self.commands = queue.Queue()
        self.child = None
        self.intent = None
        self.control_state_path = Path(control_state_path) if control_state_path else None
        self.listener = None
        self.control_closed = threading.Event()
        self.output, self.log_directory = output, log_directory

    def _read_commands(self):
        if self.output is not None:
            self.output.prompt()
        for line in input_lines(self.stream):
            if self.output is not None:
                self.output.input_finished()
            self.commands.put(line.strip().casefold())
            if self.output is not None:
                self.output.prompt()
        if self.output is not None:
            self.output.input_finished()
        self.commands.put(None)

    def _forward_output(self):
        for line in self.child.stdout:
            self.output.write(line)

    def _start_control_channel(self):
        if self.control_state_path is None:
            return
        family = "AF_PIPE" if os.name == "nt" else "AF_UNIX"
        address = rf"\\.\pipe\videolens-{os.getpid()}-{secrets.token_hex(8)}" if os.name == "nt" else str(Path(tempfile.gettempdir()) / f"vl-{secrets.token_hex(8)}.sock")
        token = secrets.token_bytes(32)
        self.listener = Listener(address, family=family, authkey=token)
        self.control_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.control_state_path.write_text(json.dumps({"pid":os.getpid(),"address":address,"family":family,"token":token.hex()}),encoding="utf-8")

        def receive():
            while not self.control_closed.is_set():
                try:
                    with self.listener.accept() as connection:
                        command = connection.recv()
                        if isinstance(command, str) and command in COMMANDS:
                            self.commands.put(command)
                            connection.send({"accepted":True,"supervisor_pid":os.getpid()})
                        else:
                            connection.send({"accepted":False})
                except (OSError, EOFError):
                    if self.control_closed.is_set():
                        return
        threading.Thread(target=receive,name="console-local",daemon=True).start()

    def _snapshot(self):
        files = {}
        for root in self.watch_paths:
            for path in root.rglob("*"):
                if path.suffix in {".py", ".js", ".css", ".html"}:
                    try:
                        files[str(path)] = path.stat().st_mtime_ns
                    except FileNotFoundError:
                        pass  # 编辑器原子替换文件时，下一轮扫描再确认。
        return files

    def _shutdown(self, intent):
        if self.intent == "stop":
            return
        already_closing = self.intent is not None
        self.intent = intent
        if already_closing:
            return
        logger.info("正在%s，等待当前请求和任务结束；不会强制终止推理", "重启" if intent == "restart" else "关闭")
        try:
            self.child.stdin.write(json.dumps({"command": "stop"}) + "\n")
            self.child.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # 子进程已经退出时由主循环统一收尾。

    def _status(self):
        host = "[::1]" if self.host == "::1" else self.host
        try:
            with urlopen(f"http://{host}:{self.port}/internal/health", timeout=2) as response:
                health = json.load(response)
            devices = "，".join(f"{name}={device}" for name, device in health.get("model_devices", {}).items())
            logger.info("服务状态=%s，PID=%s，运行中任务=%s；%s", health.get("status"), self.child.pid, health.get("jobs_running", 0), devices)
        except Exception:
            logger.info("子服务 PID=%s 尚未就绪，或正在关闭；可等待启动日志确认", self.child.pid)

    def _handle(self, command):
        if command is None:
            logger.info("控制台输入已结束，服务继续运行；Ctrl+C 可正常关闭")
        elif command in {"help", "?", "帮助"}:
            logger.info("控制台指令：help 帮助；status 状态；clear 清屏；stop 关闭；restart 重启；exit/quit 同 stop")
        elif command in {"clear", "清屏"}:
            if self.output is not None:
                self.output.clear()
        elif command in {"stop", "exit", "quit", "关闭", "退出"}:
            self._shutdown("stop")
        elif command in {"restart", "重启"}:
            self._shutdown("restart")
        elif command in {"status", "状态"}:
            self._status()
        elif command:
            logger.warning("未知指令 %r；输入 help 查看支持的指令", command)

    def run(self):
        self._start_control_channel()
        threading.Thread(target=self._read_commands, name="console-input", daemon=True).start()
        try:
            while True:
                if self.output is not None and self.log_directory is not None:
                    self.output.start_session(self.log_directory)
                logger.info("控制台已就绪，输入 help 查看指令；Ctrl+C 也可关闭服务")
                env = dict(os.environ)
                env[CONTROL_ENV] = "1"
                env["PYTHONUNBUFFERED"] = "1"
                if self.output is not None and self.log_directory is not None:
                    env[MANAGED_LOG_ENV] = "1"
                self.intent = None
                self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, text=True, encoding="utf-8", env=env,
                    stdout=subprocess.PIPE if self.output is not None else None,
                    stderr=subprocess.STDOUT if self.output is not None else None,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
                reader = None
                if self.output is not None:
                    reader = threading.Thread(target=self._forward_output, name="console-output", daemon=True)
                    reader.start()
                previous = self._snapshot()
                next_scan = time.monotonic() + 1
                while self.child.poll() is None:
                    try:
                        command = self.commands.get(timeout=0.2)
                    except queue.Empty:
                        pass
                    except KeyboardInterrupt:
                        self._shutdown("stop")
                    else:
                        self._handle(command)
                    if self.watch_paths and self.intent is None and time.monotonic() >= next_scan:
                        current = self._snapshot()
                        next_scan = time.monotonic() + 1
                        if current != previous:
                            logger.info("检测到前后端代码变更")
                            self._shutdown("restart")
                code = self.child.wait()
                if reader is not None:
                    reader.join()
                    self.child.stdout.close()
                self.child.stdin.close()
                if self.intent != "restart":
                    logger.info("服务已关闭（退出码 %s）", code)
                    return code
                logger.info("旧服务已关闭，正在启动新服务")
        finally:
            if self.child is not None and self.child.poll() is None:
                self._shutdown("stop")
                while self.child.poll() is None:
                    try:
                        self.child.wait(timeout=0.2)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        continue
                self.child.stdin.close()
            self.control_closed.set()
            if self.listener is not None:
                self.listener.close()
            if self.control_state_path is not None:
                self.control_state_path.unlink(missing_ok=True)
            if self.output is not None:
                self.output.close()


def main(argv=None):
    args = web_parser().parse_args(argv)
    from backend.config import load_app_config
    from backend.logging_setup import TaskFormatter
    from backend.runtime import SOURCE_ROOT, runtime_path

    cfg = load_app_config(root=SOURCE_ROOT)
    handler = logging.StreamHandler()
    color = cfg.logging.color == "always" or (
        cfg.logging.color == "auto"
        and (handler.stream.isatty() or (os.environ.get("MSYSTEM") and os.environ.get("TERM", "dumb") != "dumb"))
        and "NO_COLOR" not in os.environ
    )
    if color and os.name == "nt":
        from colorama import just_fix_windows_console

        just_fix_windows_console()
    interactive = sys.stdin.isatty() or bool(os.environ.get("MSYSTEM") and os.environ.get("TERM", "dumb") != "dumb")
    output = ConsoleOutput(handler.stream, interactive=interactive, color=color)
    handler.setStream(output)
    handler.setFormatter(TaskFormatter(cfg.logging.format, cfg.logging.date_format, color=color))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    command = [sys.executable, "-m", "backend.web.main", "--host", args.host, "--port", str(args.port)]
    supervisor = ConsoleSupervisor(command, host=args.host, port=args.port,
        output=output, log_directory=runtime_path(cfg.logging.directory, cfg.root_dir),
        watch_paths=(SOURCE_ROOT / "backend", SOURCE_ROOT / "frontend") if args.reload else (),
        control_state_path=runtime_path(f"state/console-{args.port}.json"))
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
