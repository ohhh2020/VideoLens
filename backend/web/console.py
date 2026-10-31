"""前台控制台：保留终端输入，子服务通过专用 stdin 管道正常关闭。"""

from __future__ import annotations

import argparse
import codecs
import ctypes
import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path
from multiprocessing.connection import Client, Listener
from urllib.request import urlopen

from backend.logging_setup import ANSI_ESCAPE, MANAGED_LOG_ENV, open_session_log
from backend.terminal import CONSOLE_COLOR_ENV, supports_ansi, interactive_input, use_color

logger = logging.getLogger("zhiying.console")
CONTROL_ENV = "VIDEOLENS_CONSOLE_PIPE"
COMMANDS = {"help", "status", "stop", "restart", "clear"}


def terminal_size():
    """优先读取 Git Bash 的真实 PTY 尺寸，原生终端使用标准控制台查询。"""
    # Windows Terminal 的 Git Bash 常直接继承 ConPTY 控制台句柄，
    # 此时不应反复启动 MSYS stty（可能与控制台输入争用）。
    try:
        return os.get_terminal_size(sys.stderr.fileno())
    except (OSError, ValueError, AttributeError):
        pass
    stty = os.environ.get("VIDEOLENS_TERMINAL_STTY")
    if stty:
        try:
            result = subprocess.run(
                [stty, "-F", os.environ.get("VIDEOLENS_TERMINAL_TTY", "/dev/tty"), "size"], stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=0.5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            rows, columns = map(int, result.stdout.split())
            if result.returncode == 0 and rows >= 3 and columns >= 4:
                return os.terminal_size((columns, rows))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return shutil.get_terminal_size(fallback=(80, 24))


class ConsoleOutput:
    """上方日志滚动、底部固定输入行；终端控制码和提示符不写入文件。"""

    def __init__(self, stream, *, interactive=False, color=False, ansi=None, size_provider=terminal_size):
        self.ansi = bool(interactive or color) if ansi is None else bool(ansi)
        self.stream, self.interactive, self.color = stream, bool(interactive and self.ansi), bool(color and self.ansi)
        self.lock = threading.RLock()
        self.file = None
        self.input_open = self.prompt_visible = False
        self.size_provider = size_provider
        self.rows = None
        self.last_size_check = 0.0
        self.pending = ""
        self.input_text = ""
        self.native_output = os.name == 'nt' and stream in (sys.stdout, sys.stderr)

    def _terminal_write(self, text):
        # 原生库/子进程可能重新设置共享控制台模式。每次渲染前核验，
        # 防止已禁用VT后把光标控制码当文字输出。
        if self.native_output and self.ansi and not supports_ansi(self.stream):
            self.ansi = self.color = self.interactive = False
            self.rows = None
        self.stream.write(text if self.ansi else ANSI_ESCAPE.sub('', text))

    def start_session(self, directory):
        with self.lock:
            if self.file is not None:
                self.file.close()
            self.file = open_session_log(directory)

    def _layout(self):
        size = self.size_provider()
        rows = max(3, size.lines)
        if rows == self.rows:
            return
        # 只保留最后一行输入；DEC 滚动区域不包含这一行。
        if self.rows is not None:
            self._terminal_write(f"\033[{min(self.rows, rows)};1H\033[2K")
        self.rows = rows
        self._terminal_write(f"\033[1;{rows - 1}r")
        self.prompt_visible = False

    def _prompt(self):
        if self.interactive and self.input_open and not self.prompt_visible:
            self._terminal_write(f"\033[{self.rows};1H\033[2K")
            self._terminal_write(("\033[33m<< \033[0m" if self.color else "<< ") + self.input_text)
            self.prompt_visible = True

    def edit_input(self, value):
        with self.lock:
            self.input_text = value
            self.prompt_visible = False
            self._prompt()
            self.stream.flush()

    def refresh_size(self, *, force=False):
        with self.lock:
            if not self.interactive or self.rows is None:
                return
            now = time.monotonic()
            if not force and now - self.last_size_check < 1:
                return
            self.last_size_check = now
            self._layout()
            self._prompt()
            self.stream.flush()

    def prompt(self):
        with self.lock:
            self.input_open = True
            if self.interactive:
                if self.rows is None:
                    self._layout()
                self._prompt()
            self.stream.flush()

    def input_finished(self):
        with self.lock:
            if self.interactive and self.rows is not None:
                self._terminal_write(f"\033[{self.rows};1H\033[2K")
                self.stream.flush()
            self.input_open = self.prompt_visible = False
            self.input_text = ""

    def _logs(self, text):
        if self.interactive and self.rows is not None:
            # 一次原子写入：保存输入光标，定位日志区，输出后恢复。
            # 不清除、不重画输入行，也不为每条日志输出提示符。
            self.pending += text
            end = self.pending.rfind("\n") + 1
            if end:
                lines, self.pending = self.pending[:end], self.pending[end:]
                # DECRC 在部分终端会限制在日志滚动区内，VPA 保留输入列并回到底行。
                self._terminal_write(
                    f"\0337\033[{self.rows - 1};1H\033[2K"
                    + lines.replace("\r\n", "\n").replace("\n", "\r\n")
                    + f"\0338\033[{self.rows}d"
                )
        else:
            self._terminal_write(text)

    def write(self, text):
        with self.lock:
            if self.file is not None:
                self.file.write(ANSI_ESCAPE.sub("", text))
                self.file.flush()
            terminal_text = text if self.ansi else ANSI_ESCAPE.sub('', text)
            if not self.color:
                terminal_text = re.sub(r'\x1b\[[0-9;:]*m', '', terminal_text)
            self._logs(terminal_text)
            self.stream.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def clear(self):
        with self.lock:
            # 直接写终端，既不调用 logger，也不把控制码/提示符写入日志。
            if not self.ansi:
                self._terminal_write('当前输出窗口不支持清屏。\n')
                self.stream.flush()
                return
            self._terminal_write("\033[2J\033[3J\033[H")
            self.prompt_visible = False
            self._prompt()
            self.stream.flush()

    def close(self):
        with self.lock:
            if self.pending:
                self._logs("\n")
            if self.rows is not None:
                # 交还终端时恢复整个屏幕可滚动，避免影响后续 Bash 命令。
                self._terminal_write(f"\033[r\033[{self.rows};1H\033[2K")
            self.input_open = self.prompt_visible = False
            self.rows = None
            self.stream.flush()
            if self.file is not None:
                self.file.close()
                self.file = None


def native_input_lines(on_edit=None):
    """原生控制台逐键非阻塞读取，不悬挂同步ReadFile占用控制台句柄。"""
    import msvcrt
    value = ''
    while True:
        if not msvcrt.kbhit():
            time.sleep(.03)
            continue
        char = msvcrt.getwch()
        if char in ('\x00', '\xe0'):
            msvcrt.getwch()  # 方向键等扩展键，不作为指令字符。
            continue
        if char == '\x03':
            yield 'stop'
            return
        if char in ('\x04', '\x1a'):
            return
        if char in ('\r', '\n'):
            yield value
            value = ''
        elif char == '\b':
            value = value[:-1]
        elif char.isprintable():
            value += char
        if on_edit is not None:
            on_edit(value)


def input_lines(stream, on_edit=None):
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
    # 仅真实标准控制台使用CRT逐键输入；重定向/MSYS管道仍走PeekNamedPipe。
    mode = ctypes.c_ulong()
    get_mode = kernel.GetConsoleMode
    get_mode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    get_mode.restype = ctypes.c_int
    if stream is sys.stdin and not is_pipe and get_mode(handle, ctypes.byref(mode)):
        yield from native_input_lines(on_edit)
        return
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
        for line in input_lines(self.stream, self.output.edit_input if self.output is not None else None):
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
                env['PYTHONUTF8'] = '1'
                env['PYTHONIOENCODING'] = 'utf-8'
                if self.output is not None and self.log_directory is not None:
                    env[MANAGED_LOG_ENV] = "1"
                if self.output is not None:
                    env[CONSOLE_COLOR_ENV] = '1' if self.output.color else '0'
                self.intent = None
                self.child = subprocess.Popen(self.command, stdin=subprocess.PIPE, text=True, encoding="utf-8", errors='replace', env=env,
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
                    if self.output is not None:
                        self.output.refresh_size()
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
                try:
                    owner = json.loads(self.control_state_path.read_text(encoding='utf-8'))
                    if owner.get('pid') == os.getpid():
                        self.control_state_path.unlink(missing_ok=True)
                except (OSError, ValueError):
                    pass
            if self.output is not None:
                self.output.close()


def main(argv=None):
    args = web_parser().parse_args(argv)
    from backend.config import load_app_config
    from backend.logging_setup import TaskFormatter
    from backend.runtime import SOURCE_ROOT, runtime_path

    cfg = load_app_config(root=SOURCE_ROOT)
    # 先初始化真实终端，再绑定日志输出；不依赖 Colorama 的旧流或不完整控制码仿真。
    stream = sys.stderr
    ansi = supports_ansi(stream)
    color = use_color(cfg.logging.color, ansi)
    interactive = ansi and interactive_input(sys.stdin)
    output = ConsoleOutput(stream, interactive=interactive, color=color, ansi=ansi)
    handler = logging.StreamHandler(output)
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
