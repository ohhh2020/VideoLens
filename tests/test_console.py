"""控制台真实子进程、管道关闭、正常重启及开发监视回归；不加载模型。"""

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

try:
    import pyte
except ImportError:
    pyte = None
from unittest.mock import patch
from contextlib import contextmanager, closing
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.web.console import ConsoleOutput, ConsoleSupervisor, start_control_reader, send_console_command, terminal_size


class ConsoleTests(unittest.TestCase):
    def test_child_non_utf8_bytes_do_not_kill_log_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            stream=io.StringIO()
            output=ConsoleOutput(stream,ansi=False)
            command=[sys.executable,'-c',"import sys;sys.stdout.buffer.write(b'bad \\xff\\nafter\\n');sys.stdout.flush()"]
            supervisor=ConsoleSupervisor(command,stream=io.StringIO(),output=output,log_directory=directory)
            self.assertEqual(supervisor.run(),0)
            self.assertIn('after',stream.getvalue())
            self.assertIn('\ufffd',stream.getvalue())

    def test_yellow_prompt_clear_and_session_logs_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            terminal=io.StringIO()
            output=ConsoleOutput(terminal,interactive=True,color=True)
            output.start_session(tmp)
            output.prompt()
            output.write("\033[32m第一条日志\033[0m\n")
            path=Path(output.file.name)
            before=path.read_bytes()
            supervisor=ConsoleSupervisor([],output=output)
            with patch("backend.web.console.logger") as logger:
                supervisor._handle("clear")
                logger.info.assert_not_called()
                logger.warning.assert_not_called()
            self.assertEqual(path.read_bytes(),before,"清屏和提示符不落盘")
            self.assertEqual(path.read_text(encoding="utf-8"),"第一条日志\n")
            self.assertIn("\033[33m<< \033[0m",terminal.getvalue())
            self.assertIn("\033[2J\033[3J\033[H",terminal.getvalue())
            output.start_session(tmp)
            output.write("第二次启动\n")
            second=Path(output.file.name)
            output.close()
            self.assertNotEqual(path,second)
            self.assertTrue(path.name.endswith("-1.log"))
            self.assertTrue(second.name.endswith("-2.log"))
            self.assertEqual(path.read_text(encoding="utf-8"),"第一条日志\n")
            self.assertEqual(second.read_text(encoding="utf-8"),"第二次启动\n")

    def test_git_bash_size_uses_explicit_pty_instead_of_native_controlling_tty(self):
        with patch.dict(os.environ, {"VIDEOLENS_TERMINAL_STTY": "C:/Git/usr/bin/stty.exe",
                                    "VIDEOLENS_TERMINAL_TTY": "/dev/pty3"}):
            with patch("backend.web.console.subprocess.run", return_value=SimpleNamespace(
                stdout="42 120\n", returncode=0)) as run:
                self.assertEqual(terminal_size(), os.terminal_size((120, 42)))
                self.assertEqual(run.call_args.args[0],
                                 ["C:/Git/usr/bin/stty.exe", "-F", "/dev/pty3", "size"])

    def test_logs_never_reprint_prompt_or_add_empty_prompt_lines(self):
        terminal = io.StringIO()
        output = ConsoleOutput(terminal, interactive=True, color=True,
                               size_provider=lambda: os.terminal_size((80, 12)))
        output.prompt()
        before = terminal.getvalue()
        for number in range(100):
            output.write(f"日志 {number}\n")
        self.assertEqual(terminal.getvalue().count("<< "), 1)
        self.assertNotIn("<<", terminal.getvalue()[len(before):])
        output.input_finished()
        output.prompt()
        self.assertEqual(terminal.getvalue().count("<< "), 2,
                         "只有下一次输入开始才绘制提示符")
        output.close()
        self.assertTrue(terminal.getvalue().endswith("\033[r\033[12;1H\033[2K"))

    @unittest.skipIf(pyte is None, "安装开发测试依赖 pyte 后可验证终端屏幕")
    def test_terminal_screen_pins_prompt_and_preserves_typed_text(self):
        screen = pyte.HistoryScreen(80, 8, history=1000)
        emulator = pyte.Stream(screen)
        terminal = io.StringIO()
        output = ConsoleOutput(terminal, interactive=True, color=True,
                               size_provider=lambda: os.terminal_size((80, 8)))
        position = 0

        def render():
            nonlocal position
            text = terminal.getvalue()
            emulator.feed(text[position:])
            position = len(text)

        with tempfile.TemporaryDirectory() as tmp, closing(output):
            output.start_session(tmp)
            output.prompt()
            render()
            emulator.feed("sta")  # 模拟终端回显尚未提交的命令。
            for number in range(100):
                output.write(f"\033[37m日志 {number}：测试内容\033[0m\n")
                render()
            self.assertEqual(screen.display[-1].rstrip(), "<< sta")
            self.assertEqual((screen.cursor.x, screen.cursor.y), (6, 7))
            self.assertTrue(any("日志 99" in line for line in screen.display[:-1]))
            self.assertEqual(sum("<<" in line for line in screen.display), 1)
            for line in screen.history.top:
                self.assertNotIn("<<", "".join(cell.data for cell in line.values()))
            self.assertEqual(screen.buffer[7][0].fg, "brown")
            path = Path(output.file.name)
            output.clear()
            render()
            self.assertEqual(screen.display[-1].rstrip(), "<< ".rstrip())
            self.assertTrue(all(not line.strip() for line in screen.display[:-1]))
            output.write("清屏后的日志\n")
            render()
            self.assertTrue(any("清屏后的日志" in line for line in screen.display[:-1]))
            output.close()
            self.assertNotIn("<<", path.read_text(encoding="utf-8"))
            self.assertNotIn("\033", path.read_text(encoding="utf-8"))

    @unittest.skipIf(pyte is None, "安装开发测试依赖 pyte 后可验证终端屏幕")
    def test_terminal_resize_moves_input_and_exit_restores_scroll_region(self):
        screen = pyte.Screen(80, 8)
        emulator = pyte.Stream(screen)
        terminal = io.StringIO()
        size = os.terminal_size((80, 8))
        output = ConsoleOutput(terminal, interactive=True, size_provider=lambda: size)
        output.prompt()
        emulator.feed(terminal.getvalue())
        position = terminal.tell()
        screen.resize(lines=12, columns=80)
        size = os.terminal_size((80, 12))
        output.refresh_size(force=True)
        emulator.feed(terminal.getvalue()[position:])
        self.assertEqual(screen.display[-1].rstrip(), "<<")
        self.assertEqual(sum("<<" in line for line in screen.display), 1)
        position = terminal.tell()
        output.close()
        emulator.feed(terminal.getvalue()[position:])
        self.assertIsNone(screen.margins)

    def test_control_pipe_stop_and_eof(self):
        for text in ('invalid\n{"command":"stop"}\n', ''):
            server=SimpleNamespace(should_exit=False)
            thread=start_control_reader(server,io.StringIO(text))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(server.should_exit)

    def test_startup_failure_is_returned_without_restart_loop(self):
        supervisor=ConsoleSupervisor([sys.executable,"-c","raise SystemExit(7)"],stream=io.StringIO())
        self.assertEqual(supervisor.run(),7)

    @contextmanager
    def fixture(self, watch=False, capture=False):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            events=work/"events.jsonl"
            source=work/"src"
            source.mkdir()
            watched=source/"change.py"
            watched.write_text("value=1",encoding="utf-8")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0))
                port=sock.getsockname()[1]
            child=work/"fixture_server.py"
            child.write_text(f'''
import asyncio,json,logging,os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from backend.web.console import start_control_reader
logging.basicConfig(level=logging.INFO,format="%(message)s")
def event(kind):
 with Path({str(events)!r}).open("a",encoding="utf-8") as stream:
  stream.write(json.dumps({{"kind":kind,"pid":os.getpid()}})+"\\n")
@asynccontextmanager
async def lifespan(app):
 event("started")
 try: yield
 finally:
  await asyncio.sleep(0.1)
  event("closed")
app=FastAPI(lifespan=lifespan)
@app.get("/internal/health")
def health():return {{"status":"ok","pid":os.getpid(),"jobs_running":0,"model_devices":{{"fixture":"cpu"}}}}
server=uvicorn.Server(uvicorn.Config(app,host="127.0.0.1",port={port},log_config=None,access_log=False))
start_control_reader(server)
# 阻塞输入线程存在时导入原生扩展，回归 Windows CRT stdin 锁与 OpenBLAS 的互锁。
import numpy
server.run()
''',encoding="utf-8")
            wrapper=work/"fixture_console.py"
            wrapper.write_text(f'''
import logging,sys
from backend.web.console import ConsoleOutput, ConsoleSupervisor
output=ConsoleOutput(sys.stderr) if {capture!r} else None
logging.basicConfig(level=logging.INFO,format="%(message)s",stream=output or sys.stderr)
raise SystemExit(ConsoleSupervisor([sys.executable,{str(child)!r}],port={port},watch_paths={[str(source)] if watch else []!r},control_state_path={str(work/'control.json')!r},output=output,log_directory={str(work/'logs')!r} if output else None).run())
''',encoding="utf-8")
            env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED="1",PYTHONUTF8="1")
            process=subprocess.Popen([sys.executable,str(wrapper)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",env=env)
            try:
                yield process,events,port,watched
            finally:
                if process.poll() is None:
                    process.stdin.write("stop\n")
                    process.stdin.flush()
                    try:
                        process.communicate(timeout=20)
                    except subprocess.TimeoutExpired:
                        # 仅清理本测试创建的隔离父进程；其管道 EOF 会关闭测试子服务。
                        process.terminate()
                        process.communicate(timeout=10)
                for stream in (process.stdin,process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()

    def wait_started(self,process,events,port,count):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            if process.poll() is not None:
                self.fail(process.communicate()[0])
            rows=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
            starts=[row for row in rows if row["kind"]=="started"]
            if len(starts)>=count:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/internal/health",timeout=.3) as response:
                        if json.load(response)["pid"]==starts[-1]["pid"]:
                            return rows
                except OSError:
                    pass
            time.sleep(.05)
        self.fail("隔离控制台服务未在期限内就绪")

    def test_real_server_help_status_restart_and_stop(self):
        with self.fixture() as (process,events,port,_):
            rows=self.wait_started(process,events,port,1)
            first=rows[0]["pid"]
            process.stdin.write("help\nstatus\nunsupported-command\n")
            process.stdin.flush()
            self.assertTrue(send_console_command("status",state_path=events.parent/"control.json")["accepted"])
            accepted=send_console_command("restart",state_path=events.parent/"control.json")
            self.assertTrue(accepted["accepted"])
            rows=self.wait_started(process,events,port,2)
            self.assertEqual(rows[1],{"kind":"closed","pid":first})
            self.assertEqual(rows[2]["kind"],"started")
            self.assertNotEqual(rows[2]["pid"],first)
            process.stdin.write("stop\n")
            process.stdin.flush()
            output,_=process.communicate(timeout=20)
            self.assertEqual(process.returncode,0,output)
            self.assertIn("控制台指令",output)
            self.assertIn("服务状态=ok",output)
            self.assertIn("未知指令",output)
            self.assertIn("旧服务已关闭",output)
            self.assertEqual(json.loads(events.read_text().splitlines()[-1])["kind"],"closed")

    def test_reload_waits_for_old_service_shutdown(self):
        with self.fixture(watch=True) as (process,events,port,watched):
            rows=self.wait_started(process,events,port,1)
            first=rows[0]["pid"]
            watched.write_text("value=2",encoding="utf-8")
            rows=self.wait_started(process,events,port,2)
            self.assertEqual(rows[1],{"kind":"closed","pid":first})
            process.stdin.write("stop\n");process.stdin.flush()
            output,_=process.communicate(timeout=20)
            self.assertEqual(process.returncode,0,output)
            self.assertIn("检测到前后端代码变更",output)

    def test_real_restart_creates_another_log_and_clear_is_not_logged(self):
        with self.fixture(capture=True) as (process,events,port,_):
            first=self.wait_started(process,events,port,1)[0]["pid"]
            files=list((events.parent/"logs").glob("*.log"))
            self.assertEqual(len(files),1)
            before=files[0].read_bytes()
            self.assertTrue(send_console_command("clear",state_path=events.parent/"control.json")["accepted"])
            # 确保 clear 已被主循环执行，再重启；清屏控制码只出现在终端输出。
            process.stdin.write("restart\n");process.stdin.flush()
            rows=self.wait_started(process,events,port,2)
            second=rows[2]["pid"]
            process.stdin.write("stop\n");process.stdin.flush()
            terminal,_=process.communicate(timeout=20)
            files=sorted((events.parent/"logs").glob("*.log"),key=lambda path:int(path.stem.rsplit("-",1)[1]))
            self.assertEqual(len(files),2)
            first_log=files[0].read_text(encoding="utf-8")
            second_log=files[1].read_text(encoding="utf-8")
            self.assertTrue(files[0].read_bytes().startswith(before))
            self.assertIn(f"Started server process [{first}]",first_log)
            self.assertIn(f"Finished server process [{first}]",first_log)
            self.assertIn(f"Started server process [{second}]",second_log)
            self.assertIn(f"Finished server process [{second}]",second_log)
            self.assertNotIn("\033",terminal,"重定向输出不应包含清屏或光标控制码")
            self.assertIn("当前输出窗口不支持清屏",terminal)
            self.assertNotIn("当前输出窗口不支持清屏",first_log+second_log)
            self.assertNotIn("\033[",first_log+second_log)
            self.assertNotIn("<< ",first_log+second_log)


if __name__=="__main__":
    unittest.main(verbosity=2)
