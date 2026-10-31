"""真实管道、Windows VT 与 MSYS PTY 检测，防止 ANSI 控制码泄露到普通输出。"""
import ctypes
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import terminal
from backend.web.console import ConsoleOutput
from backend.web.console import native_input_lines


class TerminalTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='win32','Windows原生输入')
    def test_native_input_poll_does_not_block_log_output_and_supports_editing(self):
        edited=[]
        keys=iter('stax\btus\r\x03')
        with patch('msvcrt.kbhit',return_value=True),patch('msvcrt.getwch',side_effect=lambda:next(keys)):
            self.assertEqual(list(native_input_lines(edited.append)),['status','stop'])
        self.assertIn('sta',edited)

    def test_terminal_mode_reset_falls_back_without_leaking_escape_codes(self):
        stream=io.StringIO()
        output=ConsoleOutput(stream,interactive=True,color=True,ansi=True)
        output.native_output=True
        with patch('backend.web.console.supports_ansi',return_value=False):
            output.write('\033[32m就绪\033[0m\n')
        self.assertEqual(stream.getvalue(),'就绪\n')
        self.assertFalse(output.ansi)
        self.assertFalse(output.interactive)

    def test_inherited_bash_variables_do_not_enable_redirected_terminal(self):
        with patch.dict(os.environ, {'MSYSTEM':'UCRT64','TERM':'xterm-256color'}), \
             patch('backend.terminal.sys.platform','win32'), \
             patch('backend.terminal._windows_console_vt',return_value=False), \
             patch('backend.terminal._windows_pty_name',return_value=r'\Device\NamedPipe\ordinary-output'):
            self.assertFalse(terminal.supports_ansi(io.StringIO()))
            self.assertFalse(terminal.interactive_input(io.StringIO()))

    def test_verified_msys_pipe_is_supported_without_environment_guessing(self):
        with patch('backend.terminal.sys.platform','win32'), \
             patch('backend.terminal._windows_console_vt',return_value=False), \
             patch('backend.terminal._windows_pty_name',return_value=r'\msys-1888ae32e00d56aa-pty3-to-master'), \
             patch.dict(os.environ,{'TERM':'xterm-256color'}):
            self.assertTrue(terminal.supports_ansi(io.StringIO()))
            self.assertTrue(terminal.interactive_input(io.StringIO()))

    @unittest.skipUnless(sys.platform=='win32','Windows原生控制台API')
    def test_native_console_enables_and_verifies_vt_mode(self):
        kernel=SimpleNamespace(GetConsoleMode=Mock(),SetConsoleMode=Mock(return_value=1))
        values=iter((3,7))
        def get(handle, result):
            ctypes.cast(result,ctypes.POINTER(ctypes.c_ulong))[0]=next(values)
            return 1
        kernel.GetConsoleMode.side_effect=get
        with patch('ctypes.WinDLL',return_value=kernel),patch('msvcrt.get_osfhandle',return_value=42):
            self.assertTrue(terminal._windows_console_vt(SimpleNamespace(fileno=lambda:2)))
        kernel.SetConsoleMode.assert_called_once_with(42,7)
        kernel.GetConsoleMode.side_effect=lambda *a: 0
        kernel.SetConsoleMode.reset_mock()
        with patch('ctypes.WinDLL',return_value=kernel),patch('msvcrt.get_osfhandle',return_value=42):
            self.assertFalse(terminal._windows_console_vt(SimpleNamespace(fileno=lambda:2)))
        kernel.SetConsoleMode.assert_not_called()
        # 有句柄但不支持VT，以及设置返回成功却未真正生效，都必须使用纯文本。
        def plain_mode(handle,result):
            ctypes.cast(result,ctypes.POINTER(ctypes.c_ulong))[0]=3
            return 1
        for accepted in (0,1):
            kernel.GetConsoleMode.side_effect=plain_mode
            kernel.SetConsoleMode.return_value=accepted
            with patch('ctypes.WinDLL',return_value=kernel),patch('msvcrt.get_osfhandle',return_value=42):
                self.assertFalse(terminal._windows_console_vt(SimpleNamespace(fileno=lambda:2)))

    def test_plain_output_strips_child_controls_and_never_paints_prompt_or_clear(self):
        stream=io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            output=ConsoleOutput(stream,interactive=True,color=True,ansi=False)
            output.start_session(directory)
            output.prompt()
            output.write('\0337\033[32m服务就绪\033[0m\0338\n')
            path=Path(output.file.name)
            before=path.read_bytes()
            output.clear();output.close()
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(path.read_text(encoding='utf-8'),'服务就绪\n')
            self.assertNotIn('\033',stream.getvalue())
            self.assertNotIn('<<',stream.getvalue())

    def test_no_color_keeps_supported_cursor_layout(self):
        with patch.dict(os.environ,{'NO_COLOR':'1'}):
            self.assertFalse(terminal.use_color('auto',True))
        self.assertFalse(terminal.use_color('always',False))
        stream=io.StringIO()
        output=ConsoleOutput(stream,interactive=True,color=False,ansi=True)
        output.prompt();output.write('\033[32m就绪\033[0m\n');output.close()
        self.assertIn('<< ',stream.getvalue())
        self.assertNotIn('\033[32m',stream.getvalue())

    def test_real_piped_logging_stays_plain_with_inherited_bash_environment(self):
        script='''
from backend.config import load_app_config
from backend.logging_setup import configure_logging
import logging,sys
cfg=load_app_config(root=sys.argv[1],overrides={'logging.color':sys.argv[2]});configure_logging(cfg)
logging.info('控制台已就绪')
'''
        env=dict(os.environ,MSYSTEM='UCRT64',TERM='xterm-256color',PYTHONUTF8='1',PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        env.pop('VIDEOLENS_MANAGED_LOG',None);env.pop(terminal.CONSOLE_COLOR_ENV,None)
        for color in ('auto','always'):
            with tempfile.TemporaryDirectory() as directory:
                result=subprocess.run([sys.executable,'-c',script,directory,color],env=env,stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,text=True,encoding='utf-8',timeout=20)
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertIn('控制台已就绪',result.stdout)
            self.assertNotIn('\033',result.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
