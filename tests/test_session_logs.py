"""启动日志递增、跨日重置和并发独占创建，不依赖真实网页或模型。"""
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.logging_setup import open_session_log


class SessionLogTests(unittest.TestCase):
    def test_daily_sequence_never_overwrites_previous_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"2026-10-28-1.log").write_text("原有日志",encoding="utf-8")
            (root/"2026-10-28-4.log").touch()
            (root/"zhiying.log").write_text("旧版日志",encoding="utf-8")
            with open_session_log(root,date="2026-10-28") as file:
                self.assertEqual(Path(file.name).name,"2026-10-28-5.log")
            with open_session_log(root,date="2026-10-29") as file:
                self.assertEqual(Path(file.name).name,"2026-10-29-1.log")
            self.assertEqual((root/"2026-10-28-1.log").read_text(encoding="utf-8"),"原有日志")

    def test_concurrent_startups_have_unique_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            def create(_):
                with open_session_log(tmp,date="2026-10-28") as file:
                    file.write("独立启动\n")
                    return Path(file.name).name
            with ThreadPoolExecutor(max_workers=6) as pool:
                names=list(pool.map(create,range(12)))
            self.assertEqual(len(set(names)),12)
            self.assertEqual(set(names),{f"2026-10-28-{i}.log" for i in range(1,13)})


if __name__=="__main__":unittest.main(verbosity=2)
