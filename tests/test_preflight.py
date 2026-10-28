"""缺少必需模型必须返回 BROKEN。"""

import sys, unittest, io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import preflight


class PreflightTests(unittest.TestCase):
    def test_missing_models_fail(self):
        with patch(
            "scripts.preflight.ModelManager", side_effect=RuntimeError("必需模型缺失")
        ):
            out = io.StringIO()
            with redirect_stdout(out):
                code = preflight.main(["--json"])
            self.assertEqual(code, 2)
            self.assertIn("BROKEN", out.getvalue())
            self.assertIn("必需模型缺失", out.getvalue())


if __name__ == "__main__":
    unittest.main()
