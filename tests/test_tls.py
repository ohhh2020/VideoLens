"""Conda Git Bash CA 路径错误的回归测试，不访问网络。"""
import os
from pathlib import Path
import ssl
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.runtime import repair_conda_ca_path


class CondaTlsTests(unittest.TestCase):
    def fixture(self, directory):
        prefix = Path(directory)
        (prefix / "conda-meta").mkdir()
        return prefix, str(prefix / "ssl/cacert.pem")

    @unittest.skipUnless(sys.platform == "win32", "Windows Conda 路径兼容")
    def test_repairs_only_missing_conda_default(self):
        import certifi
        with tempfile.TemporaryDirectory() as tmp:
            prefix, wrong = self.fixture(tmp)
            correct = prefix / "Library/ssl/cacert.pem"
            correct.parent.mkdir(parents=True)
            correct.write_bytes(Path(certifi.where()).read_bytes())
            with patch("backend.runtime.sys.prefix", str(prefix)), patch.dict(os.environ, {"SSL_CERT_FILE": wrong}):
                result = repair_conda_ca_path()
                self.assertEqual(result, correct)
                self.assertEqual(Path(os.environ["SSL_CERT_FILE"]), correct)
                context = ssl.create_default_context()
                self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(context.check_hostname)

    def test_preserves_valid_custom_ca(self):
        import certifi
        custom = certifi.where()
        with patch.dict(os.environ, {"SSL_CERT_FILE": custom}):
            self.assertIsNone(repair_conda_ca_path())
            self.assertEqual(os.environ["SSL_CERT_FILE"], custom)

    def test_does_not_hide_missing_custom_ca(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix, _ = self.fixture(tmp)
            custom = str(prefix / "company-custom.pem")
            with patch("backend.runtime.sys.prefix", str(prefix)), patch.dict(os.environ, {"SSL_CERT_FILE": custom}):
                self.assertIsNone(repair_conda_ca_path())
                self.assertEqual(os.environ["SSL_CERT_FILE"], custom)

    @unittest.skipUnless(sys.platform == "win32", "Windows Conda 路径兼容")
    def test_certifi_when_conda_bundle_missing(self):
        import certifi
        with tempfile.TemporaryDirectory() as tmp:
            prefix, wrong = self.fixture(tmp)
            with patch("backend.runtime.sys.prefix", str(prefix)), patch.dict(os.environ, {"SSL_CERT_FILE": wrong}):
                self.assertEqual(repair_conda_ca_path(), Path(certifi.where()))
                self.assertEqual(Path(os.environ["SSL_CERT_FILE"]), Path(certifi.where()))

    @unittest.skipUnless(sys.platform == "win32", "Windows Conda 路径兼容")
    def test_invalid_bundle_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix, wrong = self.fixture(tmp)
            correct = prefix / "Library/ssl/cacert.pem"
            correct.parent.mkdir(parents=True)
            correct.write_text("broken certificate", encoding="utf-8")
            with patch("backend.runtime.sys.prefix", str(prefix)), patch.dict(os.environ, {"SSL_CERT_FILE": wrong}):
                with self.assertRaises(ssl.SSLError):
                    repair_conda_ca_path()
                self.assertEqual(os.environ["SSL_CERT_FILE"], wrong)


if __name__ == "__main__":
    unittest.main()
