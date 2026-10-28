"""依赖清单兼容边界和已激活 Conda 环境的启动行为。"""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EnvironmentTests(unittest.TestCase):
    def test_requirements_keep_decoder_compatibility(self):
        from packaging.requirements import Requirement
        from packaging.version import Version
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        specs = {}
        for raw in lines:
            line = raw.strip()
            if line and not line.startswith("#"):
                requirement = Requirement(line)
                specs[requirement.name.lower().replace("_", "-")] = requirement
        self.assertIn(Version("16.1.0"), specs["av"].specifier)
        self.assertNotIn(Version("19.0.0"), specs["av"].specifier)
        self.assertIn(Version("1.2.1"), specs["faster-whisper"].specifier)
        for name in ("torch", "transformers", "sentencepiece", "faster-whisper", "chromadb"):
            self.assertIn(name, specs)

    @unittest.skipUnless(os.name == "nt", "Windows 批处理验证")
    def test_launcher_requires_active_environment(self):
        env = dict(os.environ)
        env.pop("CONDA_PREFIX", None)
        result = subprocess.run(["cmd.exe", "/d", "/c", "start.bat"], cwd=ROOT,
                                env=env, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("conda activate", result.stdout.decode("utf-8", "replace"))

    @unittest.skipUnless(os.name == "nt", "Windows 批处理验证")
    def test_help_does_not_require_environment(self):
        env = dict(os.environ)
        env.pop("CONDA_PREFIX", None)
        result = subprocess.run(["cmd.exe", "/d", "/c", "start.bat --help"], cwd=ROOT,
                                env=env, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--port", result.stdout.decode("utf-8", "replace"))

    @unittest.skipUnless(os.name == "nt", "Windows 批处理验证")
    def test_invalid_active_prefix_fails(self):
        env = dict(os.environ)
        env["CONDA_PREFIX"] = str(ROOT / "missing-conda-environment")
        result = subprocess.run(["cmd.exe", "/d", "/c", "start.bat"], cwd=ROOT,
                                env=env, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("python.exe", result.stdout.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()
