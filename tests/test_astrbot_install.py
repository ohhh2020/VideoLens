"""只在临时目录验证同步，不修改本机已安装的AstrBot。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.install_astrbot_plugin import install, inspect


class InstallTests(unittest.TestCase):
    def test_install_backup_config_preservation_and_version_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'main.py').write_text('# AstrBot', encoding='utf-8')
            (root / 'astrbot').mkdir()
            target = root / 'data/plugins/astrbot_plugin_videolens'
            target.mkdir(parents=True)
            (target / 'main.py').write_text('# old', encoding='utf-8')
            config = root / 'data/config/astrbot_plugin_videolens_config.json'
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({'allowed_groups': ['test-group'], 'result_format': 'text',
                                         'self_test_prefix': 'test-prefix', 'custom': 42}), encoding='utf-8')
            self.assertTrue(inspect(root)['needs_update'])
            result = install(root)
            self.assertFalse(result['needs_update'])
            self.assertEqual((Path(result['backup']) / 'main.py').read_text(encoding='utf-8'), '# old')
            saved = json.loads(config.read_text(encoding='utf-8'))
            self.assertEqual(saved['allowed_groups'], ['test-group'])
            self.assertEqual(saved['custom'], 42)
            self.assertEqual(saved['result_format'], 'image')
            self.assertEqual(saved['self_test_prefix'], '')
            self.assertTrue((target / 'rendering.py').is_file())
            # 检查操作不能更改源文件或设置。
            before = config.read_bytes()
            self.assertTrue(inspect(root)['code_matches'])
            self.assertEqual(config.read_bytes(), before)

    def test_invalid_target_and_invalid_config_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):install(root)
            (root / 'main.py').write_text('# AstrBot', encoding='utf-8')
            (root / 'astrbot').mkdir()
            config = root / 'data/config/astrbot_plugin_videolens_config.json'
            config.parent.mkdir(parents=True)
            config.write_text('broken', encoding='utf-8')
            with self.assertRaises(ValueError):install(root)
            self.assertEqual(config.read_text(encoding='utf-8'), 'broken')
            self.assertFalse((root / 'data/plugins').exists())


if __name__ == '__main__':unittest.main(verbosity=2)
