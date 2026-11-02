"""手动同步插件源码和图片输出配置；不访问网络、不启动或重载任何服务。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

NAME = 'astrbot_plugin_videolens'
SOURCE = Path(__file__).resolve().parent / NAME


def files():
    return sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix in {'.py', '.json', '.yaml'})


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inspect(root):
    """只比较文件及非私密的输出模式，不输出其他插件配置。"""
    root = Path(root).resolve()
    target = root / 'data/plugins' / NAME
    config_path = root / 'data/config' / (NAME + '_config.json')
    config = json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.is_file() else {}
    different = [p.name for p in files() if digest(p) != digest(target / p.name)]
    image = config.get('result_format', 'image') == 'image'
    testing = bool(config.get('self_test_prefix'))
    return {'code_matches': not different, 'different_files': different,
            'image_output': image, 'self_test_enabled': testing,
            'needs_update': bool(different) or not image or testing}


def install(root):
    root = Path(root).resolve()
    if not (root / 'main.py').is_file() or not (root / 'astrbot').is_dir():
        raise ValueError('该目录不是AstrBot源码根目录，应包含main.py和astrbot文件夹')
    # 校验现有配置后再写入，避免无效JSON造成部分同步。
    config_path = root / 'data/config' / (NAME + '_config.json')
    config = json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.is_file() else {}
    target = root / 'data/plugins' / NAME
    if SOURCE.resolve() == target.resolve():
        raise ValueError('源码和安装目录相同，不能覆盖自身')
    backup = root / 'data/plugin_data' / NAME / 'source_backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup.mkdir(parents=True, exist_ok=False)
    for source in files():
        old = target / source.name
        if old.is_file():
            shutil.copy2(old, backup / source.name)
    if config_path.is_file():
        shutil.copy2(config_path, backup / config_path.name)
    target.mkdir(parents=True, exist_ok=True)
    for source in files():
        shutil.copy2(source, target / source.name)
    config.update(result_format='image', self_test_prefix='')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    result = inspect(root)
    result['backup'] = str(backup)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--astrbot-root', required=True, help='AstrBot源码根目录')
    parser.add_argument('--check', action='store_true', help='只检查，不修改安装文件和设置')
    args = parser.parse_args()
    result = inspect(args.astrbot_root) if args.check else install(args.astrbot_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.check:
        print('文件已同步。请自行重载知影视频分析插件；本脚本不会启动、关闭或重载服务。')
