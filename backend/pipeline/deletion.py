"""联动删除的附件清理，只清理项目上传副本并保护其他视频仍引用的文件。"""
from pathlib import Path
import os

from backend.pipeline.helpers import _unlink_owned


def file_key(path):
    value=Path(path)
    return os.path.normcase(str(value.absolute() if value.is_symlink() else value.resolve()))


def file_keys(path):
    value=Path(path)
    keys={file_key(value)}
    if value.is_symlink():
        keys.add(os.path.normcase(str(value.resolve())))
    return keys


def attachment_paths(repo, *, video_id=None, conversation_id=None):
    if video_id is not None:
        condition, value = 'conversation_id IN (SELECT id FROM conversations WHERE video_id=?)', video_id
    else:
        condition, value = 'conversation_id=?', conversation_id
    return [row['path'] for table in ('uploaded_videos', 'upload_sessions')
            for row in repo._query(f'SELECT path FROM {table} WHERE {condition}', (value,))]


def referenced_files(repo):
    return {key
            for row in repo._query('SELECT local_path AS path FROM videos WHERE local_path<>\'\' '
                                   'UNION SELECT path FROM uploaded_videos UNION SELECT path FROM upload_sessions') if row['path']
            for key in file_keys(row['path'])}


def purge_uploads(repo, config, paths):
    references = referenced_files(repo)
    root = Path(config.path('data')) / 'uploads'
    return sum(_unlink_owned(Path(path), [root]) for path in set(paths)
               if file_key(path) not in references)
