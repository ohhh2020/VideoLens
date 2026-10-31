"""同一视频的提交与删除互斥，避免删除过程中又接受新的追问。"""
import threading

_locks = {}
_guard = threading.Lock()


def video_guard(video_id):
    with _guard:
        return _locks.setdefault(video_id, threading.RLock())
