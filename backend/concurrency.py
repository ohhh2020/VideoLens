"""有界线程池工具；每次提交复制独立上下文，供日志关联任务。"""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


def submit(pool, fn, *args, **kwargs):
    return pool.submit(copy_context().run, fn, *args, **kwargs)


def parallel_pair(first, second, *, enabled=True, prefix="local"):
    if not enabled:
        return first(), second()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix=prefix) as pool:
        a = submit(pool, first)
        b = submit(pool, second)
        # with 会等待两个 worker 退出，调用方清理索引时不会再有后台写入。
        return a.result(), b.result()
