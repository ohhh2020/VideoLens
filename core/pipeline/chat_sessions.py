"""多轮问答会话（每个视频一段对话，内存保存最近 N 轮）。

为什么需要独立这一层
--------------------
单轮问答是无状态的：用户问"那第二个进球呢"，检索器拿到的查询就是这七个
字 —— 召回自然是垃圾。C 端对话的体感来自两件事：

1. **查询改写**：把"那第二个进球呢"结合上一轮扩写成独立完整的
   "曼联对科莫比赛中第二个进球"再去检索；
2. **生成上下文**：把最近几轮问答作为 messages 前缀传给 LLM，
   模型才知道"他"指的是谁。

存储策略：内存（进程级、LRU 上限），不进 SQLite —— 对话历史是易失的
交互状态，不是分析产物；进程重启丢对话是可接受的 C 端惯例
（ChatGPT 侧栏刷新同理）。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

__all__ = ["ChatSession", "ChatSessionStore", "Turn"]


@dataclass(slots=True)
class Turn:
    """一轮对话（改写后的查询 + 回答摘要）。"""

    question: str
    answer: str
    rewritten: str = ""
    elapsed_ms: int = 0

    def as_messages(self) -> list[dict[str, str]]:
        """转成 LLM messages 前缀（截断到摘要长度，防上下文爆炸）。"""
        return [
            {"role": "user", "content": self.question[:400]},
            {"role": "assistant", "content": self.answer[:800]},
        ]


@dataclass(slots=True)
class ChatSession:
    """一个视频的一段对话。线程安全由 Store 的锁保证。"""

    video_id: str
    turns: list[Turn] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def history_messages(self, window: int) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for turn in self.turns[-max(0, window):]:
            out.extend(turn.as_messages())
        return out

    def last_answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""


class ChatSessionStore:
    """LRU 会话表（默认 64 个视频 × 8 轮，内存占用可忽略）。"""

    def __init__(self, *, max_sessions: int = 64, max_turns: int = 8) -> None:
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max(1, max_sessions)
        self._max_turns = max(1, max_turns)

    def history(self, video_id: str, window: int | None = None) -> list[dict[str, str]]:
        """最近 N 轮的 messages 前缀；无会话返回空表。"""
        with self._lock:
            session = self._sessions.get(video_id)
            if session is None:
                return []
            self._sessions.move_to_end(video_id)
            return session.history_messages(self._max_turns if window is None else window)

    def last_answer(self, video_id: str) -> str:
        with self._lock:
            session = self._sessions.get(video_id)
            return session.last_answer() if session else ""

    def remember(self, video_id: str, turn: Turn) -> None:
        with self._lock:
            session = self._sessions.get(video_id)
            if session is None:
                session = ChatSession(video_id=video_id)
                self._sessions[video_id] = session
            session.turns.append(turn)
            if len(session.turns) > self._max_turns:
                session.turns = session.turns[-self._max_turns:]
            session.updated_at = time.time()
            self._sessions.move_to_end(video_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, video_id: str | None = None) -> None:
        with self._lock:
            if video_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(video_id, None)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sessions": len(self._sessions),
                "turns": sum(len(s.turns) for s in self._sessions.values()),
            }


#: 进程级单例（HTTP 与 /chat 页共用）。
_STORE: ChatSessionStore | None = None
_STORE_LOCK = threading.Lock()


def get_chat_store() -> ChatSessionStore:
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = ChatSessionStore()
        return _STORE
