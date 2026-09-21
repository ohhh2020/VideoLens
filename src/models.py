from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Keyframe:
    path: str
    timestamp: float
    scene_score: float = 0.0


@dataclass
class SceneCaption:
    timestamp: float
    description: str
    objects: list[str] | None = None
    action: str = ""


def to_serializable(value):
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value

