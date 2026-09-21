from __future__ import annotations

import json
from pathlib import Path

from .models import to_serializable


def save_json(path: Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(to_serializable(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_text(path: Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def save_result(output_dir: Path, result: dict) -> None:
    save_json(output_dir / "result.json", result)
    save_text(output_dir / "summary.md", result.get("summary", {}).get("markdown", ""))
    save_json(output_dir / "transcript.json", result.get("transcript", []))
    save_json(output_dir / "captions.json", result.get("captions", []))

