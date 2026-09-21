from __future__ import annotations

from pathlib import Path

from .models import TranscriptSegment


class TranscriptionError(RuntimeError):
    pass


class Transcriber:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "缺少 faster-whisper，请先执行：pip install -r requirements.txt"
                ) from exc

            self._model = WhisperModel(
                self.settings.asr_model,
                device=self.settings.asr_device,
                compute_type=self.settings.asr_compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if audio_path is None or not Path(audio_path).exists():
            return []

        model = self._get_model()
        segments, _info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
        )

        results: list[TranscriptSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                results.append(
                    TranscriptSegment(
                        start=round(float(segment.start), 2),
                        end=round(float(segment.end), 2),
                        text=text,
                    )
                )
        return results

