"""并发边界、流式时序、任务日志和计时持久化回归；不依赖云端凭据。"""

from __future__ import annotations

import io
import logging
import re
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.concurrency import parallel_pair, submit
from backend.config import load_app_config
from backend.logging_setup import TaskFormatter, ProgressAccessFilter, log_task
from backend.models.asr import AsrConfig
from backend.models.embed import EmbeddingService
from backend.models.llm import (
    LlmError,
    LlmGateway,
    LlmMessage,
    LlmResponse,
    OpenAICompatibleBackend,
)
from backend.pipeline.domain import (
    AnalysisReport,
    AnalyzeOptions,
    Stage,
    StageReport,
    _INFERENCE_LOCK,
)
from backend.pipeline.helpers import _group_by_scene
from backend.pipeline.orchestrator import Orchestrator
from backend.pipeline.stages.vision import VisionStages


class PerformanceTests(unittest.TestCase):
    def test_independent_towers_really_overlap(self):
        barrier = threading.Barrier(2, timeout=2)

        def run(*args):
            barrier.wait()
            return [[1.0, 0.0]]

        text = SimpleNamespace(name="bge-m3", dim=2, embed_texts=run)
        image = SimpleNamespace(name="siglip2-base", dim=2, embed_images=run)
        service = EmbeddingService(text, image)
        a, b = parallel_pair(
            lambda: service.embed_texts(["文字"]),
            lambda: service.embed_images(["图像"]),
        )
        self.assertEqual(a, b)

    def test_same_tower_remains_serial(self):
        active = maximum = 0
        guard = threading.Lock()

        def run(*args):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with guard:
                active -= 1
            return [[1.0, 0.0]]

        tower = SimpleNamespace(dim=2, embed_texts=run)
        service = EmbeddingService(tower, tower)
        parallel_pair(
            lambda: service.embed_query("甲"), lambda: service.embed_query("乙")
        )
        self.assertEqual(maximum, 1)

    def test_cloud_stream_does_not_block_other_request(self):
        class Backend:
            name = "cloud"
            available = lambda self: True

            def complete(self, *args, **kwargs):
                return LlmResponse("完成", "cloud")

            def stream(self, *args, **kwargs):
                yield "首字"
                yield "后续"

        gateway = LlmGateway([Backend()], max_concurrency=2)
        stream = gateway.stream(["问题"])
        self.assertEqual(next(stream), "首字")
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(
                    pool.submit(gateway.complete, ["另一个问题"])
                    .result(timeout=1)
                    .text,
                    "完成",
                )
        finally:
            stream.close()

    def test_cloud_ignores_local_inference_lock(self):
        cfg = load_app_config(
            root=ROOT,
            ignore_user_settings=True,
            overrides={"limits.inference_serial": True},
        )
        orch = Orchestrator(repository=Mock(), config=cfg)
        _INFERENCE_LOCK.acquire()
        try:
            with orch._inference_slot("云端", None, local=False):
                pass
        finally:
            _INFERENCE_LOCK.release()

    def test_first_event_precedes_retrieval(self):
        orch = Orchestrator(repository=Mock())
        stream = orch.ask_stream("video", "问题")
        self.assertEqual(next(stream)[0], "status")
        orch.repository.find_video.assert_not_called()
        stream.close()

    def test_context_rewrite_does_not_call_cloud(self):
        llm = Mock()
        orch = Orchestrator(repository=Mock(), llm=llm)
        rewritten = orch._rewrite_query(
            "它有什么缺点？", [{"role": "user", "content": "CPU 混合推理"}]
        )
        self.assertIn("CPU", rewritten)
        self.assertIn("缺点", rewritten)
        llm.complete.assert_not_called()

    def test_cloud_rewrite_timeout_is_passed(self):
        cfg = load_app_config(
            root=ROOT,
            ignore_user_settings=True,
            overrides={
                "qa.query_rewrite_mode": "cloud",
                "qa.query_rewrite_timeout_s": 1.5,
            },
        )
        llm = Mock()
        llm.complete.return_value = SimpleNamespace(text="CPU 混合推理的缺点")
        orch = Orchestrator(repository=Mock(), llm=llm, config=cfg)
        orch._rewrite_query("有什么缺点", [{"role": "user", "content": "CPU 混合推理"}])
        self.assertEqual(llm.complete.call_args.kwargs["timeout"], 1.5)

    def test_log_format_colors_context_and_plain_file(self):
        fmt = "[%(asctime)s][%(levelname)s][%(task)s]: %(message)s"
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(TaskFormatter(fmt, "%H:%M:%S", color=False))
        logger = logging.getLogger("test.task.log")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            with log_task("video:123"):
                with ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="worker"
                ) as pool:
                    submit(pool, logger.warning, "测试消息").result()
            value = handler.stream.getvalue()
            self.assertIn("[WARNING][worker_0]: 测试消息", value)
            self.assertNotIn("/video:123", value)
            self.assertNotIn("\033[", value)
            for level in (logging.INFO, logging.WARNING, logging.ERROR):
                record = logging.LogRecord("test", level, "", 0, "消息", (), None)
                colored = TaskFormatter(fmt, color=True).format(record)
                self.assertIn(
                    f"{TaskFormatter.COLORS[level]}{record.levelname}{TaskFormatter.WHITE}",
                    colored,
                )
                self.assertIn(f"{TaskFormatter.WHITE}]", colored)
                self.assertNotIn("\033[", record.levelname)
        finally:
            logger.removeHandler(handler)

    def test_log_highlights_only_status_words_and_preserves_shared_record(self):
        fmt = "[%(levelname)s][%(task)s] : %(message)s"
        record = logging.LogRecord(
            "test", logging.WARNING, "", 0,
            "阶段完成 video=%s；请求超时，等待重试；未完成、未通过；普通正文",
            ("123",), None,
        )
        plain = TaskFormatter(fmt)
        expected = plain.format(record)
        colored = TaskFormatter(fmt, color=True).format(record)
        for word, color in (
            ("完成", TaskFormatter.GREEN),
            ("超时", TaskFormatter.RED),
            ("重试", TaskFormatter.ORANGE),
            ("未完成", TaskFormatter.ORANGE),
            ("未通过", TaskFormatter.RED),
        ):
            self.assertIn(f"{color}{word}{TaskFormatter.WHITE}", colored)
        self.assertIn(f"{TaskFormatter.WHITE} video=123", colored)
        self.assertIn("；普通正文\033[0m", colored)
        self.assertNotIn(f"未{TaskFormatter.GREEN}", colored)
        self.assertEqual(re.sub(r"\x1b\[[0-9;]*m", "", colored), expected)
        self.assertEqual(plain.format(record), expected)
        self.assertNotIn("\033[", record.getMessage())

    def test_log_default_shape_and_only_successful_progress_access_is_quiet(self):
        cfg = load_app_config(root=ROOT, ignore_user_settings=True)
        record = logging.LogRecord("zhiying.pipeline",logging.INFO,"",0,"问答检索 %.3fs",(1.372,),None)
        record.threadName = "asyncio_0"
        with log_task("qa:private-video/session"):
            value = TaskFormatter(cfg.logging.format,cfg.logging.date_format).format(record)
        self.assertRegex(value,r"^\[.+\]\[INFO\]\[asyncio_0\]: 问答检索 1\.372s$")
        self.assertNotIn("zhiying.pipeline",value)
        quiet = ProgressAccessFilter()
        for method,path,status,expected in (
            ("GET","/internal/tasks/tabc",200,False),
            ("GET","/internal/tasks/tabc/events",200,False),
            ("GET","/internal/tasks/tabc",404,True),
            ("GET","/internal/tasks/tabc",500,True),
            ("POST","/internal/videos/analyze",202,True),
            ("GET","/internal/health",200,True),
            ("GET","/internal/health?service_control=1",200,False),
            ("GET","/internal/health?service_control=1",500,True),
        ):
            item=logging.LogRecord("uvicorn.access",logging.INFO,"",0,"%s %s %s %s %s",("127.0.0.1",method,path,"1.1",status),None)
            self.assertEqual(quiet.filter(item),expected,(method,path,status))

    def test_cloud_visual_concurrency_bounded_and_no_frames_lost(self):
        frames = [
            SimpleNamespace(
                keyframe_id=str(i), scene_index=0, timestamp_ms=i * 1000, path="test"
            )
            for i in range(11)
        ]
        groups = _group_by_scene(frames, 4)
        self.assertEqual([len(g) for g in groups], [4, 4, 3])
        self.assertEqual([f for g in groups for f in g], frames)
        active = maximum = 0
        lock = threading.Lock()

        def describe(*args, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return SimpleNamespace(description="画面", ocr_text="文字", degraded=False)

        service = VisionStages()
        service._cfg = load_app_config(
            root=ROOT,
            ignore_user_settings=True,
            overrides={"vl.cloud_workers": 2, "vl.cloud_sequence_frames": 4},
        )
        service._vl = SimpleNamespace(describe_sequence=describe)
        payloads = service._describe_keyframes(frames, AnalysisReport(), parallel=True)
        self.assertEqual(
            [p["keyframe_id"] for p in payloads], [f.keyframe_id for f in frames]
        )
        self.assertEqual(maximum, 2)
        self.assertTrue(all(p["ocr_text"] for p in payloads))

    def test_visual_prompts_follow_ocr_parser_contract(self):
        from backend.prompts import DEFAULT_PROMPTS, PromptRegistry
        from backend.models.vl import split_description

        registry = PromptRegistry.load(ROOT / "configs/prompts.yaml")
        for key in (
            "keyframe_describe",
            "keyframe_describe_multi",
            "keyframe_describe_sequence",
            "video_segment_describe",
        ):
            self.assertIn("可读文字：", registry.get(key))
            self.assertIn("可读文字：", DEFAULT_PROMPTS[key])
        desc, ocr = split_description(
            "画面描述：蓝色硬件信息界面\n可读文字：Xeon E5-2696 v4"
        )
        self.assertIn("硬件", desc)
        self.assertEqual(ocr, "Xeon E5-2696 v4")

    def test_upstream_stream_requires_finish_marker(self):
        backend = OpenAICompatibleBackend("https://invalid.test/v1", "test", "sk-test")
        lines = [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n']
        with patch("urllib.request.urlopen") as call:
            call.return_value.__enter__.return_value = iter(lines)
            stream = backend.stream([LlmMessage("user", "test")])
            self.assertEqual(next(stream), "partial")
            with self.assertRaisesRegex(LlmError, "提前结束"):
                next(stream)

    def test_asr_config_wiring(self):
        cfg = load_app_config(
            root=ROOT,
            ignore_user_settings=True,
            overrides={"asr.cpu_threads": 3, "asr.num_workers": 2, "asr.batch_size": 4},
        )
        actual = AsrConfig.from_config(cfg)
        self.assertEqual(
            (actual.cpu_threads, actual.num_workers, actual.batch_size), (3, 2, 4)
        )

    def test_failed_analysis_still_reports_timings(self):
        orch = Orchestrator(repository=Mock())
        with patch.object(orch, "_parse_source", side_effect=RuntimeError("fixture")):
            with self.assertLogs("zhiying.pipeline", level="INFO") as logs:
                report = orch.analyze("fixture")
        self.assertEqual(report.status, "failed")
        self.assertTrue(
            any("分析耗时汇总" in item and "总耗时" in item for item in logs.output)
        )
        self.assertFalse(report.stages[0].ok)

    def test_cache_timings_do_not_mix_historical_model_work(self):
        report = AnalysisReport(
            stages=[StageReport(stage=Stage.DONE, skipped=True)],
            profile={"text_encode_ms": 12000},
            elapsed_ms=5,
        )
        orch = Orchestrator(repository=Mock())
        with self.assertLogs("zhiying.pipeline", level="INFO") as logs:
            orch._log_timings(report)
        self.assertIn("复用已有分析结果", logs.output[0])
        self.assertNotIn("BGE 文本编码", logs.output[0])

    def test_versioned_result_cache_logs_only_current_read_time(self):
        from backend.analysis.runs import _execute
        saved = {'video_id': 'vcached', 'status': 'ready', 'elapsed_ms': 13000,
                 'profile': {'text_encode_ms': 12000}, 'summary': '已保存报告'}
        repo = Mock()
        repo.find_video.return_value = SimpleNamespace(status='ready')
        repo.get_video_report.return_value = saved
        orch = Orchestrator(repository=repo, config=load_app_config(root=ROOT, ignore_user_settings=True))
        with patch('backend.media.parser.parse_source', return_value=SimpleNamespace(video_id='vcached')):
            with self.assertLogs('zhiying.pipeline', level='INFO') as logs:
                result = _execute(orch, 'fixture', AnalyzeOptions())
        self.assertEqual(result.elapsed_ms, 13000, '历史报告计时保持不变')
        self.assertEqual(saved['elapsed_ms'], 13000)
        self.assertIn('复用已有分析结果', logs.output[0])
        self.assertNotIn('BGE 文本编码', logs.output[0])
        self.assertNotIn('13.000s', logs.output[0], '本次复用日志不能冒用历史分析时间')

    def test_parallel_workers_join_before_error_cleanup(self):
        done = threading.Event()

        def fail():
            raise RuntimeError("fixture")

        def other():
            time.sleep(0.02)
            done.set()

        with self.assertRaises(RuntimeError):
            parallel_pair(fail, other)
        self.assertTrue(done.is_set())

    def test_pipeline_scene_asr_overlap_and_persist_total(self):
        from backend.segmentation.chunker import Chunker
        from test_pipeline import FakeAsr

        barrier = threading.Barrier(2, timeout=2)
        calls = {"scene": 0, "asr": 0}
        repo = Mock()
        repo.find_video.return_value = None
        orch = Orchestrator(repository=repo, chunker=Chunker())
        source = SimpleNamespace(
            video_id="vparallel", display_name="fixture", kind="local"
        )
        probe = SimpleNamespace(duration_ms=40000, has_audio=True)

        def scenes(*args):
            calls["scene"] += 1
            barrier.wait()
            time.sleep(0.03)
            return [], "fixture", []

        def transcribe(*args):
            calls["asr"] += 1
            barrier.wait()
            time.sleep(0.03)
            return FakeAsr().transcribe("fixture", "vparallel")

        with patch.multiple(
            orch,
            _parse_source=Mock(return_value=source),
            _resolve_local=Mock(return_value=(Path("fixture"), probe)),
            _register_video=Mock(),
            _apply_fingerprint=Mock(return_value=False),
            _resume_point=Mock(return_value=None),
            _start_task=Mock(return_value=""),
            _extract_audio=Mock(return_value=(Path("audio.wav"), True)),
            _transcribe=transcribe,
            _detect_scenes=scenes,
            _prepare_keyframes=Mock(return_value=([], [])),
            _describe_and_store=Mock(return_value=[]),
            _summarize=Mock(return_value=("测试", "cloud")),
        ):
            with self.assertLogs("zhiying.pipeline", level="INFO") as logs:
                report = orch.analyze(
                    "fixture", AnalyzeOptions(force=True, skip_vectors=True)
                )
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(calls, {"scene": 1, "asr": 1})
        self.assertEqual(sum("阶段完成 镜头检测" in line for line in logs.output), 1)
        self.assertTrue(any("阶段接收并行结果 镜头检测" in line for line in logs.output))
        persisted = repo.set_video_report.call_args.args[1]
        self.assertEqual(persisted["elapsed_ms"], report.elapsed_ms)
        self.assertGreater(persisted["elapsed_ms"], 0)
        self.assertGreaterEqual(
            next(s for s in report.stages if s.stage.value == "scene").elapsed_ms, 25
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
