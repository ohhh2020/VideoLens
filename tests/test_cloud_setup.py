"""首次无云端配置可进入设置，禁止隐式使用占位模型或生成替代结果。"""
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import load_app_config
from backend.pipeline.factory import build_orchestrator
from backend.pipeline.orchestrator import PipelineError
from backend.user_settings import describe_backends


class CloudSetupTests(unittest.TestCase):
    def test_default_cloud_fields_are_empty(self):
        cfg=load_app_config(ignore_user_settings=True,env={})
        self.assertEqual(cfg.llm.cloud.base_url,"")
        self.assertEqual(cfg.llm.cloud.model,"")
        self.assertEqual(cfg.vl.cloud_model,"")

    def test_setup_mode_keeps_mandatory_local_verification_and_rejects_generation(self):
        with tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
            cfg=load_app_config(root=Path(tmp),env={},overrides={"llm.cloud.api_key_env":"","vl.cloud_api_key_env":""})
            verify=stack.enter_context(patch("backend.models.required.verify_required_models"))
            stack.enter_context(patch("backend.models.manager.ModelManager"))
            embed=Mock();embed._text=None;embed._image=None
            asr=Mock();asr.active_device=("cpu","int8")
            for name,value in {"_build_embedder":embed,"_build_asr":asr,"_build_vectors":Mock(),"_build_scene":Mock(),"_build_chunker":Mock(),"_build_context":Mock(),"_build_downloader":Mock()}.items():
                stack.enter_context(patch("backend.pipeline.factory."+name,return_value=value))
            stack.enter_context(patch("backend.memory.store.MemoryStore"))
            orch=build_orchestrator(cfg,repository=Mock())
            try:
                verify.assert_called_once()
                embed.embed_query.assert_called_once()
                embed.embed_image.assert_called_once()
                asr.load.assert_called_once()
                self.assertEqual(len(orch._cloud_setup_required),2)
                self.assertTrue(all(not item.enabled for item in describe_backends(cfg)))
                with self.assertRaisesRegex(PipelineError,"网页设置"):
                    orch.ask("video","问题")
                with self.assertRaisesRegex(PipelineError,"网页设置"):
                    next(orch.ask_stream("video","问题"))
                report=orch.analyze("unused.mp4")
                self.assertEqual(report.status,"failed")
                self.assertTrue(any("网页设置" in error for error in report.errors))
            finally:orch.close()


if __name__=="__main__":unittest.main(verbosity=2)
