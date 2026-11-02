"""在独立AstrBot环境运行；使用真实SDK和替身MCP测试引用消息的触发边界。"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_ASTRBOT = importlib.util.find_spec('astrbot') is not None
if HAS_ASTRBOT:
    _sandbox = tempfile.TemporaryDirectory(prefix='astrbot-sdk-test-', ignore_cleanup_errors=True)
    os.environ['ASTRBOT_ROOT'] = _sandbox.name
    from astrbot.core.message.message_event_result import MessageEventResult
    from integrations.astrbot_plugin_videolens.main import VideoLensPlugin


class Event:
    def __init__(self, text, quoted='video', sender='member', group='test-group', mid='1'):
        segments = ([{'type': 'reply', 'data': {'id': quoted}}] if quoted else [])
        segments += [{'type': 'text', 'data': {'text': text}}]
        self.message_obj = SimpleNamespace(raw_message={'message': segments}, message_id=mid)
        self.group, self.sender, self.stopped = group, sender, False
        self.bot = SimpleNamespace(call_action=AsyncMock(return_value={'group_id': group,
            'message': [{'type': 'json', 'data': {'data': '{"meta":{"detail_1":{"qqdocurl":"https://www.bilibili.com/video/BV1rkxi66Ej2"}}}'}}]}))
    def get_self_id(self):return 'bot'
    def get_sender_id(self):return self.sender
    def get_group_id(self):return self.group
    def stop_event(self):self.stopped = True
    def chain_result(self, chain):return MessageEventResult(chain=chain)


@unittest.skipUnless(HAS_ASTRBOT, '请在AstrBot环境和源码PYTHONPATH下单独运行')
class ReplyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        context = SimpleNamespace(platform_manager=SimpleNamespace(platform_insts=[]))
        with patch('integrations.astrbot_plugin_videolens.main.get_astrbot_data_path', return_value=self.temp.name):
            self.plugin = VideoLensPlugin(context, {'allowed_groups': ['test-group']})
        self.plugin.tool = AsyncMock(return_value={'status': 'done', 'markdown': '## 内容概览\n报告', 'kind': 'report'})
        from PIL import Image
        self.image_path = Path(self.temp.name) / 'result.png'
        Image.new('RGB', (80, 80), 'white').save(self.image_path)
        self.renderer = patch('integrations.astrbot_plugin_videolens.main.render_result',
                              new=AsyncMock(return_value=[str(self.image_path)]))
        self.render_mock = self.renderer.start()
    async def asyncTearDown(self):
        self.renderer.stop()
        await self.plugin.terminate();self.temp.cleanup()
    async def consume(self, event):
        rows = []
        async for row in self.plugin.on_group_message(event):
            self.assertFalse(event.stopped, 'yield之前停止事件会导致AstrBot中断后续MCP调用')
            rows.append(row)
        return rows

    async def test_unquoted_nonvideo_and_other_groups_do_not_call_mcp(self):
        for event in [Event('视频分析', quoted=''), Event('普通聊天', quoted='unknown'),
                      Event('视频分析', group='other-group'), Event('视频分析', sender='bot')]:
            self.assertEqual(await self.consume(event), [])
        self.plugin.tool.assert_not_awaited()

    async def test_card_analysis_followup_and_deduplication(self):
        first = Event('视频分析')
        rows = await self.consume(first)
        self.assertEqual(len(rows), 2);self.assertTrue(first.stopped)
        from astrbot.api.message_components import Image, Reply
        self.assertTrue(any(isinstance(part, Image) for part in rows[-1].chain))
        self.assertTrue(any(isinstance(part, Reply) and str(part.id) == 'video' for part in rows[-1].chain))
        self.render_mock.assert_awaited_once()
        first_call = self.plugin.tool.await_args.kwargs
        self.assertEqual(first_call['url'], 'https://www.bilibili.com/video/BV1rkxi66Ej2')
        self.assertEqual(first_call['question'], '')
        self.assertEqual(await self.consume(Event('视频分析')), [])
        self.assertEqual(self.plugin.tool.await_count, 1)
        self.plugin.tool.return_value = {'status': 'done', 'outputs': [{'kind': 'answer', 'markdown': '结论 [1]'}]}
        rows = await self.consume(Event('比较了哪些CPU？', mid='2'))
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(isinstance(part, Image) for part in rows[-1].chain))
        self.assertEqual(self.plugin.tool.await_args.kwargs['question'], '比较了哪些CPU？')
        self.assertEqual(first_call['session_key'], self.plugin.tool.await_args.kwargs['session_key'])
        await self.consume(Event('比较了哪些CPU？', sender='another', mid='3'))
        self.assertNotEqual(first_call['session_key'], self.plugin.tool.await_args.kwargs['session_key'])
        self.assertEqual(await self.consume(Event('这是普通回复', quoted='analysis-report', mid='4')), [])

    async def test_cross_group_reference_is_rejected_before_tool(self):
        event = Event('视频分析')
        event.bot.call_action.return_value['group_id'] = 'other-group'
        rows = await self.consume(event)
        self.assertTrue(any('不属于当前群' in part.text for row in rows for part in row.chain if hasattr(part, 'text')))
        self.plugin.tool.assert_not_awaited()

    async def test_image_error_is_visible_and_does_not_fall_back_to_report_text(self):
        self.render_mock.side_effect = RuntimeError('本地图片渲染失败')
        rows = await self.consume(Event('视频分析'))
        text = ''.join(part.text for row in rows for part in row.chain if hasattr(part, 'text'))
        self.assertIn('本地图片渲染失败', text)
        self.assertNotIn('## 内容概览', text)

    async def test_text_output_requires_explicit_configuration(self):
        self.plugin.config['result_format'] = 'text'
        rows = await self.consume(Event('视频分析'))
        self.assertTrue(any('## 内容概览' in part.text for row in rows for part in row.chain if hasattr(part, 'text')))
        self.render_mock.assert_not_awaited()

    async def test_actual_onebot_sender_receives_image_bytes_not_plain_markdown(self):
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
        event = Event('视频分析')
        rows = await self.consume(event)
        bot = SimpleNamespace(send_group_msg=AsyncMock())
        await AiocqhttpMessageEvent.send_message(bot, rows[-1], is_group=True, session_id='12345')
        payload = bot.send_group_msg.await_args.kwargs
        self.assertEqual(payload['group_id'], 12345)
        self.assertEqual([part['type'] for part in payload['message']], ['reply', 'image'])
        self.assertEqual(str(payload['message'][0]['data']['id']), 'video')
        import base64
        data = payload['message'][1]['data']['file']
        self.assertTrue(data.startswith('base64://'))
        self.assertEqual(base64.b64decode(data.removeprefix('base64://')), self.image_path.read_bytes())
        self.assertFalse(rows[-1].use_t2i_)
        self.assertFalse(rows[-1].use_markdown_)

    async def test_real_markdown_rendering_and_onebot_encoding_without_network(self):
        from io import BytesIO
        from PIL import Image
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
        from integrations.astrbot_plugin_videolens.rendering import render_result
        import base64
        def save(image):
            image.save(self.image_path)
            return str(self.image_path)
        markdown = '## 测试结论\n\n**价格**为859元。\n\n| 指标 | 数值 |\n| --- | --- |\n| 功耗 | 125W |\n\n```python\nprint(123)\n```\n\n![外部图片](https://invalid.example/x.png)'
        with patch('integrations.astrbot_plugin_videolens.rendering.save_temp_img', side_effect=save), \
             patch('astrbot.core.utils.t2i.local_strategy.ImageBlock.load', side_effect=AssertionError('不应下载外部图片')):
            paths = await render_result(markdown, 'answer', 'https://www.bilibili.com/video/BV1rkxi66Ej2')
        self.assertEqual(len(paths), 1)
        from astrbot.api.message_components import Image as ImageComponent, Reply
        chain = MessageEventResult(chain=[Reply(id='video'), ImageComponent.fromBytes(Path(paths[0]).read_bytes())])
        payload = await AiocqhttpMessageEvent._parse_onebot_json(chain)
        self.assertEqual([p['type'] for p in payload], ['reply', 'image'])
        with Image.open(BytesIO(base64.b64decode(payload[-1]['data']['file'][9:]))) as image:
            self.assertEqual(image.width, 1000)
            self.assertGreater(image.height, 200)

    async def test_empty_render_is_an_error_not_a_silent_success(self):
        self.render_mock.return_value = []
        rows = await self.consume(Event('视频分析'))
        self.assertTrue(any('未生成任何页面' in part.text for row in rows for part in row.chain if hasattr(part, 'text')))

    async def test_pagination_preserves_every_pixel(self):
        from PIL import Image, ImageChops
        from integrations.astrbot_plugin_videolens.rendering import paginate
        source = Image.new('RGB', (40, 9000), 'white')
        for y in range(9000):
            source.putpixel((3, y), (y % 256, y % 251, y % 241))
        pages = list(paginate(source))
        self.assertTrue(all(page.height <= 3600 for page in pages))
        self.assertEqual(sum(page.height for page in pages), source.height)
        restored = Image.new('RGB', source.size)
        y = 0
        for page in pages:
            restored.paste(page, (0, y));y += page.height
        self.assertIsNone(ImageChops.difference(restored, source).getbbox())


if __name__ == '__main__':unittest.main(verbosity=2)
