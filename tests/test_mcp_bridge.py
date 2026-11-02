"""本机MCP桥接、QQ回复边界与真实AstrBot接入前的无凭据回归。"""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.videolens_mcp.bridge import Bridge, bilibili_url
from integrations.astrbot_plugin_videolens.routing import video_urls, quoted_id, plain_text, chunks

SOURCE = 'https://www.bilibili.com/video/BV1rkxi66Ej2'


class FakeApi:
    def __init__(self):
        self.messages = []; self.operation = None; self.video = ''; self.calls = []
    def __call__(self, path, data=None):
        self.calls.append((path, data))
        if path == '/conversations':return {'id': 'c1'}
        if path.endswith('/messages'):
            self.operation = {'id': 'o' + str(len(self.calls)), 'status': 'running', 'phase': 'analysis'}
            return {'operation_id': self.operation['id']}
        return {'conversation': {'id': 'c1', 'video_id': self.video}, 'operation': self.operation, 'messages': self.messages}


class MCPBridgeTests(unittest.TestCase):
    def test_local_endpoint_and_bilibili_source_constraints(self):
        with self.assertRaises(ValueError):Bridge(base='https://remote.example', state='unused')
        self.assertEqual(bilibili_url(SOURCE + '?share_source=qq'), SOURCE)
        for value in ['file:///d:/video.mp4', 'https://evil.com/video/BV1rkxi66Ej2', 'https://bilibili.com.evil.com/video/BV1rkxi66Ej2', 'https://www.bilibili.com:1234/video/BV1rkxi66Ej2']:
            with self.assertRaises(ValueError):bilibili_url(value)

    def test_async_submission_owned_result_and_ready_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            bridge = Bridge(state=Path(temp) / 'mcp.sqlite3'); api = FakeApi(); bridge.api = api
            result = bridge.analyze(SOURCE, 'group1:message1:user1', 'event1')
            self.assertEqual(result['status'], 'accepted')
            again = bridge.analyze(SOURCE, 'group1:message1:user1', 'event2')
            self.assertEqual(again['job_id'], result['job_id'])
            self.assertEqual(len([c for c in api.calls if c[0].endswith('/messages')]), 1)
            api.video = 'v1'; api.operation['status'] = 'done'
            api.messages = [{'kind': 'report', 'text': '## 原始Markdown', 'payload': {'summary': '## 原始Markdown'}, 'operation_id': result['job_id']}]
            data = bridge.result(result['job_id']);self.assertEqual(data['outputs'][0]['markdown'], '## 原始Markdown')
            import json
            from contextlib import closing
            with closing(bridge.connect()) as conn:
                saved = json.loads(conn.execute('SELECT result FROM jobs WHERE id=?', (result['job_id'],)).fetchone()[0])
            self.assertNotIn('outputs', saved, '桥接不能另存业务正文，避免删除后还能读取旧副本')
            self.assertEqual(bridge.result(result['job_id'])['outputs'][0]['markdown'], '## 原始Markdown')
            self.assertEqual(bridge.analyze(SOURCE, 'group1:message1:user1', 'event3')['status'], 'done')
            self.assertEqual(bridge.ask('group1:message1:user1', '比较对象是谁？', 'event4')['status'], 'accepted')
            with self.assertRaises(ValueError):bridge.result('not-owned')
            with self.assertRaises(ValueError):bridge.ask('group2:message1:user1', '其他群？', 'event5')

    def test_reply_and_card_parsing_keeps_real_candidates_only(self):
        card = [{'type': 'json', 'data': {'data': '{"meta":{"detail_1":{"qqdocurl":"https:\\/\\/www.bilibili.com\\/video\\/BV1rkxi66Ej2?from=qq"}}}'}}]
        self.assertEqual(video_urls(card), [SOURCE])
        self.assertEqual(video_urls({'meta': {'detail_1': {'qqdocurl': 'https://b23.tv/AbcDe1', 'url': SOURCE}}}),
                         ['https://b23.tv/AbcDe1'], '同一QQ卡片的备用地址不能误判为多个视频')
        self.assertEqual(video_urls('【分享】https://b23.tv/AbcDe1，'), ['https://b23.tv/AbcDe1'])
        segments = [{'type': 'reply', 'data': {'id': 123}}, {'type': 'at', 'data': {'qq': 12}}, {'type': 'text', 'data': {'text': '视频分析'}}]
        self.assertEqual(quoted_id(segments), '123');self.assertEqual(plain_text(segments), '视频分析')
        self.assertEqual(quoted_id(segments[1:]), '')
        self.assertEqual(video_urls('https://bilibili.com.evil.com/video/BV1rkxi66Ej2'), [])

    def test_qq_long_output_has_no_dropped_characters(self):
        text = '# 概览\n\n' + '代码与数字123' * 1000 + '\n## 结论\n末尾内容'
        rows = list(chunks(text, 500))
        self.assertEqual(''.join(rows), text)
        self.assertTrue(all(len(row) <= 500 for row in rows))


if __name__ == '__main__':unittest.main(verbosity=2)
