"""替身HTTP验证消息类型，不访问Bilibili、NapCat或真实QQ群。"""
import copy
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.send_bilibili_card import send_card

URL = 'https://www.bilibili.com/video/BV1rkxi66Ej2'
ARK = {'app': 'com.tencent.miniapp_01', 'view': 'view_test', 'ver': '1.0.0.1',
       'meta': {'detail_1': {'preview': 'https://i0.hdslb.com/test.jpg', 'qqdocurl': URL}}}


class CardTests(unittest.TestCase):
    def call(self, ark):
        calls = []
        class Http:
            def open(self, request, timeout=None):
                calls.append(request)
                if '/x/web-interface/view?' in request.full_url:
                    value = {'code': 0, 'data': {'title': '测试标题', 'pic': 'https://i0.hdslb.com/test.jpg',
                                               'owner': {'name': '测试作者'}}}
                elif request.full_url.endswith('/get_mini_app_ark'):
                    value = {'retcode': 0, 'data': {'data': ark}}
                elif request.full_url.endswith('/send_group_msg'):
                    value = {'retcode': 0, 'data': {'message_id': 123}}
                else:
                    raise AssertionError('不应调用其他接口')
                return BytesIO(json.dumps(value).encode())
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'fake-credentials.json'
            config.write_text(json.dumps({'onebot_token': 'fake-token'}), encoding='utf-8')
            with patch('integrations.send_bilibili_card.build_opener', return_value=Http()):
                try:
                    result = send_card(config, 12345, URL)
                except RuntimeError:
                    self.assertFalse(any(r.full_url.endswith('/send_group_msg') for r in calls))
                    raise
        return result, calls

    def test_signed_ark_is_sent_as_json_without_plain_text_fallback(self):
        result, calls = self.call(ARK)
        self.assertEqual(result['message_id'], 123)
        generated = json.loads(calls[1].data)
        self.assertEqual(generated['type'], 'bili')
        self.assertEqual(generated['jumpUrl'], URL)
        sent = json.loads(calls[2].data)
        self.assertEqual(sent['group_id'], 12345)
        self.assertEqual([part['type'] for part in sent['message']], ['json'])
        self.assertEqual(json.loads(sent['message'][0]['data']['data']), ARK)

    def test_incomplete_old_style_cards_are_rejected_without_sending(self):
        for ark in [None, {}, {'app': 'com.tencent.miniapp_01', 'view': 'notification'},
                    {**ARK, 'view': 'notification'}]:
            with self.subTest(ark=ark), self.assertRaises(RuntimeError):self.call(ark)
        no_preview = copy.deepcopy(ARK)
        del no_preview['meta']['detail_1']['preview']
        with self.assertRaises(RuntimeError):self.call(no_preview)


if __name__ == '__main__':unittest.main(verbosity=2)
