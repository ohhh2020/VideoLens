"""在明确指定的群发送真实 Bilibili 分享卡片，不手工伪造小程序 JSON。"""
import argparse
import json
from pathlib import Path
import re
import ssl
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener

import certifi


def send_card(credentials, group, url):
    match = re.fullmatch(r'https://www\.bilibili\.com/video/(BV[0-9A-Za-z]{10})/?', url)
    if not match:
        raise ValueError('请输入标准 https://www.bilibili.com/video/BV… 地址')
    cfg = json.loads(Path(credentials).read_text(encoding='utf-8'))
    base = cfg.get('onebot_url', 'http://127.0.0.1:3001').rstrip('/')
    parsed = urlsplit(base)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username or parsed.password:
        raise ValueError('OneBot 地址必须为本机 HTTP 地址')
    http = build_opener(ProxyHandler({}), HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())))

    def action(name, body):
        request = Request(base + '/' + name, data=json.dumps(body, ensure_ascii=False).encode(),
                          headers={'Authorization': 'Bearer ' + cfg['onebot_token'], 'Content-Type': 'application/json'})
        with http.open(request, timeout=40) as response:
            result = json.load(response)
        if result.get('retcode') != 0:
            raise RuntimeError('OneBot 操作失败：' + name)
        return result['data']

    request = Request('https://api.bilibili.com/x/web-interface/view?' + urlencode({'bvid': match[1]}),
                      headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.bilibili.com/'})
    with http.open(request, timeout=30) as response:
        metadata = json.load(response)
    if metadata.get('code') != 0:
        raise RuntimeError('Bilibili 视频信息读取失败，未发送卡片')
    video = metadata['data']
    ark = action('get_mini_app_ark', {'type': 'bili', 'title': video['title'],
                 'desc': video.get('owner', {}).get('name', 'Bilibili'), 'picUrl': video['pic'],
                 'jumpUrl': url, 'webUrl': url})['data']
    detail = (ark.get('meta') or {}).get('detail_1', {}) if isinstance(ark, dict) else {}
    if (not isinstance(ark, dict) or ark.get('app') != 'com.tencent.miniapp_01'
            or not ark.get('view') or ark.get('view') == 'notification'
            or not isinstance(detail, dict) or not detail.get('preview')
            or not (detail.get('qqdocurl') or detail.get('url'))):
        raise RuntimeError('NapCat 未返回有效的小程序分享数据，未发送卡片')
    return action('send_group_msg', {'group_id': int(group), 'message': [
        {'type': 'json', 'data': {'data': json.dumps(ark, ensure_ascii=False)}}]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credentials', required=True, help='本机私密 OneBot 配置文件')
    parser.add_argument('--group', required=True, type=int, help='明确授权发送的群号')
    parser.add_argument('--url', required=True, help='标准 Bilibili BV 视频地址')
    args = parser.parse_args()
    print(json.dumps(send_card(args.credentials, args.group, args.url)))
