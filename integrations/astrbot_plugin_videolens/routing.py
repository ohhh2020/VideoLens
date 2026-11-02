"""纯消息解析，不依赖大模型，不从标题猜测下载地址。"""
import html
import json
import re
from urllib.parse import urlsplit

URL = re.compile(r'https?://[^\s<>"\[\]{}\\]+', re.I)


def video_urls(value):
    found = []
    def walk(item, depth=0):
        if depth > 12:
            return
        if isinstance(item, dict):
            # QQ分享卡片的qqdocurl是卡片本身的跳转目标；不将备用url再当另一视频。
            primary = video_urls(item['qqdocurl']) if isinstance(item.get('qqdocurl'), str) else []
            if primary:
                walk(primary, depth + 1)
                return
            for child in item.values():walk(child, depth + 1)
        elif isinstance(item, list):
            for child in item:walk(child, depth + 1)
        elif isinstance(item, str):
            text = html.unescape(item).replace('\\/', '/')
            if text.lstrip().startswith(('{', '[')):
                try:walk(json.loads(text), depth + 1);return
                except (ValueError, RecursionError):pass
            for match in URL.finditer(text):
                url = match[0].rstrip('，。！？；、,.!?;:：）》】）)')
                parsed = urlsplit(url)
                host = (parsed.hostname or '').lower()
                try:
                    if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:continue
                except ValueError:
                    continue
                if host in {'www.bilibili.com', 'm.bilibili.com', 'bilibili.com'} and re.fullmatch(r'/video/(BV[0-9A-Za-z]{10}|av\d+)/?', parsed.path):
                    normalized = 'https://www.bilibili.com' + parsed.path.rstrip('/')
                elif host == 'b23.tv' and re.fullmatch(r'/[A-Za-z0-9]{4,32}/?', parsed.path):
                    normalized = 'https://b23.tv' + parsed.path.rstrip('/')
                else:continue
                if normalized not in found:found.append(normalized)
    walk(value)
    return found


def quoted_id(segments):
    for part in segments:
        if part.get('type') == 'reply':
            return str(part.get('data', {}).get('id') or '')
    return ''


def plain_text(segments):
    return ''.join(str(part.get('data', {}).get('text', '')) for part in segments if part.get('type') == 'text').strip()


def chunks(text, limit=2800):
    """按换行分条；超长单行继续切分，全部字符保留。"""
    current = ''
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > limit:
            yield current;current = ''
        while len(line) > limit:
            yield line[:limit];line = line[limit:]
        current += line
    if current:yield current
