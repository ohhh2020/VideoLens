"""OneBot回复门控 → AstrBot注册的MCP客户端 → VideoLens持久化工作流。"""
import asyncio
from contextlib import closing
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from .routing import video_urls, quoted_id, plain_text, chunks
from .rendering import render_result


class VideoLensPlugin(Star):
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.config = config or {}
        directory = Path(get_astrbot_data_path()) / 'plugin_data/astrbot_plugin_videolens'
        directory.mkdir(parents=True, exist_ok=True)
        self.db = directory / 'messages.sqlite3'
        self.running = set()
        self.self_test_hooks = []
        with closing(self.connect()) as conn, conn:
            conn.executescript('''CREATE TABLE IF NOT EXISTS roots(scope TEXT, message TEXT, url TEXT,
                PRIMARY KEY(scope,message));CREATE TABLE IF NOT EXISTS seen(id TEXT PRIMARY KEY);''')

    async def initialize(self):
        await self.install_self_test()

    @filter.on_platform_loaded()
    async def install_self_test(self):
        """仅自测时接回NapCat的message_sent；正常群友消息不经过此分支。"""
        if not self.config.get('self_test_prefix'):
            return
        for adapter in self.context.platform_manager.platform_insts:
            bot = getattr(adapter, 'bot', None)
            if bot is None or any(item[0] is bot for item in self.self_test_hooks):
                continue
            async def receive(raw, platform=adapter):
                if str(raw.get('user_id')) != str(raw.get('self_id')):
                    return
                allowed = [str(value) for value in self.config.get('allowed_groups', [])]
                if allowed and str(raw.get('group_id')) not in allowed:
                    return
                if not plain_text(raw.get('message', [])).startswith(self.config.get('self_test_prefix', '\0')):
                    return
                from aiocqhttp import Event
                copied = Event.from_payload({**raw, 'post_type': 'message', 'message_type': 'group'})
                message = await platform.convert_message(copied)
                if message:
                    await platform.handle_msg(message)
            bot.subscribe('message_sent', receive)
            self.self_test_hooks.append((bot, receive))

    async def terminate(self):
        for bot, callback in self.self_test_hooks:
            bot.unsubscribe('message_sent', callback)
        self.self_test_hooks.clear()

    def connect(self):
        return sqlite3.connect(self.db)

    async def tool(self, name, **arguments):
        manager = self.context.get_llm_tool_manager()
        client = manager.mcp_client_dict.get(self.config.get('mcp_server', 'videolens'))
        if client is None:
            raise RuntimeError('VideoLens MCP尚未连接，请检查AstrBot的MCP设置')
        result = await client.call_tool_with_reconnect(name, arguments, timedelta(seconds=30))
        text = '\n'.join(part.text for part in result.content if getattr(part, 'type', '') == 'text')
        if result.isError:
            raise RuntimeError(text or 'MCP调用失败')
        return getattr(result, 'structuredContent', None) or json.loads(text)

    async def original(self, event, message_id):
        data = await asyncio.wait_for(event.bot.call_action('get_msg', message_id=message_id), 25)
        if data.get('group_id') and str(data['group_id']) != event.get_group_id():
            raise ValueError('引用消息不属于当前群')
        content = data.get('message', [])
        # 合并转发只展开一层，限制数量，不递归执行消息中的任何指令。
        for part in list(content)[:8]:
            if isinstance(part, dict) and part.get('type') == 'forward':
                forwarded = await asyncio.wait_for(event.bot.call_action('get_forward_msg',
                    message_id=part.get('data', {}).get('id')), 25)
                content = [content, (forwarded.get('messages') or [])[:20]]
        return video_urls(content)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=100)
    async def on_group_message(self, event: AstrMessageEvent):
        if not hasattr(event, 'bot'):
            return
        group = event.get_group_id()
        allowed = [str(value) for value in self.config.get('allowed_groups', [])]
        if allowed and group not in allowed:
            return
        raw = event.message_obj.raw_message
        segments = raw.get('message', []) if hasattr(raw, 'get') else []
        root, text = quoted_id(segments), plain_text(segments)
        if not root or not text:
            return
        if event.get_sender_id() == event.get_self_id():
            prefix = self.config.get('self_test_prefix', '')
            if not prefix or not text.startswith(prefix):
                return
            text = text[len(prefix):].strip()
        trigger = self.config.get('trigger', '视频分析')
        analyzing = trigger in text
        scope = event.get_self_id() + ':' + group
        with closing(self.connect()) as conn:
            binding = conn.execute('SELECT url FROM roots WHERE scope=? AND message=?', (scope, root)).fetchone()
        if not analyzing and not binding:
            return
        def reply(value):
            return event.chain_result([Reply(id=root), Plain(text='知影 · ' + value)]).use_t2i(False)
        lock_key = scope + ':' + root
        if lock_key in self.running:
            yield reply('这条视频正在处理，请等本轮完成后再追问。')
            event.stop_event()
            return
        key = hashlib.sha256((scope + ':' + str(event.message_obj.message_id)).encode()).hexdigest()
        with closing(self.connect()) as conn, conn:
            if not conn.execute('INSERT OR IGNORE INTO seen VALUES(?)', (key,)).rowcount:
                event.stop_event()
                return
        self.running.add(lock_key)
        try:
            urls = await self.original(event, root)
            if len(urls) != 1:
                yield reply('请回复只包含一个Bilibili视频链接或分享卡片的原消息；不能回复分析报告代替原视频。')
                return
            url = urls[0]
            if binding and binding[0] != url:
                raise ValueError('原视频链接与会话绑定不一致')
            question = text.replace(trigger, '', 1).strip(' ：:，,\n') if analyzing else text
            if len(question) > 2000:
                raise ValueError('问题最多2000字符，请缩短后重新发送')
            with closing(self.connect()) as conn, conn:
                conn.execute('INSERT OR IGNORE INTO roots VALUES(?,?,?)', (scope, root, url))
            # 原视频消息关联共享报告，用户ID隔离问答上下文；不同群绝不共用上下文。
            session = f'qq:{scope}:video-message:{root}:user:{event.get_sender_id()}'
            yield reply('已收到，正在分析视频…' if analyzing else '正在结合这条视频回答…')
            result = await self.tool('videolens_analyze', url=url, session_key=session,
                                     request_id=key, question=question)
            deadline = time.monotonic() + max(60, int(self.config.get('timeout_seconds', 1800)))
            while result['status'] in {'accepted', 'running', 'queued'}:
                if time.monotonic() >= deadline:
                    yield reply('本次等待超时，后端任务仍在运行。稍后请回复原视频重新发送“视频分析”查询结果。')
                    return
                await asyncio.sleep(max(1, int(self.config.get('poll_seconds', 2))))
                result = await self.tool('videolens_result', job_id=result['job_id'])
            if result['status'] != 'done':
                raise RuntimeError(result.get('error') or '分析未完成，请重新触发')
            outputs = result.get('outputs') or [{'kind': result.get('kind', 'report'), 'markdown': result.get('markdown', '')}]
            if not analyzing:
                outputs = [item for item in outputs if item['kind'] == 'answer']
            if not outputs or not any(item.get('markdown') for item in outputs):
                raise RuntimeError('未收到有效报告或回答，请重新发送问题')
            limit = min(3500, max(500, int(self.config.get('message_characters', 2800))))
            for output in outputs:
                if not output.get('markdown'):
                    continue
                if self.config.get('result_format', 'image') == 'text':
                    for text_part in chunks(output['markdown'], limit):
                        yield reply(text_part)
                else:
                    # 图片失败时明确报错，不悄悄用文字替代用户要求的图片。
                    pages = await render_result(output['markdown'], output['kind'], url)
                    if not pages:
                        raise RuntimeError('图片渲染未生成任何页面，请检查插件版本与AstrBot日志')
                    for path in pages:
                        # 在进入发送流水线前读取真实图片。OneBot收到base64图片内容，
                        # 不依赖NapCat和AstrBot共享文件路径，也不把路径当成文字发送。
                        content = await asyncio.to_thread(Path(path).read_bytes)
                        if not content:
                            raise RuntimeError('图片页面为空，请检查AstrBot临时目录')
                        yield event.chain_result([Reply(id=root), Image.fromBytes(content)]).use_t2i(False).use_markdown(False)
        except Exception as exc:
            self.logger.warning('视频回复处理失败：%s', type(exc).__name__)
            yield reply('处理失败：' + str(exc))
        finally:
            self.running.discard(lock_key)
            # 先让每次yield经过AstrBot发送阶段；提前stop会中止后续MCP调用。
            event.stop_event()
