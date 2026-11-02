"""使用 AstrBot 自带的本地 Markdown 渲染器，报告正文不发往外部转图服务。"""
import asyncio
import re

from astrbot.core.utils.io import save_temp_img
from astrbot.core.utils.t2i.local_strategy import MarkdownRenderer


def paginate(image, maximum=3600):
    """优先在空白行分页；所有像素按原顺序保留，不截断长报告。"""
    top = 0
    while top < image.height:
        bottom = min(top + maximum, image.height)
        if bottom < image.height:
            for row in range(bottom - 1, max(top + maximum // 2, bottom - 240), -1):
                extrema = image.crop((0, row, image.width, row + 1)).convert('L').getextrema()
                if extrema[1] - extrema[0] < 8:
                    bottom = row + 1
                    break
        yield image.crop((0, top, image.width, bottom))
        top = bottom


async def render_result(markdown, kind, source):
    """在线程内渲染，避免阻塞机器人收消息；外部图片只保留说明文字。"""
    label = '视频分析' if kind == 'report' else '视频追问'
    # 不让云端输出的图片语法引发渲染器访问任意网络地址或本地文件。
    safe = re.sub(r'!\[([^\]]*)\]\([^\n]*?\)', r'[图片说明：\1]', markdown)
    safe = re.sub(r'^\s*\[[^\]]+\]:\s*.*$', '', safe, flags=re.M)
    content = f'# 知影 · {label}\n\n{safe}\n\n---\n来源：{source}\n\n继续提问请回复原视频消息。'

    def render():
        image = asyncio.run(MarkdownRenderer(font_size=25, width=1000).render(content))
        try:
            paths = []
            for page in paginate(image):
                try:
                    paths.append(save_temp_img(page))
                finally:
                    page.close()
            return paths
        finally:
            image.close()

    return await asyncio.to_thread(render)
