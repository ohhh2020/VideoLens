"""有界的 TF-IDF/TextRank 关键词模型：一次分词，不加载权重或调用 GPU/云端。"""
from collections import Counter, defaultdict
import logging
import math
import re


MODEL = 'tfidf-textrank-v1'
LEGACY_MODEL = 'tfidf-v1'
LEGACY_STOP = frozenset({'用户','视频','这个','那个','我们','你们','他们','就是','可以','一个',
    '然后','因为','所以','还有','进行','以及','一些','不是','这样','现在','已经','没有','时候',
    '这次','大家','比较','内容','画面','文字','一下','可能','如果','对于'})
STOP = frozenset({'用户', '视频', '这个', '那个', '我们', '你们', '他们', '就是', '可以',
    '一个', '然后', '因为', '所以', '还有', '进行', '以及', '一些', '不是', '这样',
    '现在', '已经', '没有', '时候', '这次', '大家', '比较', '内容', '画面', '文字',
    '一下', '可能', '如果', '对于', '大家好', '感谢', '观看', '点赞', '关注', '转发',
    '显示', '展示', '一台', '正在', '画面中'})
IDENTIFIER = re.compile(r'[a-zA-Z0-9]+(?:[-_.+][a-zA-Z0-9]+)*')
MAX_NODES = 128
MAX_EDGES = 4096
MAX_ANCHORS = 12000
ITERATIONS = 8


def legacy_extract(documents, limit=24):
    """保留既有评分，实验关闭时不运行共现图或更改关键词取材。"""
    import jieba
    jieba.setLogLevel(logging.WARNING)
    tokenized = []
    for text in documents:
        words = [word.lower() for word in jieba.lcut(text)
                 if 2 <= len(word.strip()) <= 24 and word not in LEGACY_STOP
                 and re.search(r'[\w\u4e00-\u9fff]', word) and not word.isdecimal()]
        words += re.findall(r'\b[a-zA-Z][a-zA-Z0-9_.+-]*\d[a-zA-Z0-9_.+-]*\b', text)
        tokenized.append(words)
    df, scores = Counter(), Counter()
    for words in tokenized:
        df.update(set(words))
    for words in tokenized:
        for word, count in Counter(words).items():
            scores[word] += (1 + math.log(count)) * (1 + math.log((len(documents) + 1) / (df[word] + 1)))
    top = scores.most_common(limit)
    scale = top[0][1] if top else 1
    return [(word, value / scale) for word, value in top]


def _words(text, tokenizer):
    """型号/版本号作为整词；不把 i5-13600KF 拆成无关联的多个分词。"""
    position = 0
    for match in IDENTIFIER.finditer(text):
        yield from tokenizer.cut(text[position:match.start()])
        yield match.group()
        position = match.end()
    yield from tokenizer.cut(text[position:])


def _rank(sequences, nodes):
    """限制图大小、扫描锚点与迭代数；长输入的图扫描机会分散至全文。"""
    graph = defaultdict(Counter)
    edges = set()
    stride = max(1, math.ceil(sum(map(len, sequences)) / MAX_ANCHORS))
    for words in sequences:
        for index in range(0, len(words), stride):
            word = words[index]
            if word not in nodes:
                continue
            for other in words[index + 1:index + 5]:
                if other == word or other not in nodes:
                    continue
                edge = tuple(sorted((word, other)))
                if edge in edges or len(edges) >= MAX_EDGES:
                    continue
                # 只记录是否共现，重复字幕/水印不重复累加边权。
                edges.add(edge)
                graph[word][other] = 1
                graph[other][word] = 1
    if not edges:
        return {}
    ranks = dict.fromkeys(nodes, 1.0)
    for _ in range(ITERATIONS):
        updated = dict.fromkeys(nodes, .15)
        for word, adjacent in graph.items():
            share = .85 * ranks[word] / len(adjacent)
            for other in adjacent:
                updated[other] += share
        ranks = updated
    scale = max(ranks.values())
    return {word: value / scale for word, value in ranks.items()}


def extract(documents, limit=24):
    """关键词全部来自原文；仅供证据排序，不向报告增加未经证实的事实。"""
    if limit <= 0:
        return []
    import jieba
    jieba.setLogLevel(logging.WARNING)
    # 完全相同的 OCR/字幕只参与一次统计；原始证据与时间关联由打包层保留。
    unique = dict.fromkeys(re.sub(r'\s+', ' ', text).strip() for text in documents if text.strip())
    sequences = []
    for text in unique:
        words = [word for raw in _words(text, jieba) if (word := raw.strip().lower())
                 and 2 <= len(word) <= 40 and word not in STOP
                 and re.search(r'[a-z\u4e00-\u9fff]', word)]
        if words:
            sequences.append(words)
    if not sequences:
        return []
    df, scores = Counter(), Counter()
    for words in sequences:
        df.update(set(words))
    for words in sequences:
        normalization = 1 + math.log(len(words))
        for word, count in Counter(words).items():
            scores[word] += ((1 + math.log(count)) / normalization
                             * (1 + math.log((len(sequences) + 1) / (df[word] + 1))))
    top = scores.most_common(MAX_NODES)
    graph = _rank(sequences, {word for word, _ in top})
    scale = top[0][1]
    combined = []
    for word, score in top:
        identifier = bool(IDENTIFIER.fullmatch(word) and any(char.isdigit() for char in word))
        # 主体仍是全量文本统计；共现与完整型号只作有界的辅助加权。
        combined.append((word, .75 * score / scale + .20 * graph.get(word, 0) + .05 * identifier))
    combined.sort(key=lambda pair: (-pair[1], pair[0]))
    maximum = combined[0][1]
    return [(word, value / maximum) for word, value in combined[:limit]]


def matcher(word):
    """ASCII 型号按词边界匹配，避免 A1 为 A10 等不同型号加分。"""
    if IDENTIFIER.fullmatch(word):
        pattern = re.compile(r'(?<![a-z0-9_.+-])' + re.escape(word) + r'(?![a-z0-9_.+-])', re.I)
        return lambda text: bool(pattern.search(text))
    return lambda text: word in text
