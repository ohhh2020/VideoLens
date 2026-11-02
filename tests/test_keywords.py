"""关键词排序的来源、完整型号、重复OCR、图预算与报告接入边界。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.analysis.keywords import extract, matcher, MAX_NODES
from backend.analysis.packing import candidates, select
from backend.analysis.policy import estimate_tokens


class KeywordTests(unittest.TestCase):
    def test_models_and_versions_stay_whole_and_grounded(self):
        text = 'i5-13600KF售价1116元，E5 2696v3对比R5 5600XT。DDR4-3200不支持DDR5，CUDA12.8。'
        tags = dict(extract([text]))
        for term in ('i5-13600kf', '2696v3', '5600xt', 'ddr4-3200', 'cuda12.8'):
            self.assertIn(term, tags)
        self.assertNotIn('xt', tags)
        self.assertTrue(all(term in text.lower() for term in tags))
        self.assertTrue(all(0 < weight <= 1 for weight in tags.values()))

    def test_duplicate_ocr_does_not_change_keyword_weights(self):
        documents = ['点赞关注支持作者。', '先备份数据，再安装驱动，最后重启。',
                     '5600XT网游领先32%，生产力落后12%。']
        self.assertEqual(extract(documents), extract(documents + [documents[0]] * 200))
        self.assertNotIn('点赞', dict(extract(documents)))

    def test_ascii_boundaries_and_chinese_adjacent_models(self):
        match = matcher('a1')
        self.assertTrue(match('型号A1售价750元。'))
        for text in ('型号A10售价', '型号BA1售价', 'A1-Pro', 'A1.2'):
            self.assertFalse(match(text), text)
        self.assertTrue(matcher('i5-13600kf')('型号i5-13600KF售价1116元。'))

    def test_graph_is_bounded_and_not_required_for_grounding(self):
        texts = [' '.join(f'model{j}' for j in range(i, i + 80)) for i in range(0, 400, 20)]
        from backend.analysis.keywords import _rank
        def bounded(sequences, nodes):
            self.assertLessEqual(len(nodes), MAX_NODES)
            return _rank(sequences, nodes)
        with patch('backend.analysis.keywords._rank', side_effect=bounded) as rank:
            tags = extract(texts)
        rank.assert_called_once()
        self.assertEqual(len(tags), 24)
        self.assertTrue(all(any(term in text for text in texts) for term, _ in tags))
        self.assertEqual(extract(['', '123', '这个视频']), [])
        self.assertEqual(extract(['有效信息'], limit=0), [])

    def test_keywords_affect_selection_without_adding_prompt_claims(self):
        evidence = [{'id': f'e{i}', 'clip_id': f'c{i}', 'kind': 'ocr',
                     'start_ms': i * 1000, 'end_ms': i * 1000,
                     'text': 'i5-13600KF售价1116元，仅限盒装。不能使用DDR4。'} for i in range(40)]
        make = lambda rows: [{'role': 'user', 'content': json.dumps(rows, ensure_ascii=False)}]
        units, tags = candidates(evidence, keyword_assist=True)
        self.assertTrue(any(word == 'i5-13600kf' for word, _ in tags))
        rows, meta = select(evidence, make, 1500, keyword_assist=True)
        self.assertEqual(meta['keyword_model'], 'tfidf-textrank-v1')
        self.assertGreaterEqual(meta['evidence_scoring_ms'], 0)
        self.assertEqual(len(meta['selected_clips']), 40)
        self.assertLessEqual(estimate_tokens(json.dumps(make(rows), ensure_ascii=False)), 1500)
        self.assertTrue(all(row['text'] in evidence[0]['text'] for row in rows))
        self.assertEqual(sum(len(unit['occurrences']) for unit in units), 80)

    def test_report_budget_checks_reuse_one_preference_recall(self):
        from backend.pipeline.orchestrator import Orchestrator
        from backend.pipeline.domain import AnalyzeOptions
        from backend.analysis.policy import IntegritySection
        orch = Orchestrator(repository=None)
        orch._analysis_run = SimpleNamespace(get=lambda key: {'_analysis_strategy': 'fast-v4'})
        orch._cfg = SimpleNamespace(integrity=IntegritySection())
        prefix = [{'role': 'system', 'content': '偏好记忆与约束'}]
        def summarize(repo, video_id, title, llm, policy, ledger, preference):
            for i in range(100):
                messages = [{'role': 'user', 'content': f'第{i}轮预算试算'}]
                self.assertEqual(preference(messages), prefix + messages)
            return '报告'
        with patch.object(orch, '_preference_messages', return_value=prefix) as recall:
            with patch('backend.analysis.packing.summarize', side_effect=summarize):
                self.assertEqual(orch._summarize('v', '标题', 1000, AnalyzeOptions()), ('报告', 'cloud'))
        recall.assert_called_once_with([], '标题', report=True)

    def test_incremental_budget_matches_full_json_with_escaping(self):
        evidence = [{'id': f'e{i}', 'clip_id': f'c{i}', 'kind': 'ocr', 'start_ms': i * 1000,
                     'end_ms': i * 1000, 'text': f'型号A{i}价格750元，不能使用DDR4。\n路径C:\\测试\\a，代码"x\\n"。'}
                    for i in range(50)]
        def messages(rows):
            return [{'role': 'system', 'content': '偏好、主题与约束'}, {'role': 'user', 'content':
                     json.dumps({'title': '型号对比', 'evidence_units': rows}, ensure_ascii=False)}]
        for budget in (1500, 4000, 19392):
            for experiment in (False, True):
                regular, _ = select(evidence, messages, budget, keyword_assist=experiment)
                incremental, _ = select(evidence, messages, budget, incremental=True, keyword_assist=experiment)
                self.assertEqual(regular, incremental)
                self.assertLessEqual(estimate_tokens(json.dumps(messages(incremental), ensure_ascii=False)), budget)

    def test_default_does_not_run_experimental_model(self):
        from backend.analysis.policy import IntegritySection
        self.assertFalse(IntegritySection().keyword_assist)
        evidence = [{'id':'e', 'clip_id':'c', 'kind':'speech', 'start_ms':0, 'end_ms':1000,
                     'text':'5600XT网游领先32%，生产力落后12%。'}]
        with patch('backend.analysis.packing.keywords', side_effect=AssertionError('默认不得运行实验模型')):
            rows, meta = select(evidence, lambda rows: rows, 2000)
        self.assertTrue(rows)
        self.assertEqual(meta['keyword_model'], 'tfidf-v1')
        self.assertFalse(meta['keyword_assist'])

    def test_opt_in_reaches_report_without_extra_cloud_calls(self):
        from backend.analysis.packing import summarize
        from backend.analysis.policy import IntegritySection
        from backend.analysis.runs import RunLedger
        repo = SimpleNamespace(iter_clip_payloads=lambda vid: iter([{'clip_id':'c', 'start_ms':0,
            'end_ms':1000, 'text':'型号A1售价750元。不能使用DDR4。'}]),
            get_segments=lambda vid: [], get_keyframes=lambda vid: [])
        calls = []
        def complete(messages, **options):
            calls.append(messages)
            return SimpleNamespace(text='型号A1售价750元，不能使用DDR4。', finish_reason='stop')
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger(directory, 'keywords')
            text = summarize(repo, 'v', '测试视频', SimpleNamespace(complete=complete),
                             IntegritySection(keyword_assist=True), ledger)
            self.assertTrue(text)
            self.assertTrue(ledger.get('report_selection')['keyword_assist'])
            self.assertTrue(ledger.get('report_keywords'))
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
