import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from memory import TradingMemory, TradingMemorySystem
from graph_util import TechnicalTools
from run_graph import run, load_data
from offline_model import OfflineModel
from strategy_agents.vwap_agent import create_vwap_agent
from run_graph import working_directory

ROOT = Path(__file__).resolve().parents[1]

class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'memory.jsonl'
        self.store = TradingMemorySystem(memory_file=self.path)

    def current(self):
        return TradingMemory('2025-01-02', np.array([1., 0.]), 'New analysis',
                             trade_action='SHORT', trade_outcome=-2., reflection='New lesson')

    def test_cosine_threshold_and_top_k(self):
        self.store.add_memory([1, 0], 'aligned', timestamp='1')
        self.store.add_memory([0, 1], 'orthogonal', timestamp='2')
        self.store.add_memory([-1, 0], 'opposite', timestamp='3')
        result = self.store.retrieve_similar_memories([1, 0], top_k=1)
        self.assertEqual(result[0][0].timestamp, '1')
        self.assertAlmostEqual(result[0][1], 1)
        self.assertEqual(self.store.retrieve_similar_memories([0, 0]), [])

    def test_roundtrip_reflection(self):
        self.store.add_memory([1, 0], 'past', reflection='Remember this', timestamp='1')
        self.store.save_memories()
        loaded = TradingMemorySystem(memory_file=self.path)
        self.assertEqual(loaded.memories[0].reflection, 'Remember this')
        np.testing.assert_array_equal(loaded.memories[0].market_vector, [1, 0])

    def test_management_update_add_preserves_facts(self):
        self.store.add_memory([1, 0], 'old', trade_action='LONG', trade_outcome=7, timestamp='1')
        counts = self.store.apply_memory_updates(json.dumps({'memory': [
            {'id':'0','event':'UPDATE','analysis_text':'refined','reflection':'lesson','trade_outcome':99},
            {'id':'NEW','event':'ADD'}]}), {'0':'1'}, self.current())
        self.assertEqual(counts['UPDATE'], 1)
        self.assertEqual(counts['ADD'], 1)
        self.assertEqual(self.store.memories[0].trade_outcome, 7)
        self.assertEqual(self.store.memories[0].reflection, 'lesson')
        self.assertEqual(self.store.memories[1].reflection, 'New lesson')

    def test_delete_and_none(self):
        self.store.add_memory([1, 0], 'old', timestamp='1')
        counts = self.store.apply_memory_updates(json.dumps({'memory': [
            {'id':'0','event':'DELETE'}, {'id':'NEW','event':'NONE'}]}), {'0':'1'}, self.current())
        self.assertEqual(counts['DELETE'], 1)
        self.assertEqual(self.store.memories, [])

    def test_invalid_transaction_is_not_partially_applied(self):
        self.store.add_memory([1, 0], 'old', timestamp='1')
        self.store.save_memories()
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.apply_memory_updates(json.dumps({'memory': [
                {'id':'0','event':'DELETE'}, {'id':'unknown','event':'UPDATE'},
                {'id':'NEW','event':'ADD'}]}), {'0':'1'}, self.current())
        self.assertEqual(len(self.store.memories), 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_vectors_and_dimensions(self):
        with self.assertRaises(ValueError):
            self.store.add_memory([float('nan')], 'bad')
        self.store.add_memory([1, 2], 'okay')
        with self.assertRaises(ValueError):
            self.store.retrieve_similar_memories([1, 2, 3])

    def test_corrupted_jsonl_is_not_silently_reset(self):
        self.path.write_text('broken')
        with self.assertRaises(ValueError):
            TradingMemorySystem(memory_file=self.path)
        self.assertEqual(self.path.read_text(), 'broken')

    def test_semantic_retrieval_uses_text(self):
        class Embeddings:
            def __init__(self): self.texts=[]
            def embed_query(self, text):
                self.texts.append(text)
                return [1, 0] if 'bull' in text else [0, 1]
        embeddings = Embeddings()
        store = TradingMemorySystem(memory_file=self.path, embeddings=embeddings)
        store.add_memory([0, 1], 'bull market', timestamp='1')
        store.add_memory([1, 0], 'bear market', timestamp='2')
        results = store.retrieve_similar_memories([1, 0], {'rsi':'bull signal'}, mode='semantic')
        self.assertEqual(results[0][0].timestamp, '1')
        self.assertIn('rsi: bull signal', embeddings.texts)

    def test_capacity(self):
        store = TradingMemorySystem(memory_file=self.path, max_memories=2)
        for i in range(3): store.add_memory([1, 0], str(i), timestamp=str(i))
        self.assertEqual([m.timestamp for m in store.memories], ['1','2'])

    def test_single_memory_optional_standardization(self):
        self.store.normalize_vectors = True
        self.store.add_memory([1, 2], 'one')
        self.assertAlmostEqual(self.store.retrieve_similar_memories([1, 2])[0][1], 1)

class WorkflowTests(unittest.TestCase):
    def test_real_indicator_vector_and_flat_series(self):
        data = load_data(ROOT/'examples/synthetic_ohlcv.csv').iloc[:90].to_dict('list')
        vector = np.concatenate([TechnicalTools.compute_all_trading_indicators(data),
                                 TechnicalTools.compute_normalized_ohlcv_vector(data)])
        self.assertEqual(vector.shape, (600,))
        self.assertTrue(np.isfinite(vector).all())
        np.testing.assert_array_equal(TechnicalTools.standard_normalize([3,3,3]), [0,0,0])

    def test_outcome_entry_and_complete_horizon(self):
        result = TechnicalTools.simulate_trade_outcome({'Close':[10,12,15,18]}, 1, 'LONG', 2)
        self.assertEqual(result['entry_price'], 12)
        self.assertEqual(result['exit_price'], 18)
        self.assertEqual(result['pnl'], 6)
        invalid = TechnicalTools.simulate_trade_outcome({'Close':[10,12]}, 1, 'LONG', 2)
        self.assertEqual(invalid['outcome'], 'INVALID')

    def test_whole_graph_two_steps_and_vwap(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            results = run(ROOT/'examples/synthetic_ohlcv.csv', output, offline=True, steps=2)
            self.assertEqual([r['memory_count'] for r in results], [1,2])
            self.assertEqual(results[0]['retrieved_memory_count'], 0)
            self.assertGreater(results[1]['retrieved_memory_count'], 0)
            self.assertGreaterEqual(results[1]['timestamp'], results[0]['outcome_timestamp'])
            self.assertEqual(len(results[0]['reports']), 11)
            self.assertEqual(len(list((output/'charts'/'step_00000').glob('*.png'))), 10)
            memories = TradingMemorySystem(memory_file=output/'memories.jsonl').memories
            self.assertEqual(memories[0].metadata['available_at'], results[0]['outcome_timestamp'])
            self.assertTrue(memories[0].reflection)
            with self.assertRaises(ValueError):
                run(ROOT/'examples/synthetic_ohlcv.csv', output, offline=True, steps=1)
            data = load_data(ROOT/'examples/synthetic_ohlcv.csv').iloc[:90].to_dict('list')
            with working_directory(Path(tmp)):
                result = create_vwap_agent(OfflineModel(), OfflineModel(), TechnicalTools())(
                    {'kline_data':data,'time_frame':'4h'})
            self.assertTrue(result['vwap_image'])
            self.assertTrue(result['vwap_report'])

if __name__ == '__main__':
    unittest.main()
