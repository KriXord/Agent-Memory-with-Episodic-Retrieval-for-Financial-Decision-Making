import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from benchmark_runner import run_benchmark, prepare_samples
from run_graph import load_data

ROOT = Path(__file__).resolve().parents[1]

class BenchmarkTests(unittest.TestCase):
    def test_folder_order_overlap_and_asset_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'benchmark'
            btc, nq = root/'btc', root/'nq'
            btc.mkdir(parents=True); nq.mkdir()
            data = pd.read_csv(ROOT/'examples/synthetic_ohlcv.csv')
            data.iloc[:100].to_csv(btc/'BTC_4h_10.csv', index=False)
            data.iloc[1:101].to_csv(btc/'BTC_4h_2.csv', index=False)
            data.iloc[4:104].to_csv(btc/'BTC_4h_1.csv', index=False)
            data.iloc[:100].to_csv(nq/'NQ_4h_1.csv', index=False)
            summary = run_benchmark(benchmark_root=root, assets=['btc','nq'],
                                    output=Path(temp)/'results', offline=True,
                                    config={'memory_config': {'similarity_threshold': -1.0}})
            self.assertEqual(summary['btc']['samples'], 3)
            self.assertEqual(summary['nq']['samples'], 1)
            rows = [json.loads(line) for line in (Path(temp)/'results/btc/results.jsonl').read_text().splitlines()]
            self.assertEqual([r['sample_file'] for r in rows], ['BTC_4h_10.csv','BTC_4h_2.csv','BTC_4h_1.csv'])
            self.assertEqual([r['observed_rows'] for r in rows], [97]*3)
            self.assertEqual([r['retrieved_memory_count'] for r in rows[:2]], [0,0])
            self.assertEqual(rows[2]['retrieved_memory_count'], 2)
            other = json.loads((Path(temp)/'results/nq/results.jsonl').read_text())
            self.assertEqual(other['retrieved_memory_count'], 0)
            memories = [json.loads(line) for line in (Path(temp)/'results/btc/memories.jsonl').read_text().splitlines()]
            self.assertEqual(len(memories),3)
            self.assertEqual({m['metadata']['symbol'] for m in memories}, {'BTC'})
            self.assertTrue(all(r['memory_actions']['ADD']==1 for r in rows))

    def test_csv_aliases_and_epoch_ms(self):
        with tempfile.TemporaryDirectory() as temp:
            data = pd.read_csv(ROOT/'examples/synthetic_ohlcv.csv')
            data.columns = [c.lower() for c in data.columns]
            data['timestamp'] = pd.to_datetime(data.pop('datetime')).astype('int64')//1_000_000
            p = Path(temp)/'sample.csv';data.to_csv(p,index=False)
            loaded = load_data(p)
            self.assertEqual(loaded['Datetime'].iloc[0], '2025-01-01 00:00:00')
            self.assertIn('Close',loaded)

    def test_invalid_folder_is_checked_before_models(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'btc';folder.mkdir()
            pd.read_csv(ROOT/'examples/synthetic_ohlcv.csv').iloc[:20].to_csv(folder/'BTC_4h_1.csv',index=False)
            with patch('benchmark_runner.TradingGraph') as model:
                with self.assertRaisesRegex(ValueError,'observed candles'):
                    run_benchmark(benchmark_dir=folder, output=Path(temp)/'output')
                model.assert_not_called()

    def test_duplicate_decision_timestamps_have_distinct_sample_ids(self):
        # The memory node includes the sample filename in its identifier.
        from memory_agent import create_memory_agent
        from offline_model import OfflineModel
        with tempfile.TemporaryDirectory() as temp:
            data = load_data(ROOT/'examples/synthetic_ohlcv.csv').iloc[:100]
            state = {'kline_data':data.iloc[:-3].to_dict('list'),'orig_kline_data':data.to_dict('list'),
                'stock_name':'BTC','time_frame':'4h','memory_file':str(Path(temp)/'m.jsonl'),
                'report_dict':{'indicator':'fixture'},'final_trade_decision':'LONG','trade_outcome':1,
                'analysis_results':'fixture','reflection':'fixture'}
            node=create_memory_agent(OfflineModel())
            node({**state,'sample_id':'BTC_4h_1.csv'})
            result=node({**state,'sample_id':'BTC_4h_2.csv'})
            self.assertEqual(result['memory_count'],2)
