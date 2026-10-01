"""Evaluate one held-out future horizon per CSV; consolidate memories only when mature."""
import heapq
import json
from pathlib import Path
import re
import pandas as pd
from memory_agent import create_memory_agent
from run_graph import load_data, working_directory
from trading_graph import TradingGraph


def prepare_samples(folder, horizon, time_frame=None, steps=None):
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == '.csv')
    if not files:
        raise ValueError(f'No CSV files found in {folder}')
    samples = []
    for path in files:
        data = load_data(path)
        if len(data) - horizon < 70:
            raise ValueError(f'{path.name}: need >= 70 observed candles plus {horizon} future candles')
        match = re.fullmatch(r'.+_([0-9]+[A-Za-z]+)_[0-9]+', path.stem)
        interval = time_frame or (match.group(1) if match else None)
        if interval is None:
            raise ValueError(f'Cannot infer timeframe from {path.name}; supply --time-frame')
        samples.append({'path': path.resolve(), 'data': data, 'time_frame': interval,
                        'decision_time': data['Datetime'].iloc[-horizon-1],
                        'outcome_time': data['Datetime'].iloc[-1]})
    if len({s['time_frame'] for s in samples}) != 1:
        raise ValueError(f'{folder}: mixed candle intervals; use separate benchmark folders')
    # File numbers may be random sample IDs. Market timestamps determine evaluation order.
    samples.sort(key=lambda s: (s['decision_time'], s['path'].name))
    return samples if steps is None else samples[:steps]


def run_benchmark(output, benchmark_dir=None, benchmark_root=None, assets=None, symbol=None,
                  time_frame=None, horizon=3, steps=None, offline=False, mode='indicator', config=None):
    if bool(benchmark_dir) == bool(benchmark_root):
        raise ValueError('Specify exactly one benchmark-dir or benchmark-root')
    if horizon < 1 or (steps is not None and steps < 1):
        raise ValueError('horizon and steps must be positive')
    if offline and mode == 'semantic':
        raise ValueError('Offline CLI supports indicator mode only')
    if benchmark_dir:
        if assets:
            raise ValueError('--assets requires --benchmark-root')
        folders = [Path(benchmark_dir).resolve()]
    else:
        if symbol:
            raise ValueError('--symbol is only for a single benchmark folder')
        base = Path(benchmark_root).resolve()
        if assets:
            if len(set(assets)) != len(assets) or any(Path(a).name != a or a in {'.','..'} for a in assets):
                raise ValueError('Assets must be unique subfolder names')
            folders = [base/a for a in assets]
        else:
            folders = sorted(p for p in base.iterdir() if p.is_dir() and any(p.glob('*.csv')))
    if not folders or any(not p.is_dir() for p in folders):
        raise ValueError('Benchmark asset folders not found')
    # Validate every selected dataset before starting model calls or creating outputs.
    prepared = [(folder, prepare_samples(folder, horizon, time_frame, steps)) for folder in folders]
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new/empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    run_config = {**(config or {}), 'horizon': horizon, 'retrieval_mode': mode}
    if offline:
        from offline_model import OfflineModel
        engine = TradingGraph(run_config, agent_llm=OfflineModel(), graph_llm=OfflineModel())
    else:
        engine = TradingGraph(run_config)
    consolidate = create_memory_agent(engine.graph_llm)
    summary = {}
    for folder, samples in prepared:
        asset = symbol or folder.name.upper()
        dest = output/folder.name
        dest.mkdir()
        memory_file = dest/'memories.jsonl'
        rows, pending, events = [], [], []
        (dest/'config.json').write_text(json.dumps({'asset':asset, 'folder':str(folder),
            'offline':offline, 'sample_count':len(samples), 'split':'all rows except final horizon',
            'order':'decision timestamp, then filename', **engine.config}, indent=2))
        (dest/'manifest.json').write_text(json.dumps([{
            'sample_file':str(s['path']), 'decision_time':s['decision_time'],
            'outcome_time':s['outcome_time'], 'time_frame':s['time_frame'],
            'observed_rows':len(s['data'])-horizon} for s in samples], indent=2))

        def flush(until=None):
            while pending and (until is None or pending[0][0] <= until):
                available, index, state = heapq.heappop(pending)
                update = consolidate({**state, 'defer_memory_update':False})
                rows[index].update(update)
                event = {'sample_file':rows[index]['sample_file'], 'available_at':available, **update}
                events.append(event)
                with (dest/'memory_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event, allow_nan=False)+'\n')

        for index, sample in enumerate(samples):
            # Includes overlapping sample windows: only already observable outcomes are committed.
            flush(sample['decision_time'])
            data = sample['data']
            observed, full = data.iloc[:-horizon].to_dict('list'), data.to_dict('list')
            charts = dest/'charts'/sample['path'].stem
            charts.mkdir(parents=True)
            state = {'kline_data':observed, 'orig_kline_data':full, 'stock_name':asset,
                     'time_frame':sample['time_frame'], 'memory_file':str(memory_file),
                     'sample_id':sample['path'].name, 'defer_memory_update':True}
            with working_directory(charts):
                final = engine.invoke(state)
            outcome = final['trade_outcome_details']
            row = {'sample_file':sample['path'].name, 'symbol':asset,
                   'time_frame':sample['time_frame'], 'observed_rows':len(data)-horizon,
                   'timestamp':sample['decision_time'], 'outcome_timestamp':sample['outcome_time'],
                   'decision':final['final_trade_decision'], 'pnl':final['trade_outcome'],
                   'return':final['trade_outcome']/outcome['entry_price'], 'outcome':outcome,
                   'analysis':final['analysis_results'], 'reports':final['report_dict'],
                   'reflection':final['reflection'], 'retrieved_memory_count':final['retrieved_memory_count'],
                   'memory_actions':None, 'memory_count':None}
            rows.append(row)
            # Audit prediction output immediately. Final results include delayed consolidation stats.
            with (dest/'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            heapq.heappush(pending, (sample['outcome_time'], index, final))
            print(f'{asset} {index+1}/{len(samples)} {sample["path"].name}: {row["decision"]}, retrieved={row["retrieved_memory_count"]}')
        flush()
        (dest/'results.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
        pd.DataFrame([{k:v for k,v in row.items() if k not in {'reports','outcome','memory_actions'}}
                      for row in rows]).to_csv(dest/'results.csv', index=False)
        score = {'samples':len(rows), 'offline':offline,
                 'positive_pnl_fraction':sum(row['pnl']>0 for row in rows)/len(rows),
                 'mean_signed_return':sum(row['return'] for row in rows)/len(rows),
                 'final_memory_count':events[-1]['memory_count'],
                 'note':'One prediction per CSV; final horizon held out; no costs/order execution.'}
        (dest/'summary.json').write_text(json.dumps(score, indent=2))
        summary[folder.name] = score
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
