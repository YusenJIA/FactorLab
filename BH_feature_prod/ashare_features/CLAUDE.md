# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A real-time and offline feature engineering system for Chinese A-share stocks (~5,200 symbols). It computes 574 behavioral features from 1-minute OHLCV bars using 7 specialized processors, producing 582-column output (8 base + 574 features).

## Running

**Realtime** (production, runs during market hours 09:30-15:00):
```bash
# TRADE_DATE can be provided by environment; defaults are production paths
TRADE_DATE=2026-06-15 ./run_bh_prod.sh instant
# Or directly:
TRADE_DATE=2026-06-15 python ${ASHARE_FEATURE_HOME:-/home/yusen/ashare_feature}/realtime_feature_cmp.py
```

**Offline** (batch computation for a full trading day):
```bash
# Run offline_feature_cmp.ipynb — update Config.TRADE_DATE in the notebook first
```

**Validation** (compare offline vs realtime output):
```bash
# Run validation_test.ipynb — update TRADE_DATE variable
```

## Architecture

### Two computation paths, same pipeline

- **`realtime_feature_cmp.py`** — Imports `FeaturePipeline` from `factor_engine`. Runs incrementally: loads previous 5 trading days for context, then processes each new 1-minute bar as it arrives. Current-day kbar CSV files are the primary source, with JQData as fallback. Outputs one parquet file per timestamp to `/data/BH/{TRADE_DATE}/`.
- **`offline_feature_cmp.ipynb`** — Imports `FeaturePipeline` from `factor_engine_lazy`. Runs batch on a complete trading day. Outputs a single parquet to `offline_features/{TRADE_DATE}.parquet`.

The two pipelines use different `factor_engine` variants (`factor_engine` vs `factor_engine_lazy`) which can produce minor numerical differences, particularly in 241-minute window features and features involving division that can produce inf/-inf values.

### Data flow

1. **Auth**: `dataloader.api_base.api_conf.auth(username='sihang', password='sihang123', env='ali')`
2. **Symbol pool**: `get_all_securities(types='stock', date=TRADE_DATE)` returns ~5,200 A-share stocks
3. **Historical context**: Previous 5 trading days' data loaded (auction snapshot + 1-min bars)
4. **Current day**: Level2 auction snapshot for 09:30 open + incremental 1-min bars from CSV first, JQData fallback second
5. **Feature computation**: `FeaturePipeline.run(df)` where df has columns renamed `symbol->code`, `time->datetime`
6. **Output**: Rename back `code->symbol`, `datetime->time`, write parquet

### Column conventions

- Pipeline input: `datetime, code, open, close, high, low, volume, money` (all Float64, code as Utf8)
- Pipeline output: 582 columns, features as Float32
- Symbol format: 6-digit string (e.g., "000001"), not exchange-suffixed. Symbols starting with '2' or '9' are filtered out (non-A-shares).

### Key dependencies

`realtime_feature_cmp.py` prepends `ASHARE_PYTHONPATH_ROOT` to `sys.path`, defaulting to `/home/yusen`.

- `factor_engine` / `factor_engine_lazy` — Feature pipeline with 7 processors: RoundNumberProcessor, FOMOFUDProcessor, RetailPatternProcessor, HerdingProcessor, MicrostructureProcessor, SentimentCycleProcessor, AttentionProcessor
- `dataloader.jqdata` — `get_price()`, `get_open_price()`, `get_all_securities()`
- `polars` is the primary DataFrame library; `pandas` is used only for API return conversion

### Realtime-specific design

- **Checkpoint-restart**: Scans output parquet filenames (e.g., `093500.parquet`) to find last processed time, then fills gaps
- **CSV primary source**: Polls `/data/shenrun/dump_1m_kbar/{TRADE_DATE}/kbar_HHMM.csv` for a short tolerance window
- **JQData fallback**: Retries up to 30 times when the CSV source is missing or incomplete
- **Lunch break handling**: `next_bar_time()` jumps from 11:30 to 13:01
- **`get_prev_trade_date()`** uses `get_trade_days()` and caches the returned trading calendar

### Path configuration

Production data-path defaults remain unchanged:
- `ASHARE_OUTPUT_DIR` defaults to `/data/BH`
- `ASHARE_KBAR_CSV_ROOT` defaults to `/data/shenrun/dump_1m_kbar`

Home/project paths can be overridden for local or nonstandard deployments:
- `ASHARE_FEATURE_HOME` defaults to `/home/yusen/ashare_feature`
- `ASHARE_PYTHONPATH_ROOT` defaults to `/home/yusen`
- `ASHARE_PYTHON_BIN` defaults to `/home/yusen/miniconda3/envs/yusen/bin/python`

### Known data quality issues

- `social_attention_*` features produce inf/-inf from division by zero in rolling correlation when volume is zero
- `volume_fomo_241min` and `panic_sell_241min` show significant offline/online divergence due to 241-minute window initialization differences
- 19 of 582 features show some discrepancy between offline and realtime computation (most >99.9% equal)

## File layout

- `realtime_feature_cmp.py` — Production realtime engine (single-file, ~830 lines)
- `run_test_api.sh` — nohup launcher, logs to `logs/{date}_realtime_feature.log`
- `offline_feature_cmp.ipynb` — Batch feature computation
- `validation_test.ipynb` — Offline vs realtime consistency comparison
- `test.ipynb` — Data exploration / debugging
- `logs/` — Execution logs
- `realtime_features/` — Per-bar computation timing records (CSV)
- `offline_features/` — Offline batch output (parquet, ~2GB/day)

