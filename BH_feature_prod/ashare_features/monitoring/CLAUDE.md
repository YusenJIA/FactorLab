# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A lightweight monitoring subsystem for the A-share feature engineering pipeline. It records per-bar computation metrics, abandon events, and feature-level statistics into a SQLite database during realtime feature computation. Zero external dependencies — uses only Python stdlib (`sqlite3`, `smtplib`, `logging`).

This module is consumed by `realtime_feature_cmp.py` in the parent project. It does **not** import any project modules (`factor_engine`, `dataloader`, etc.) — that boundary is intentional.

## Running

```bash
# Run the demo (simulates a full trading day of metrics, from ashare_feature/):
python -m monitoring.demo

# Or directly:
python monitoring/demo.py
```

There are no tests, linter config, or build steps in this module.

## Architecture

### Module structure

- **`config.py`** — All configuration: DB path, email settings, tracked feature list. DB path defaults to `../monitor.db` (parent `ashare_feature/` directory), overridable via `MONITOR_DB_PATH` env var.
- **`collector.py`** — `MetricsCollector` class. Instantiated once per trading day with a `trade_date` string. Holds a single SQLite connection (WAL mode). Provides `record_bar()`, `record_abandon()`, and `record_feature_stats()`.
- **`alert.py`** — `send_abandon_alert()` sends email via SMTP_SSL when a bar is abandoned. Only fires if `EMAIL_ENABLED=True` and credentials are configured in `config.py`.
- **`demo.py`** — Standalone demo that generates synthetic data for all 240 bars of a trading day. Creates its own `demo_monitor.db` in this directory.

### SQLite schema (5 tables)

| Table | Purpose | Primary Key |
|---|---|---|
| `bar_metrics` | Per-bar cost, symbol count, null/inf counts, retry count, ok/abandon status | `(trade_date, bar_time)` |
| `abandon_events` | Detailed abandon records with reason and email-sent flag | autoincrement `id` |
| `feature_stats` | Per-bar descriptive stats (mean/std/min/max/null/inf) for tracked features | `(trade_date, bar_time, feature_name)` |
| `validation_summary` | Per-day offline-vs-realtime comparison summary | `trade_date` |
| `validation_columns` | Per-feature divergence details (equal_ratio, max_abs_diff, correlation) | `(trade_date, feature_name)` |

Tables are auto-created on `MetricsCollector.__init__`.

### Integration point

The parent `realtime_feature_cmp.py` calls:
1. `MetricsCollector(trade_date)` — once at start
2. `record_bar(bar_time, cost_seconds, n_symbols, retry_count, status, df_feat)` — after each bar
3. `record_feature_stats(bar_time, df_feat)` — after each bar (optional, for tracked features)
4. `record_abandon(bar_time, retry_count, reason)` — when a bar is abandoned after max retries

`df_feat` is a **polars** DataFrame. The collector uses polars-specific APIs (`null_count()`, `is_infinite()`, `drop_nulls()`, `filter()`) but does not import polars itself — it operates on the passed DataFrame duck-typing style.

### Design constraints

- **No polars/pandas import** — the module must remain zero-dependency so it never interferes with the main pipeline's import chain.
- **`bar_time` flexibility** — accepts both `datetime` objects and strings; auto-converts via `_to_time_str()`.
- **`INSERT OR REPLACE`** — bar_metrics and feature_stats are idempotent on re-runs for the same `(trade_date, bar_time)`.

## Config: tracked features

`config.TRACKED_FEATURES` lists ~12 features selected to represent each of the 7 processors plus known-problematic features (inf-producing `social_attention_*`, divergent `volume_fomo_241min` / `panic_sell_241min`). Only these get per-bar statistics in `feature_stats`.
