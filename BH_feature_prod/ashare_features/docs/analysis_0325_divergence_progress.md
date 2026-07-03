# 2026-03-25 RT vs Offline 特征差异分析进度

**日期**: 2026-03-26
**分析 notebook**: `/home/yusen/ashare_feature/analysis_0325_rt_vs_offline.ipynb`
**状态**: 分析进行中，Section 13（Float32 精度分析）待运行确认结果

---

## 问题背景

03-25 validation 发现以下特征 equal_ratio < 98%（来自 monitor.db）：

| 特征 | equal_ratio | max_abs_diff |
|------|-------------|-------------|
| attention_dispersion_241min | 89.7% | 279.08 |
| attention_dispersion_60min | 92.8% | 1100.45 |
| info_shock_absorption | 93.4% | 19.70 |
| fear_greed_241min | 96.9% | 2.07 |
| fomo_surge_241min | 96.9% | 2.07 |
| attention_dispersion_15min | 97.6% | 3004.33 |

## 已完成的分析（Section 1-12）

### 关键发现

1. **基础 OHLCV 差异很小**：open/close/high/low 差异 <0.2%，money 差异 7%（15:00 收盘 bar 为主）

2. **差异非 inf/nan 导致**：所有问题特征的 RT/OFF 两边均无 inf/nan/null

3. **差异非 OHLCV 数据源导致**：96.7% 的特征差异行 OHLCV 完全相同，是纯计算差异

4. **09:30 无差异，随时间发散**：所有特征在 09:30 的 equal_ratio=100%，差异随 bar 数增加逐渐累积
   - attention_dispersion_241min 午盘差异 27%（早盘仅 0.84%）
   - 15:00 最后一根 bar 差异最大（74%）

5. **fear_greed_241min 和 fomo_surge_241min 完全同步**：差异出现在完全相同的 (code, datetime) 对，确认来自共同分母 `ret_rolling_std_241`

6. **差异股票数���**：
   - attention_dispersion: 3880/5193 股票有差异（普遍性问题）
   - fear_greed/fomo_surge: 789/5193（少数股票）

7. **跨日期不稳定**：03-12/03-16 为 100%（当天跑 offline），其他日期（事后补跑）差异较大

### 特征公式

- `attention_dispersion_{N}min` = `rolling_std(volume / ((high-low)/(close+EPS) + EPS), window=N)`
- `fear_greed_{N}min` = `close_ret_N / (ret_rolling_std_N + EPS)`
- `fomo_surge_{N}min` = `close_ret_N / (ret_rolling_std_241 + EPS)`
- `info_shock_absorption` = `rolling_mean(|volume.pct_change()| / (|close_ret_1| + EPS), window=60)`

### 两个引擎的实现差异

- **factor_engine (RT)**: `group_by('code').agg()` + `.last()` 模式，rolling 在 group_by context 中
- **factor_engine_lazy (OFF)**: `.rolling_std().over('code')` 行级模式

## 待确认：Float32 精度假说（Section 13-14）

**核心假说**：这些特征存储为 Float32，值域很大（attention_dispersion 均值 ~1800），绝对阈值 1e-3 可能把浮点精度噪声误判为真实差异。

### 已添加但待运行的分析 cell

1. **13a** - 各特征值域统计 + Float32 精度上限（f32_eps@mean）
2. **13b** - 绝对阈值 vs 相对阈值对比：同一特征在 abs>1e-3 / abs>0.01 / abs>0.1 / rel>1e-4 / rel>1e-3 / rel>1e-2 下的差异行数
3. **13c** - attention_dispersion_241min 按特征值 magnitude 分桶，看 abs_diff 是否与 magnitude 成正比（精度问题特征），还是 rel_diff 也很大（真实计算差异）

### 预期结论分支

- **如果 rel_diff 普遍 < 1e-5**：差异主要是 Float32 精度导致，建议 validation 改用相对阈值
- **如果 rel_diff > 1e-3**：存在真实计算路径差异，需统一两个 factor_engine 的 rolling 实现

## 下一步（收盘后）

1. 重新 Run All notebook，确认 Section 13 结果
2. 根据结果判断是精度问题还是真实差异
3. 如果是精度问题：修改 `monitoring/` 中的 validation 逻辑，改用相对阈值
4. 如果是真实差异：将这些特征加入 KNOWN_DIVERGENT_FEATURES，或统一引擎实现
