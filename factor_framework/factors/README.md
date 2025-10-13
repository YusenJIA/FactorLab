# 预定义因子库使用指南

## 📁 目录结构

```
factor_framework/factors/
├── __init__.py        # 因子库入口，导出所有因子
├── momentum.py        # 动量类因子：RSI, MACD, Momentum等
├── liquidity.py       # 流动性因子：PriceToVolume, VolumeRatio等
├── volatility.py      # 波动率因子：ATR, BollingerBands, HistoricalVolatility等
├── value.py           # 价值因子：PriceToMA, PercentFromHigh等
└── README.md          # 本文档
```

## 🚀 快速开始

### 使用预定义因子

```python
from factor_framework import FactorEngine, FrequencyConfig
from factor_framework.factors import RSI, MACD, ATR

# 1. 创建因子实例
rsi_factor = RSI(window_length=14)

# 2. 配置频率
config = FrequencyConfig(input_freq='daily', window_length=14, calc_freq='daily')

# 3. 计算因子
engine = FactorEngine()
results = engine.compute_factor(
    rsi_factor,
    assets=['510050.SH', '510300.SH'],
    start_date='2025-01-01',
    end_date='2025-01-31',
    config=config
)
```

### 运行完整示例

```bash
python example_using_predefined_factors.py
```

## 📚 已实现因子列表

### 动量类 (momentum.py)

| 因子名称 | 函数 | 参数 | 说明 |
|---------|------|------|------|
| RSI | `RSI(window_length=14)` | window_length | 相对强弱指标 |
| MACD | `MACD(fast=12, slow=26, signal=9)` | fast, slow, signal | 指数移动平均差异 |
| Momentum | `Momentum(window_length=20)` | window_length | 简单动量 |
| ROC | `RateOfChange(window_length=10)` | window_length | 价格变化率 |

### 流动性类 (liquidity.py)

| 因子名称 | 函数 | 参数 | 说明 |
|---------|------|------|------|
| PriceToVolume | `PriceToVolume(window_length=20)` | window_length | 价格与平均成交量比 |
| VolumeRatio | `VolumeRatio(short=5, long=20)` | short, long | 成交量比率 |
| AmihudIlliquidity | `AmihudIlliquidity(window_length=20)` | window_length | Amihud非流动性指标 |
| RelativeVolume | `RelativeVolume(window_length=20)` | window_length | 相对成交量 |

### 波动率类 (volatility.py)

| 因子名称 | 函数 | 参数 | 说明 |
|---------|------|------|------|
| ATR | `ATR(window_length=14)` | window_length | 平均真实波幅 |
| BollingerBands | `BollingerBands(window_length=20, num_std=2)` | window_length, num_std | 布林带 |
| HistoricalVolatility | `HistoricalVolatility(window_length=20)` | window_length | 历史波动率（年化） |
| ParkinsonVolatility | `ParkinsonVolatility(window_length=20)` | window_length | Parkinson波动率 |

### 价值类 (value.py)

| 因子名称 | 函数 | 参数 | 说明 |
|---------|------|------|------|
| PriceToMA | `PriceToMA(window_length=20)` | window_length | 价格与均线比率 |
| PercentFromHigh | `PercentFromHigh(window_length=252)` | window_length | 距离历史高点百分比 |
| PriceRange | `PriceRange(window_length=20)` | window_length | 价格在区间中的位置 |

## ✍️ 如何添加自定义因子

### 方法1: 在现有分类中添加

编辑对应分类文件（如 `momentum.py`），添加新函数：

```python
# 在 momentum.py 中添加

def MyCustomMomentum(window_length=10, threshold=0.05):
    """
    我的自定义动量因子

    计算公式：
    CustomMom = (Close - Mean(Close, window_length)) / Std(Close, window_length)

    参数：
        window_length: 计算窗口
        threshold: 阈值参数

    返回：
        FactorNode: 自定义动量因子
    """
    close = Close()
    mean_price = Mean(close, window_length=window_length)
    std_price = Std(close, window_length=window_length)

    custom_mom = (close - mean_price) / (std_price + Constant(1e-10))

    return custom_mom
```

然后在 `__init__.py` 中导出：

```python
from .momentum import RSI, MACD, Momentum, MyCustomMomentum

__all__ = [
    'RSI',
    'MACD',
    'Momentum',
    'MyCustomMomentum',  # 新增
    # ...
]
```

### 方法2: 创建新的分类文件

创建新文件 `factor_framework/factors/my_category.py`：

```python
"""
我的自定义因子分类
"""

from factor_framework.nodes import *


def MyFactor1(param1=10):
    """我的第一个因子"""
    # 因子定义
    return Close() / Mean(Volume(), window_length=param1)


def MyFactor2(param1=20, param2=5):
    """我的第二个因子"""
    # 因子定义
    return ...
```

然后在 `__init__.py` 中导入：

```python
from .my_category import MyFactor1, MyFactor2

__all__ = [
    # 原有因子...
    'MyFactor1',
    'MyFactor2',
]
```

## 🎯 因子设计最佳实践

### 1. 遵循命名规范

```python
def FactorName(param1=default1, param2=default2):
    """
    因子简短描述

    计算公式：
    明确的数学公式

    参数：
        param1: 参数1说明
        param2: 参数2说明

    返回：
        FactorNode: 因子节点

    示例：
        >>> factor = FactorName(param1=10)
        >>> config = FrequencyConfig(...)
        >>> results = engine.compute_factor(factor, ...)
    """
    # 实现代码
    pass
```

### 2. 使用基础节点构建

所有因子都应该基于 `factor_framework.nodes` 中的基础节点构建：

```python
from factor_framework.nodes import (
    # 原子数据
    Close, Open, High, Low, Volume,

    # 数学运算
    Add, Sub, Mul, Div, Abs, Log, Sqrt, Max, Min,

    # 时间聚合
    Mean, Std, Sum, Tsmax, Tsmin, Ref, EMA,

    # 横截面运算
    Rank, Zscore,

    # 常量
    Constant
)
```

### 3. 处理边界情况

```python
# ✅ 好的做法：避免除零
avg_volume = Mean(Volume(), window_length=20)
factor = Close() / (avg_volume + Constant(1e-10))

# ❌ 不好的做法：可能除零
factor = Close() / Mean(Volume(), window_length=20)
```

### 4. 返回字典处理多个输出

```python
def MACD(fast=12, slow=26, signal=9):
    """MACD返回多个输出"""
    macd_line = ...
    signal_line = ...
    histogram = ...

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

# 使用时
macd_dict = MACD()
results = engine.compute_factor(macd_dict['macd'], ...)
```

## 🔍 因子验证清单

在添加新因子后，请确保：

- [ ] 因子有清晰的文档字符串（包括公式、参数、返回值）
- [ ] 处理了边界情况（除零、空值等）
- [ ] 在 `__init__.py` 中导出了因子
- [ ] 编写了使用示例
- [ ] 在多个资产上测试过
- [ ] 在不同频率下测试过（如果适用）

## 📖 扩展阅读

- 框架核心文档: `../README.md`
- 使用示例: `../../example_using_predefined_factors.py`
- 测试套件: `../../test_framework.py`
- 因子库管理系统设计: `../../CLAUDE.md`
