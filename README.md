# FactorLab

**A unified, production-ready quantitative factor research framework for multi-frequency trading strategies**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-success.svg)](https://github.com/yourusername/FactorLab)

[English](#english) | [中文](#中文)

---

## English

### Overview

FactorLab is a comprehensive quantitative factor research system that unifies factor calculation, analysis, and management across different time frequencies. It enables researchers to define factors once and apply them seamlessly to daily, minute, or any other frequency data.

### Key Features

- **Unified Factor Framework** (`factor_framework/`)
  - Abstract factor definition using expression trees
  - Multi-frequency support (daily, minute, tick, etc.)
  - High-performance computation engine with DAG optimization
  - Strict lookahead bias prevention

- **Factor Analysis Module** (`factor_analysis/`)
  - Single-factor analysis (IC, portfolio sorting)
  - Multi-factor analysis (correlation, combination, PCA)
  - Performance evaluation (time series, cross-section)
  - Risk testing (out-of-sample, robustness)
  - Automatic visualization generation

- **Factor Library System** (`factor_library/`)
  - Factor registration and metadata management
  - Version control with rollback support
  - Performance monitoring and decay detection
  - Factor grouping and batch operations

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Factor Calculation Framework (factor_framework/)     │
│  FactorNode → FrequencyConfig → FactorEngine                     │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────────────────────┐                  ┌───────────────────────┐
│   Factor Analysis     │                  │   Factor Library      │
│   (factor_analysis/)  │                  │   (factor_library/)   │
│                       │                  │                       │
│ • IC Analysis         │                  │ • Registration        │
│ • Portfolio Sorting   │                  │ • Version Control     │
│ • Multi-Factor        │                  │ • Monitoring          │
│ • Risk Testing        │                  │ • Grouping            │
└───────────────────────┘                  └───────────────────────┘
```

### Quick Start

#### Installation

```bash
pip install pandas numpy scipy matplotlib seaborn
```

#### Basic Example

```python
from factor_framework import FactorEngine, FrequencyConfig, Close, Volume, Mean

# 1. Define factor (frequency-agnostic)
factor = Close() / Mean(Volume(), window_length=20)

# 2. Configure frequency
config = FrequencyConfig(
    input_freq='daily',
    window_length=20,
    calc_freq='daily'
)

# 3. Compute factor values
engine = FactorEngine()
results = engine.compute_factor(
    factor,
    assets=['510050.SH', '510300.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    config=config
)
```

#### Factor Analysis

```python
from factor_analysis import FactorData, compute_ic, portfolio_sorting_test

# Convert to standard format
factor_data = FactorData.from_engine_output(
    factor_df=results,
    returns_df=returns,
    factor_name='Price-to-Volume'
)

# Analyze
ic_result = compute_ic(factor_data, method='spearman')
sorting_result = portfolio_sorting_test(factor_data, n_quantiles=5)

print(f"ICIR: {ic_result.metrics['icir']:.4f}")
print(f"Long-Short Sharpe: {sorting_result.metrics['ls_sharpe']:.2f}")
```

#### Factor Management

```python
from factor_library import FactorLibrary

# Create factor library
library = FactorLibrary(registry_path='./data/factor_registry')

# Register factor
factor_id = library.register_factor(
    factor_node=factor,
    factor_configs=[config],
    metadata={
        'name': 'Price-to-Volume Ratio',
        'category': 'liquidity',
        'description': 'Close price divided by average volume',
        'author': 'your_name',
        'tags': ['liquidity', 'volume']
    }
)

# Monitor performance
performance = library.monitor_factor(
    factor_id=factor_id,
    assets=['510050.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### Project Structure

```
FactorLab/
├── factor_framework/          # Core computation framework
│   ├── __init__.py
│   ├── config.py             # FrequencyConfig
│   ├── engine.py             # FactorEngine
│   ├── data_loader.py        # Data interface
│   └── nodes/                # Factor nodes
│       ├── base.py           # Base classes
│       ├── atomic.py         # Atomic data nodes
│       ├── math_ops.py       # Math operations
│       ├── time_ops.py       # Time aggregations
│       └── cross_ops.py      # Cross-sectional ops
│
├── factor_analysis/           # Analysis module
│   ├── __init__.py
│   ├── core.py               # FactorData, AnalysisResult
│   ├── univariate/           # Single-factor analysis
│   ├── multivariate/         # Multi-factor analysis
│   ├── performance/          # Performance evaluation
│   ├── risk/                 # Risk testing
│   └── utils/                # Utilities
│
└── factor_library/            # Management system
    ├── __init__.py
    ├── metadata.py           # Metadata definitions
    ├── registry.py           # Factor registry
    ├── version_control.py    # Version control
    ├── monitor.py            # Performance monitoring
    ├── groups.py             # Factor groups
    └── utils.py              # Utilities
```

### Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0 (optional)

### Documentation

For detailed documentation, please refer to the `docs/` directory:
- Factor Framework Guide
- Factor Analysis Guide
- Factor Library Guide
- API Reference

### Examples

Check the `examples/` directory for complete working examples:
- `example_price_to_avg_volume.py` - Factor framework example
- `example_factor_analysis.py` - Factor analysis example
- `example_factor_library_usage.py` - Factor library example

### Testing

Run unit tests:

```bash
python tests/test_framework.py
python tests/test_factor_library.py
```

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

Developed with Claude Code in January-October 2025.

---

## 中文

### 项目简介

FactorLab 是一个生产级的量化因子研究系统，统一了不同时间频率下的因子计算、分析和管理。研究人员可以定义一次因子，无缝应用于日频、分钟频或任何其他频率的数据。

### 核心特性

- **统一因子计算框架** (`factor_framework/`)
  - 使用表达式树进行抽象因子定义
  - 多频率支持（日线、分钟线、tick级等）
  - 高性能计算引擎，支持DAG优化
  - 严格的未来函数检查

- **因子分析模块** (`factor_analysis/`)
  - 单因子分析（IC分析、分组回测）
  - 多因子分析（相关性、组合、主成分分析）
  - 表现评估（时间序列、横截面分析）
  - 风险检验（样本外测试、鲁棒性检验）
  - 自动生成可视化图表

- **因子库管理系统** (`factor_library/`)
  - 因子注册与元数据管理
  - 版本控制与回滚支持
  - 性能监控与衰减检测
  - 因子分组与批量操作

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│              因子计算框架 (factor_framework/)                     │
│  FactorNode → FrequencyConfig → FactorEngine                     │
└─────────────────────────────────────────────────────────────────┘
                                ▲
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────────────────────┐                  ┌───────────────────────┐
│   因子分析模块         │                  │   因子库管理系统       │
│   (factor_analysis/)  │                  │   (factor_library/)   │
│                       │                  │                       │
│ • IC分析              │                  │ • 因子注册            │
│ • 分组回测            │                  │ • 版本控制            │
│ • 多因子分析          │                  │ • 性能监控            │
│ • 风险检验            │                  │ • 组合管理            │
└───────────────────────┘                  └───────────────────────┘
```

### 快速开始

#### 安装依赖

```bash
pip install pandas numpy scipy matplotlib seaborn
```

#### 基础示例

```python
from factor_framework import FactorEngine, FrequencyConfig, Close, Volume, Mean

# 1. 定义因子（与频率无关）
factor = Close() / Mean(Volume(), window_length=20)

# 2. 配置频率
config = FrequencyConfig(
    input_freq='daily',      # 日线数据
    window_length=20,        # 20期窗口
    calc_freq='daily'        # 每日计算
)

# 3. 计算因子值
engine = FactorEngine()
results = engine.compute_factor(
    factor,
    assets=['510050.SH', '510300.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31',
    config=config
)
```

#### 因子分析

```python
from factor_analysis import FactorData, compute_ic, portfolio_sorting_test

# 转换为标准格式
factor_data = FactorData.from_engine_output(
    factor_df=results,
    returns_df=returns,
    factor_name='价格成交量比'
)

# 执行分析
ic_result = compute_ic(factor_data, method='spearman')
sorting_result = portfolio_sorting_test(factor_data, n_quantiles=5)

print(f"ICIR: {ic_result.metrics['icir']:.4f}")
print(f"多空夏普比率: {sorting_result.metrics['ls_sharpe']:.2f}")
```

#### 因子管理

```python
from factor_library import FactorLibrary

# 创建因子库
library = FactorLibrary(registry_path='./data/factor_registry')

# 注册因子
factor_id = library.register_factor(
    factor_node=factor,
    factor_configs=[config],
    metadata={
        'name': '价格成交量比',
        'category': 'liquidity',
        'description': '收盘价除以平均成交量',
        'author': 'your_name',
        'tags': ['liquidity', 'volume']
    }
)

# 监控性能
performance = library.monitor_factor(
    factor_id=factor_id,
    assets=['510050.SH'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### 项目结构

```
FactorLab/
├── factor_framework/          # 核心计算框架
│   ├── __init__.py
│   ├── config.py             # 频率配置
│   ├── engine.py             # 计算引擎
│   ├── data_loader.py        # 数据接口
│   └── nodes/                # 因子节点
│       ├── base.py           # 基类
│       ├── atomic.py         # 原子数据节点
│       ├── math_ops.py       # 数学运算
│       ├── time_ops.py       # 时间聚合
│       └── cross_ops.py      # 横截面运算
│
├── factor_analysis/           # 分析模块
│   ├── __init__.py
│   ├── core.py               # 核心数据结构
│   ├── univariate/           # 单因子分析
│   ├── multivariate/         # 多因子分析
│   ├── performance/          # 表现评估
│   ├── risk/                 # 风险检验
│   └── utils/                # 工具函数
│
└── factor_library/            # 管理系统
    ├── __init__.py
    ├── metadata.py           # 元数据定义
    ├── registry.py           # 因子注册
    ├── version_control.py    # 版本控制
    ├── monitor.py            # 性能监控
    ├── groups.py             # 组合管理
    └── utils.py              # 工具函数
```

### 技术特点

1. **代码复用性**：同一因子定义可用于不同频率
2. **类型安全**：完整的类型提示和验证
3. **性能优化**：缓存机制、DAG优化、并行计算
4. **未来函数检查**：严格防止使用未来数据
5. **扩展性强**：易于添加新的运算节点和数据源

### 系统要求

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0（可选）

### 使用文档

详细文档请参阅 `docs/` 目录：
- 因子框架指南
- 因子分析指南
- 因子库管理指南
- API参考文档

### 完整示例

查看 `examples/` 目录获取完整的可运行示例：
- `example_price_to_avg_volume.py` - 因子计算框架示例
- `example_factor_analysis.py` - 因子分析模块示例
- `example_factor_library_usage.py` - 因子库管理系统示例

### 测试

运行单元测试：

```bash
python tests/test_framework.py
python tests/test_factor_library.py
```

### 性能指标解读

#### ICIR (Information Coefficient IR)
- **ICIR > 0.5**: 因子稳定性强，预测能力优秀
- **ICIR 0.3-0.5**: 因子有一定预测能力
- **ICIR < 0.3**: 因子预测能力较弱

#### Long-Short Sharpe Ratio
- **夏普 > 1.0**: 因子区分能力强，风险调整后收益优秀
- **夏普 0.5-1.0**: 因子有一定区分能力
- **夏普 < 0.5**: 因子区分能力较弱

### 开发路线图

#### 已完成功能 (v1.0.0)
- [x] 统一因子计算框架
- [x] 多频率支持
- [x] 因子分析模块（单因子、多因子、风险检验）
- [x] 因子库管理系统（注册、版本控制、监控）
- [x] 完整的单元测试

#### 规划中功能 (v2.0.0)
- [ ] 策略回测引擎
- [ ] 更多技术指标节点（RSI, MACD, Bollinger Bands等）
- [ ] 机器学习因子节点
- [ ] Web可视化界面
- [ ] 实时数据接口
- [ ] 分布式计算支持

### 常见问题

**Q: 如何准备收益率数据？**

A: 从价格数据计算收益率：

```python
returns_df = price_df.copy()
returns_df['return'] = returns_df.groupby('asset')['close'].pct_change()
returns_df = returns_df.dropna()
```

**Q: Pearson 和 Spearman IC 有什么区别？**

A:
- **Pearson**: 衡量线性相关，对异常值敏感
- **Spearman**: 衡量单调相关（Rank IC），对异常值鲁棒，实践中更常用

**Q: 如何添加自定义因子节点？**

A: 继承 `FactorNode` 基类并实现 `compute` 方法：

```python
from factor_framework.nodes.base import FactorNode

class MyCustomNode(FactorNode):
    def compute(self, data, config, timestamp):
        # 实现你的计算逻辑
        return result
```

### 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

### 致谢

本项目由 Claude Code 在 2025年1月至10月期间开发完成。

### 联系方式

- 提交问题：[GitHub Issues](https://github.com/yourusername/FactorLab/issues)
- 邮件联系：your.email@example.com

---

**版本**: v1.0.0
**状态**: 生产就绪
**最后更新**: 2025年10月
