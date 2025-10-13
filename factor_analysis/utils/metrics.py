"""
性能指标计算函数

提供各种因子和投资组合的性能指标计算。

主要功能：
- compute_ic_metrics: IC相关指标
- compute_returns_metrics: 收益率指标（年化收益、夏普、回撤等）
- compute_turnover: 换手率
- compute_information_ratio: 信息比率
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats
import warnings


def compute_ic_metrics(ic_series: pd.Series,
                      confidence_level: float = 0.95) -> Dict[str, float]:
    """
    计算 IC 相关指标

    Args:
        ic_series: IC 时间序列
        confidence_level: 置信水平（用于计算置信区间）

    Returns:
        指标字典，包含：
        - ic_mean: IC均值
        - ic_std: IC标准差
        - icir: IC信息比率 = mean(IC) / std(IC)
        - ic_positive_ratio: IC > 0 的比例
        - ic_abs_mean: |IC| 的均值
        - t_statistic: t统计量
        - p_value: p值
        - confidence_interval: 置信区间

    Example:
        >>> ic_metrics = compute_ic_metrics(ic_series)
        >>> print(f"ICIR: {ic_metrics['icir']:.4f}")
    """
    # 过滤掉 NaN
    valid_ic = ic_series.dropna()

    if len(valid_ic) == 0:
        warnings.warn("IC序列为空，返回空指标")
        return {
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'ic_positive_ratio': np.nan,
            'ic_abs_mean': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'confidence_lower': np.nan,
            'confidence_upper': np.nan
        }

    ic_mean = valid_ic.mean()
    ic_std = valid_ic.std()
    n = len(valid_ic)

    # ICIR
    icir = ic_mean / ic_std if ic_std > 0 else 0

    # 正值比例
    ic_positive_ratio = (valid_ic > 0).mean()

    # 绝对值均值
    ic_abs_mean = valid_ic.abs().mean()

    # t统计量和p值
    t_statistic = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), n - 1))

    # 置信区间
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, n - 1)
    margin_of_error = t_critical * (ic_std / np.sqrt(n))
    confidence_lower = ic_mean - margin_of_error
    confidence_upper = ic_mean + margin_of_error

    return {
        'ic_mean': float(ic_mean),
        'ic_std': float(ic_std),
        'icir': float(icir),
        'ic_positive_ratio': float(ic_positive_ratio),
        'ic_abs_mean': float(ic_abs_mean),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'confidence_lower': float(confidence_lower),
        'confidence_upper': float(confidence_upper)
    }


def compute_returns_metrics(returns: pd.Series,
                           risk_free_rate: float = 0.0,
                           periods_per_year: int = 252) -> Dict[str, float]:
    """
    计算收益率相关指标

    Args:
        returns: 收益率序列（日频）
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年的周期数（日频=252, 周频=52）

    Returns:
        指标字典，包含：
        - total_return: 总收益率
        - annual_return: 年化收益率
        - annual_volatility: 年化波动率
        - sharpe_ratio: 夏普比率
        - max_drawdown: 最大回撤
        - calmar_ratio: Calmar比率 = 年化收益 / 最大回撤
        - win_rate: 胜率（正收益天数占比）
        - best_day: 最好的一天
        - worst_day: 最差的一天

    Example:
        >>> metrics = compute_returns_metrics(portfolio_returns)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    # 过滤 NaN
    valid_returns = returns.dropna()

    if len(valid_returns) == 0:
        warnings.warn("收益率序列为空")
        return {
            'total_return': np.nan,
            'annual_return': np.nan,
            'annual_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'calmar_ratio': np.nan,
            'win_rate': np.nan,
            'best_day': np.nan,
            'worst_day': np.nan
        }

    # 总收益率
    total_return = (1 + valid_returns).prod() - 1

    # 年化收益率
    n_periods = len(valid_returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    # 年化波动率
    annual_volatility = valid_returns.std() * np.sqrt(periods_per_year)

    # 夏普比率
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = valid_returns - daily_rf
    sharpe_ratio = (excess_returns.mean() * periods_per_year / annual_volatility
                   if annual_volatility > 0 else 0)

    # 最大回撤
    cumulative_returns = (1 + valid_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar 比率
    calmar_ratio = (annual_return / abs(max_drawdown)
                   if max_drawdown < 0 else np.inf)

    # 胜率
    win_rate = (valid_returns > 0).mean()

    # 最好/最差的一天
    best_day = valid_returns.max()
    worst_day = valid_returns.min()

    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'win_rate': float(win_rate),
        'best_day': float(best_day),
        'worst_day': float(worst_day)
    }


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    计算最大回撤

    Args:
        returns: 收益率序列

    Returns:
        最大回撤（负值）

    Example:
        >>> mdd = compute_max_drawdown(portfolio_returns)
        >>> print(f"最大回撤: {mdd:.2%}")
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def compute_turnover(positions: pd.DataFrame) -> pd.Series:
    """
    计算换手率

    Args:
        positions: 持仓 DataFrame (index: datetime, columns: assets, values: weights)

    Returns:
        每期的换手率序列

    Example:
        >>> turnover = compute_turnover(portfolio_positions)
        >>> print(f"平均换手率: {turnover.mean():.2%}")
    """
    # 计算持仓变化
    position_changes = positions.diff().abs()

    # 换手率 = 调整量之和 / 2
    turnover = position_changes.sum(axis=1) / 2

    return turnover.iloc[1:]  # 去掉第一期（NaN）


def compute_information_ratio(active_returns: pd.Series,
                             periods_per_year: int = 252) -> float:
    """
    计算信息比率

    Args:
        active_returns: 主动收益序列（策略收益 - 基准收益）
        periods_per_year: 每年的周期数

    Returns:
        信息比率 = 年化主动收益 / 年化跟踪误差

    Example:
        >>> active_ret = portfolio_returns - benchmark_returns
        >>> ir = compute_information_ratio(active_ret)
    """
    active_returns = active_returns.dropna()

    if len(active_returns) == 0:
        return np.nan

    annual_active_return = active_returns.mean() * periods_per_year
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)

    if tracking_error > 0:
        return annual_active_return / tracking_error
    else:
        return np.inf if annual_active_return > 0 else 0


def compute_sortino_ratio(returns: pd.Series,
                         risk_free_rate: float = 0.0,
                         periods_per_year: int = 252) -> float:
    """
    计算 Sortino 比率

    与夏普比率类似，但只考虑下行波动率。

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年周期数

    Returns:
        Sortino 比率

    Example:
        >>> sortino = compute_sortino_ratio(portfolio_returns)
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - daily_rf

    # 下行波动率（只考虑负超额收益）
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    else:
        downside_std = 0

    annual_excess_return = excess_returns.mean() * periods_per_year

    if downside_std > 0:
        return annual_excess_return / downside_std
    else:
        return np.inf if annual_excess_return > 0 else 0


def compute_var(returns: pd.Series,
               confidence_level: float = 0.95) -> float:
    """
    计算 VaR (Value at Risk)

    Args:
        returns: 收益率序列
        confidence_level: 置信水平

    Returns:
        VaR值（正值表示损失）

    Example:
        >>> var_95 = compute_var(portfolio_returns, confidence_level=0.95)
        >>> print(f"95% VaR: {var_95:.2%}")
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    var = -returns.quantile(1 - confidence_level)
    return var


def compute_cvar(returns: pd.Series,
                confidence_level: float = 0.95) -> float:
    """
    计算 CVaR (Conditional VaR / Expected Shortfall)

    Args:
        returns: 收益率序列
        confidence_level: 置信水平

    Returns:
        CVaR值（正值表示损失）

    Example:
        >>> cvar_95 = compute_cvar(portfolio_returns, confidence_level=0.95)
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return np.nan

    var_threshold = returns.quantile(1 - confidence_level)
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) > 0:
        cvar = -tail_returns.mean()
    else:
        cvar = -var_threshold

    return cvar


def compute_beta(returns: pd.Series,
                benchmark_returns: pd.Series) -> float:
    """
    计算 Beta

    Args:
        returns: 策略收益率
        benchmark_returns: 基准收益率

    Returns:
        Beta 系数

    Example:
        >>> beta = compute_beta(portfolio_returns, market_returns)
    """
    # 对齐数据
    aligned = pd.DataFrame({
        'returns': returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return np.nan

    covariance = aligned['returns'].cov(aligned['benchmark'])
    benchmark_variance = aligned['benchmark'].var()

    if benchmark_variance > 0:
        return covariance / benchmark_variance
    else:
        return np.nan


def compute_alpha(returns: pd.Series,
                 benchmark_returns: pd.Series,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """
    计算 Alpha (Jensen's Alpha)

    Alpha = 策略收益 - (无风险收益 + Beta * (基准收益 - 无风险收益))

    Args:
        returns: 策略收益率
        benchmark_returns: 基准收益率
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年周期数

    Returns:
        年化 Alpha

    Example:
        >>> alpha = compute_alpha(portfolio_returns, market_returns)
    """
    # 对齐数据
    aligned = pd.DataFrame({
        'returns': returns,
        'benchmark': benchmark_returns
    }).dropna()

    if len(aligned) < 2:
        return np.nan

    # 计算 Beta
    beta = compute_beta(aligned['returns'], aligned['benchmark'])

    # 计算年化收益率
    annual_return = aligned['returns'].mean() * periods_per_year
    annual_benchmark = aligned['benchmark'].mean() * periods_per_year

    # Alpha = 策略年化收益 - (rf + Beta * (基准年化收益 - rf))
    alpha = annual_return - (risk_free_rate + beta * (annual_benchmark - risk_free_rate))

    return alpha


def compute_all_metrics(returns: pd.Series,
                       benchmark_returns: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 252) -> Dict[str, float]:
    """
    计算所有常用指标

    Args:
        returns: 策略收益率
        benchmark_returns: 基准收益率（可选）
        risk_free_rate: 无风险利率
        periods_per_year: 每年周期数

    Returns:
        包含所有指标的字典

    Example:
        >>> all_metrics = compute_all_metrics(portfolio_returns, market_returns)
        >>> for key, value in all_metrics.items():
        ...     print(f"{key}: {value:.4f}")
    """
    # 基本指标
    metrics = compute_returns_metrics(returns, risk_free_rate, periods_per_year)

    # Sortino 比率
    metrics['sortino_ratio'] = compute_sortino_ratio(returns, risk_free_rate, periods_per_year)

    # VaR 和 CVaR
    metrics['var_95'] = compute_var(returns, 0.95)
    metrics['cvar_95'] = compute_cvar(returns, 0.95)

    # 如果提供了基准收益率，计算 Beta、Alpha、IR
    if benchmark_returns is not None:
        metrics['beta'] = compute_beta(returns, benchmark_returns)
        metrics['alpha'] = compute_alpha(returns, benchmark_returns, risk_free_rate, periods_per_year)

        active_returns = returns - benchmark_returns
        metrics['information_ratio'] = compute_information_ratio(active_returns, periods_per_year)
        metrics['tracking_error'] = active_returns.std() * np.sqrt(periods_per_year)

    return metrics
