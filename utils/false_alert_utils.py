# utils/false_alert_utils.py
"""
大鼠麻醉监测系统 - 误报特征提取与分析工具
包含18种误报特征计算及对应优化策略推荐
"""

import numpy as np
import pandas as pd
from .false_alert_strategies import get_strategy


def compute_trend_features(pre_vals):
    """计算趋势特征（斜率、上升/下降比例）"""
    if len(pre_vals) <= 1:
        return np.nan, np.nan, np.nan
    x = np.arange(len(pre_vals))
    slope = np.polyfit(x, pre_vals, 1)[0]
    diffs = np.diff(pre_vals)
    up_ratio = np.sum(diffs > 0) / len(diffs) if len(diffs) > 0 else np.nan
    down_ratio = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else np.nan
    return slope, up_ratio, down_ratio


def compute_volatility_features(pre_vals):
    """计算波动性特征（标准差、范围、变异系数）"""
    if len(pre_vals) == 0:
        return np.nan, np.nan, np.nan
    std = np.std(pre_vals)
    rng = np.max(pre_vals) - np.min(pre_vals)
    mean = np.mean(pre_vals)
    cv = std / mean if mean != 0 else np.nan
    return std, rng, cv


def compute_distribution_features(pre_vals, baseline, limits):
    """计算分布特征（均值、中位数、异常比例）"""
    if len(pre_vals) == 0:
        return [np.nan] * 5
    mean = np.mean(pre_vals)
    median = np.median(pre_vals)
    q25, q75 = np.percentile(pre_vals, [25, 75])
    prop_out = np.sum((pre_vals < limits[0]) | (pre_vals > limits[1])) / len(pre_vals)
    return mean, median, q25, q75, prop_out


def compute_pattern_features(pre_vals):
    """计算模式特征（最长上升/下降序列）"""
    diffs = np.diff(pre_vals)
    max_inc = max_dec = cur_inc = cur_dec = 0
    for d in diffs:
        if d > 0:
            cur_inc += 1
            max_inc = max(max_inc, cur_inc)
            cur_dec = 0
        elif d < 0:
            cur_dec += 1
            max_dec = max(max_dec, cur_dec)
            cur_inc = 0
        else:
            cur_inc = cur_dec = 0
    return max_inc, max_dec


def is_local_extremum(true_seg, idx, window=5):
    """判断是否为局部极值点"""
    if idx < window or idx + window >= len(true_seg):
        return False
    before = true_seg[idx - window:idx]
    after = true_seg[idx + 1:idx + window + 1]
    return (np.all(np.diff(before) > 0) and np.all(np.diff(after) < 0)) or \
           (np.all(np.diff(before) < 0) and np.all(np.diff(after) > 0))


def time_position(idx, total):
    """计算时间位置（归一化+分类）"""
    norm = idx / total
    cat = "start" if norm < 0.33 else ("center" if norm < 0.66 else "end")
    return norm, cat


def collect_false_alerts(true_vals, preds, limits):
    """识别误报（虚报/漏报）并返回详情"""
    false_alerts = []
    for i, (p, t) in enumerate(zip(preds, true_vals)):
        # 虚报：预测超出范围但真实值正常
        if (p < limits[0] or p > limits[1]) and (limits[0] <= t <= limits[1]):
            false_alerts.append({
                "index": i,
                "type": "false_positive",
                "predicted": p,
                "true": t,
                "limits": limits
            })
        # 漏报：预测正常但真实值超出范围
        elif (limits[0] <= p <= limits[1]) and (t < limits[0] or t > limits[1]):
            false_alerts.append({
                "index": i,
                "type": "false_negative",
                "predicted": p,
                "true": t,
                "limits": limits
            })
    return false_alerts


def analyze_false_alert_features(false_alerts, true_vals, pre_window=100, total_steps=None):
    """分析误报特征并推荐优化策略"""
    if not false_alerts:
        return pd.DataFrame()
    
    total_steps = total_steps or len(true_vals)
    records = []
    
    for alert in false_alerts:
        idx = alert["index"]
        # 提取报警前的序列（用于特征分析）
        start = max(0, idx - pre_window)
        pre_vals = true_vals[start:idx]
        
        # 计算各类特征
        slope, up_ratio, down_ratio = compute_trend_features(pre_vals)
        std, rng, cv = compute_volatility_features(pre_vals)
        mean, median, q25, q75, prop_out = compute_distribution_features(
            pre_vals,
            baseline=np.mean(pre_vals) if len(pre_vals) > 0 else np.nan,
            limits=alert["limits"]
        )
        max_inc, max_dec = compute_pattern_features(pre_vals)
        extremum = is_local_extremum(true_vals, idx)
        norm_pos, pos_cat = time_position(idx, total_steps)
        abs_err = abs(alert["predicted"] - alert["true"])
        rel_err = abs_err / alert["true"] if alert["true"] != 0 else np.nan
        
        # 计算严重程度（基于连续报警长度）
        # 这里简化为根据绝对误差分级，实际应用中可基于连续报警长度
        severity_num = 0  # Low
        if abs_err > alert["limits"][1] * 0.2:  # 超过上限20%
            severity_num = 2  # High
        elif abs_err > alert["limits"][1] * 0.1:  # 超过上限10%
            severity_num = 1  # Medium
        
        # 整合特征并关联优化策略
        feature_record = {
            "index": idx,
            "alert_type": alert["type"],
            "predicted": alert["predicted"],
            "true": alert["true"],
            "lower_limit": alert["limits"][0],
            "upper_limit": alert["limits"][1],
            # 趋势特征
            "Slope": slope,
            "Up Ratio": up_ratio,
            "Down Ratio": down_ratio,
            # 波动性特征
            "Std Dev": std,
            "Range": rng,
            "Coef Var": cv,
            # 分布特征
            "Mean": mean,
            "Median": median,
            "Q25": q25,
            "Q75": q75,
            "Prop Out Baseline": prop_out,
            # 模式特征
            "Max Inc Run": max_inc,
            "Max Dec Run": max_dec,
            # 其他特征
            "Abs Error": abs_err,
            "Rel Error": rel_err,
            "Time Pos Norm": norm_pos,
            "Time Pos Cat": pos_cat,
            "Is Extremum": extremum,
            "Severity_num": severity_num,
            # 关联优化策略（按特征名称映射）
            "Slope_strategy": get_strategy("Slope"),
            "Up Ratio_strategy": get_strategy("Up Ratio"),
            "Down Ratio_strategy": get_strategy("Down Ratio"),
            "Std Dev_strategy": get_strategy("Std Dev"),
            "Range_strategy": get_strategy("Range"),
            "Coef Var_strategy": get_strategy("Coef Var"),
            "Mean_strategy": get_strategy("Mean"),
            "Median_strategy": get_strategy("Median"),
            "Q25_strategy": get_strategy("Q25"),
            "Q75_strategy": get_strategy("Q75"),
            "Prop Out Baseline_strategy": get_strategy("Prop Out Baseline"),
            "Max Inc Run_strategy": get_strategy("Max Inc Run"),
            "Max Dec Run_strategy": get_strategy("Max Dec Run"),
            "Abs Error_strategy": get_strategy("Abs Error"),
            "Rel Error_strategy": get_strategy("Rel Error"),
            "Time Pos Norm_strategy": get_strategy("Time Pos Norm"),
            "Is Extremum_strategy": get_strategy("Is Extremum"),
            "Severity_num_strategy": get_strategy("Severity_num")
        }
        
        records.append(feature_record)
    
    return pd.DataFrame(records)


def apply_optimization_strategy(feature_df, config=None):
    """
    根据误报特征自动应用优化策略
    返回更新后的配置参数
    """
    if feature_df.empty:
        return config or {}
    
    # 默认配置
    updated_config = config.copy() if config else {}
    
    # 分析高频率误报特征（Top 3）
    feature_counts = {}
    for col in feature_df.columns:
        if col.endswith("_strategy") and not feature_df[col].isna().all():
            feature_name = col.replace("_strategy", "")
            feature_counts[feature_name] = len(feature_df[~feature_df[col].isna()])
    
    # 获取最频繁导致误报的特征
    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # 针对Top特征应用对应策略
    for feature_name, count in top_features:
        # 获取策略
        strategy = get_strategy(feature_name)
        params = strategy["parameters"]
        
        # 根据策略类型更新配置
        if feature_name in ["Slope", "Up Ratio", "Down Ratio"]:
            # 更新趋势相关配置
            updated_config["trend_features"] = {
                **updated_config.get("trend_features", {}),
                **params
            }
        
        elif feature_name in ["Std Dev", "Range", "Coef Var"]:
            # 更新波动性相关配置
            updated_config["volatility_features"] = {
                **updated_config.get("volatility_features", {}),
                **params
            }
        
        elif feature_name in ["Mean", "Median", "Q25", "Q75", "Prop Out Baseline"]:
            # 更新分布相关配置
            updated_config["distribution_features"] = {
                **updated_config.get("distribution_features", {}),
                **params
            }
        
        elif feature_name in ["Max Inc Run", "Max Dec Run"]:
            # 更新模式相关配置
            updated_config["pattern_features"] = {
                **updated_config.get("pattern_features", {}),
                **params
            }
        
        elif feature_name in ["Abs Error", "Rel Error", "Time Pos Norm", "Is Extremum", "Severity_num"]:
            # 更新其他特征配置
            updated_config["other_features"] = {
                **updated_config.get("other_features", {}),
                **params
            }
    
    return updated_config