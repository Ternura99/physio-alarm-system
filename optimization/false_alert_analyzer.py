# optimization/false_alert_analyzer.py
"""
大鼠麻醉监测系统 - 误报特征分析器
计算18种误报特征并生成优化策略
"""

import numpy as np
import pandas as pd
from .false_alert_strategies import get_strategy


class FalseAlertAnalyzer:
    """误报特征分析器，计算18种误报特征并生成优化策略"""
    
    def __init__(self, config=None):
        """初始化分析器"""
        self.config = config or {}
        self.feature_functions = {
            # 趋势特征
            "Slope": self._calculate_slope,
            "Up Ratio": self._calculate_up_ratio,
            "Down Ratio": self._calculate_down_ratio,
            # 波动性特征
            "Std Dev": self._calculate_std_dev,
            "Range": self._calculate_range,
            "Coef Var": self._calculate_coef_var,
            # 分布特征
            "Mean": self._calculate_mean,
            "Median": self._calculate_median,
            "Q25": self._calculate_q25,
            "Q75": self._calculate_q75,
            "Prop Out Baseline": self._calculate_prop_out_baseline,
            # 模式特征
            "Max Inc Run": self._calculate_max_inc_run,
            "Max Dec Run": self._calculate_max_dec_run,
            # 其他特征
            "Abs Error": self._calculate_abs_error,
            "Rel Error": self._calculate_rel_error,
            "Time Pos Norm": self._calculate_time_pos_norm,
            "Is Extremum": self._calculate_is_extremum,
            "Severity_num": self._calculate_severity
        }
    
    def analyze(self, false_alerts, true_values, pre_window=100):
        """分析误报特征并生成优化策略"""
        if not false_alerts:
            return pd.DataFrame()
        
        records = []
        total_steps = len(true_values)
        
        for alert in false_alerts:
            idx = alert["index"]
            alert_type = alert["type"]
            
            # 提取报警前的窗口数据
            start = max(0, idx - pre_window)
            pre_vals = true_values[start:idx]
            
            # 计算所有18种特征
            features = {}
            for feature_name, func in self.feature_functions.items():
                features[feature_name] = func(pre_vals, idx, total_steps, alert)
            
            # 添加报警信息和策略
            feature_record = {
                "alert_index": idx,
                "alert_type": alert_type,
                **features,
                # 为每个特征添加对应策略
                **{f"{feature_name}_strategy": get_strategy(feature_name) 
                   for feature_name in self.feature_functions}
            }
            
            records.append(feature_record)
        
        return pd.DataFrame(records)
    
    def apply_strategy(self, feature_df, config=None):
        """根据误报特征应用优化策略并更新配置"""
        if feature_df.empty:
            return config or {}
        
        updated_config = config.copy() if config else self.config.copy()
        
        # 分析最频繁出现的误报特征
        feature_counts = {}
        for col in feature_df.columns:
            if col.endswith("_strategy") and not col.startswith("Time Pos"):
                feature_name = col.replace("_strategy", "")
                # 统计非NaN值的数量
                count = len(feature_df[~feature_df[col].isna()])
                if count > 0:
                    feature_counts[feature_name] = count
        
        # 按出现频率排序
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 为Top3特征应用对应策略
        for feature_name, count in sorted_features[:3]:
            strategy = get_strategy(feature_name)
            strategy_params = strategy.get("parameters", {})
            
            # 根据特征类型更新对应配置部分
            if feature_name in ["Slope", "Up Ratio", "Down Ratio"]:
                updated_config["trend_features"] = {
                    **updated_config.get("trend_features", {}),
                    **strategy_params
                }
            elif feature_name in ["Std Dev", "Range", "Coef Var"]:
                updated_config["volatility_features"] = {
                    **updated_config.get("volatility_features", {}),
                    **strategy_params
                }
            elif feature_name in ["Mean", "Median", "Q25", "Q75", "Prop Out Baseline"]:
                updated_config["distribution_features"] = {
                    **updated_config.get("distribution_features", {}),
                    **strategy_params
                }
            elif feature_name in ["Max Inc Run", "Max Dec Run"]:
                updated_config["pattern_features"] = {
                    **updated_config.get("pattern_features", {}),
                    **strategy_params
                }
            elif feature_name in ["Abs Error", "Rel Error", "Time Pos Norm", "Is Extremum", "Severity_num"]:
                updated_config["other_features"] = {
                    **updated_config.get("other_features", {}),
                    **strategy_params
                }
        
        return updated_config
    
    # 以下是18种误报特征的计算函数
    def _calculate_slope(self, pre_vals, idx, total_steps, alert):
        """计算线性回归斜率"""
        if len(pre_vals) < 2:
            return np.nan
        x = np.arange(len(pre_vals))
        slope, _ = np.polyfit(x, pre_vals, 1)
        return slope
    
    def _calculate_up_ratio(self, pre_vals, idx, total_steps, alert):
        """计算上升步数比例"""
        if len(pre_vals) < 2:
            return np.nan
        diffs = np.diff(pre_vals)
        return np.sum(diffs > 0) / len(diffs)
    
    def _calculate_down_ratio(self, pre_vals, idx, total_steps, alert):
        """计算下降步数比例"""
        if len(pre_vals) < 2:
            return np.nan
        diffs = np.diff(pre_vals)
        return np.sum(diffs < 0) / len(diffs)
    
    def _calculate_std_dev(self, pre_vals, idx, total_steps, alert):
        """计算标准差"""
        if len(pre_vals) == 0:
            return np.nan
        return np.std(pre_vals)
    
    def _calculate_range(self, pre_vals, idx, total_steps, alert):
        """计算范围（最大值-最小值）"""
        if len(pre_vals) == 0:
            return np.nan
        return np.max(pre_vals) - np.min(pre_vals)
    
    def _calculate_coef_var(self, pre_vals, idx, total_steps, alert):
        """计算变异系数（标准差/均值）"""
        if len(pre_vals) == 0 or np.mean(pre_vals) == 0:
            return np.nan
        return np.std(pre_vals) / np.mean(pre_vals)
    
    def _calculate_mean(self, pre_vals, idx, total_steps, alert):
        """计算均值"""
        if len(pre_vals) == 0:
            return np.nan
        return np.mean(pre_vals)
    
    def _calculate_median(self, pre_vals, idx, total_steps, alert):
        """计算中位数"""
        if len(pre_vals) == 0:
            return np.nan
        return np.median(pre_vals)
    
    def _calculate_q25(self, pre_vals, idx, total_steps, alert):
        """计算25%分位数"""
        if len(pre_vals) == 0:
            return np.nan
        return np.percentile(pre_vals, 25)
    
    def _calculate_q75(self, pre_vals, idx, total_steps, alert):
        """计算75%分位数"""
        if len(pre_vals) == 0:
            return np.nan
        return np.percentile(pre_vals, 75)
    
    def _calculate_prop_out_baseline(self, pre_vals, idx, total_steps, alert):
        """计算超出基线范围的比例"""
        if len(pre_vals) == 0:
            return np.nan
        
        baseline = np.mean(pre_vals)
        lower_limit = 0.7 * baseline
        upper_limit = 1.3 * baseline
        
        return np.sum((pre_vals < lower_limit) | (pre_vals > upper_limit)) / len(pre_vals)
    
    def _calculate_max_inc_run(self, pre_vals, idx, total_steps, alert):
        """计算最长连续上升序列"""
        if len(pre_vals) < 2:
            return 0
        
        max_run = 0
        current_run = 0
        
        for i in range(1, len(pre_vals)):
            if pre_vals[i] > pre_vals[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def _calculate_max_dec_run(self, pre_vals, idx, total_steps, alert):
        """计算最长连续下降序列"""
        if len(pre_vals) < 2:
            return 0
        
        max_run = 0
        current_run = 0
        
        for i in range(1, len(pre_vals)):
            if pre_vals[i] < pre_vals[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        return max_run
    
    def _calculate_abs_error(self, pre_vals, idx, total_steps, alert):
        """计算绝对误差"""
        return abs(alert["predicted"] - alert["true"])
    
    def _calculate_rel_error(self, pre_vals, idx, total_steps, alert):
        """计算相对误差"""
        if alert["true"] == 0:
            return np.nan
        return abs(alert["predicted"] - alert["true"]) / alert["true"]
    
    def _calculate_time_pos_norm(self, pre_vals, idx, total_steps, alert):
        """计算归一化时间位置"""
        return idx / total_steps
    
    def _calculate_is_extremum(self, pre_vals, idx, total_steps, alert):
        """判断是否为局部极值点"""
        if idx < 5 or idx >= total_steps - 5:
            return False
        
        window_before = true_values[idx-5:idx]
        window_after = true_values[idx+1:idx+6]
        
        current_value = true_values[idx]
        
        # 局部最大值
        if (current_value > max(window_before) and 
            current_value > max(window_after)):
            return True
        
        # 局部最小值
        if (current_value < min(window_before) and 
            current_value < min(window_after)):
            return True
        
        return False
    
    def _calculate_severity(self, pre_vals, idx, total_steps, alert):
        """计算严重程度（基于连续报警长度）"""
        abs_error = abs(alert["predicted"] - alert["true"])
        upper_limit = alert["limits"][1]
        
        if abs_error > upper_limit * 0.2:
            return 2  # High
        elif abs_error > upper_limit * 0.1:
            return 1  # Medium
        else:
            return 0  # Low