# optimization/false_alert_strategies.py
"""
大鼠麻醉监测系统 - 误报优化策略
定义18种误报特征对应的优化策略
"""

# 趋势特征优化策略
TREND_STRATEGIES = {
    "Slope": {
        "description": "针对斜率特征的优化策略，降低对短期陡峭趋势的敏感度",
        "parameters": {
            "smoothing_window": 150,  # 增加趋势平滑窗口大小
            "trend_weight": 0.7,  # 降低趋势项权重
            "min_consecutive_points": 5  # 增加连续点要求
        },
        "implementation": "增加趋势平滑窗口，降低模型对短期陡峭趋势的权重"
    },
    "Up Ratio": {
        "description": "针对上升比例特征的优化策略，减少对小幅上升的误报",
        "parameters": {
            "min_consecutive_up": 3,  # 最小连续上升步数
            "min_up_amplitude": 0.05,  # 最小上升幅度（相对于基线）
            "up_threshold_multiplier": 1.2  # 提高上升阈值倍数
        },
        "implementation": "提高上升趋势的报警阈值，要求连续上升步数和幅度"
    },
    "Down Ratio": {
        "description": "针对下降比例特征的优化策略，减少对小幅下降的误报",
        "parameters": {
            "min_consecutive_down": 3,  # 最小连续下降步数
            "min_down_amplitude": 0.05,  # 最小下降幅度（相对于基线）
            "down_threshold_multiplier": 0.8  # 降低下降阈值倍数
        },
        "implementation": "提高下降趋势的报警阈值，要求连续下降步数和幅度"
    }
}

# 波动性特征优化策略
VOLATILITY_STRATEGIES = {
    "Std Dev": {
        "description": "针对标准差特征的优化策略，适应数据波动性变化",
        "parameters": {
            "volatility_threshold": 0.2,  # 波动性阈值（相对于基线）
            "expanded_range_factor": 1.2,  # 高波动时扩大范围因子
            "smoothing_factor": 0.3  # 平滑因子
        },
        "implementation": "动态扩大正常范围，当标准差超过阈值时放宽限制"
    },
    "Range": {
        "description": "针对范围特征的优化策略，减少对瞬时波动的误报",
        "parameters": {
            "rolling_window": 50,  # 滚动窗口大小
            "range_threshold": 1.5,  # 范围阈值倍数
            "outlier_rejection_factor": 2.0  # 异常值剔除因子
        },
        "implementation": "采用滚动窗口计算范围，用平均范围替代单窗口范围"
    },
    "Coef Var": {
        "description": "针对变异系数特征的优化策略，处理高变异数据",
        "parameters": {
            "cv_threshold": 0.15,  # 变异系数阈值
            "kalman_q": 0.1,  # 卡尔曼滤波过程噪声
            "kalman_r": 0.5  # 卡尔曼滤波测量噪声
        },
        "implementation": "对高变异系数数据启用卡尔曼滤波，平滑短期波动"
    }
}

# 分布特征优化策略
DISTRIBUTION_STRATEGIES = {
    "Mean": {
        "description": "针对均值特征的优化策略，处理数据分布漂移",
        "parameters": {
            "baseline_update_interval": 500,  # 基线更新间隔
            "outlier_exclusion": True,  # 是否排除异常值
            "update_factor": 0.1  # 基线更新因子
        },
        "implementation": "定期更新基线，用最新窗口均值替代初始基线"
    },
    "Median": {
        "description": "针对中位数特征的优化策略，处理偏态分布",
        "parameters": {
            "use_median_as_baseline": True,  # 使用中位数作为基线
            "median_range_factor": 0.85,  # 中位数范围因子
            "percentile_window": 100  # 百分位数计算窗口
        },
        "implementation": "对偏态分布数据改用中位数作为基线，减少极端值影响"
    },
    "Q25": {
        "description": "针对25%分位数特征的优化策略，动态调整正常范围",
        "parameters": {
            "use_iqr_range": True,  # 使用IQR范围
            "iqr_factor": 1.5,  # IQR倍数
            "quantile_update_interval": 200  # 分位数更新间隔
        },
        "implementation": "基于分位数动态调整正常范围，定期更新分位数"
    },
    "Q75": {
        "description": "针对75%分位数特征的优化策略，动态调整正常范围",
        "parameters": {
            "use_iqr_range": True,  # 使用IQR范围
            "iqr_factor": 1.5,  # IQR倍数
            "quantile_update_interval": 200  # 分位数更新间隔
        },
        "implementation": "基于分位数动态调整正常范围，定期更新分位数"
    },
    "Prop Out Baseline": {
        "description": "针对超出基线比例特征的优化策略，校准不合理基线",
        "parameters": {
            "prop_threshold": 0.1,  # 超出比例阈值
            "calibration_factor": 1.25,  # 校准因子
            "recalibration_window": 300  # 重新校准窗口
        },
        "implementation": "当超出基线比例过高时，自动校准基线范围"
    }
}

# 模式特征优化策略
PATTERN_STRATEGIES = {
    "Max Inc Run": {
        "description": "针对最长连续上升特征的优化策略，减少对孤立上升序列的误报",
        "parameters": {
            "min_consecutive_inc": 5,  # 最小连续上升步数
            "validation_windows": 2,  # 验证窗口数
            "amplitude_growth_threshold": 0.8  # 幅度增长阈值
        },
        "implementation": "对长上升序列增加趋势验证，要求在多个窗口中一致"
    },
    "Max Dec Run": {
        "description": "针对最长连续下降特征的优化策略，减少对孤立下降序列的误报",
        "parameters": {
            "min_consecutive_dec": 5,  # 最小连续下降步数
            "validation_windows": 2,  # 验证窗口数
            "amplitude_growth_threshold": 0.8  # 幅度增长阈值
        },
        "implementation": "对长下降序列增加趋势验证，要求在多个窗口中一致"
    }
}

# 其他特征优化策略
OTHER_STRATEGIES = {
    "Abs Error": {
        "description": "针对绝对误差特征的优化策略，减少大误差误报",
        "parameters": {
            "error_threshold": 0.1,  # 误差阈值（相对于基线）
            "secondary_model": "XGBoost",  # 二次验证模型
            "validation_steps": 2  # 验证步数
        },
        "implementation": "对大绝对误差启用二次预测验证，两次均超范围才报警"
    },
    "Rel Error": {
        "description": "针对相对误差特征的优化策略，平衡大小值误报",
        "parameters": {
            "small_value_threshold": 50,  # 小值阈值
            "small_abs_error_threshold": 10,  # 小值绝对误差阈值
            "large_rel_error_threshold": 0.15,  # 大值相对误差阈值
            "epsilon": 0.1  # 防止除零的小常量
        },
        "implementation": "对小值用绝对误差判断，对大值用相对误差判断"
    },
    "Time Pos Norm": {
        "description": "针对时间位置特征的优化策略，减少序列边缘误报",
        "parameters": {
            "edge_region_threshold": 0.33,  # 边缘区域阈值
            "relaxed_range_factor": 1.2,  # 放宽范围因子
            "edge_delay_steps": 5  # 边缘延迟判断步数
        },
        "implementation": "对序列起始和结束段放宽报警阈值，增加延迟判断"
    },
    "Is Extremum": {
        "description": "针对局部极值特征的优化策略，减少孤立极值误报",
        "parameters": {
            "extremum_window": 5,  # 极值判断窗口
            "consecutive_extrema": 2,  # 连续极值要求
            "amplitude_threshold": 0.08  # 幅度阈值（相对于基线）
        },
        "implementation": "对局部极值点增加延迟判断，需连续多个时间步均为极值"
    },
    "Severity_num": {
        "description": "针对严重程度特征的优化策略，减少高严重度误报",
        "parameters": {
            "high_severity_threshold": 2,  # 高严重度阈值
            "retrain_weight": 2.0,  # 再训练权重
            "manual_review_threshold": 0.05  # 人工审核阈值
        },
        "implementation": "对高严重度误报自动记录特征模式，用于模型再训练"
    }
}

# 整合所有策略
ALL_STRATEGIES = {
    **TREND_STRATEGIES,
    **VOLATILITY_STRATEGIES,
    **DISTRIBUTION_STRATEGIES,
    **PATTERN_STRATEGIES,
    **OTHER_STRATEGIES
}


def get_strategy(feature_name):
    """根据特征名称获取对应的优化策略"""
    if feature_name in ALL_STRATEGIES:
        return ALL_STRATEGIES[feature_name]
    else:
        return {
            "description": f"未找到特征 {feature_name} 的优化策略",
            "parameters": {},
            "implementation": "无"
        }


def apply_strategy(feature_name, current_config):
    """根据特征名称应用对应的优化策略，更新配置"""
    strategy = get_strategy(feature_name)
    if not strategy or not strategy.get("parameters"):
        return current_config
    
    # 根据特征类型确定配置更新路径
    if feature_name in TREND_STRATEGIES:
        config_key = "trend_features"
    elif feature_name in VOLATILITY_STRATEGIES:
        config_key = "volatility_features"
    elif feature_name in DISTRIBUTION_STRATEGIES:
        config_key = "distribution_features"
    elif feature_name in PATTERN_STRATEGIES:
        config_key = "pattern_features"
    elif feature_name in OTHER_STRATEGIES:
        config_key = "other_features"
    else:
        return current_config
    
    # 更新配置
    current_config[config_key] = {
        **current_config.get(config_key, {}),
        **strategy["parameters"]
    }
    
    return current_config