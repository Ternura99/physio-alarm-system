# utils/false_alert_strategies.py
"""内置18种误报特征对应的优化策略"""

FALSE_ALERT_STRATEGIES = {
    # 趋势特征策略
    "Slope": {
        "issue": "预测过度跟随短期陡峭趋势，导致误报",
        "strategy": "1. 增加趋势平滑窗口至150点；2. 调整模型趋势权重（LSTM/Transformer的趋势项系数×0.7）；3. 对斜率绝对值＞0.5的区域启用二次验证",
        "parameters": {"smoothing_window": 150, "trend_weight": 0.7, "slope_threshold": 0.5}
    },
    "Up Ratio": {
        "issue": "对上升趋势敏感度过高，小幅上升即触发报警",
        "strategy": "1. 提高上升趋势报警阈值：连续上升≥3步且累计幅度＞基线5%；2. 降低上升步长权重（原权重×0.8）",
        "parameters": {"min_consecutive_up": 3, "min_up_amplitude": 0.05, "up_step_weight": 0.8}
    },
    "Down Ratio": {
        "issue": "对下降趋势敏感度过高，小幅下降即触发报警",
        "strategy": "1. 提高下降趋势报警阈值：连续下降≥3步且累计幅度＞基线5%；2. 降低下降步长权重（原权重×0.8）",
        "parameters": {"min_consecutive_down": 3, "min_down_amplitude": 0.05, "down_step_weight": 0.8}
    },
    
    # 波动性特征策略
    "Std Dev": {
        "issue": "数据波动大，固定范围无法适应，导致误报",
        "strategy": "1. 动态扩大正常范围：当标准差＞基线20%时，范围扩展至[0.8·bl,1.2·bl]；2. 对高波动区域（std＞0.15·bl）启用滚动窗口（50步）计算范围",
        "parameters": {"volatility_threshold": 0.2, "expanded_range": (0.8, 1.2), "rolling_window": 50}
    },
    "Range": {
        "issue": "瞬时波动导致范围骤增，触发误报",
        "strategy": "1. 采用近3个窗口的平均范围替代单窗口范围；2. 范围突变（较前一窗口增加50%）时，延迟10步再判断",
        "parameters": {"avg_window_count": 3, "range_jump_threshold": 0.5, "delay_steps": 10}
    },
    "Coef Var": {
        "issue": "数据稳定性差，变异系数高导致误报",
        "strategy": "1. 对变异系数＞0.15的数据启用卡尔曼滤波（Q=0.1, R=0.5）；2. 滤波后再判断是否超出范围",
        "parameters": {"cv_threshold": 0.15, "kalman_Q": 0.1, "kalman_R": 0.5}
    },
    
    # 分布特征策略
    "Mean": {
        "issue": "均值偏离基线，基线过时导致误报",
        "strategy": "1. 定期更新基线（每500步），用最新窗口均值替代初始基线；2. 基线更新时排除超出原范围的异常值",
        "parameters": {"baseline_update_interval": 500, "outlier_exclusion": True}
    },
    "Median": {
        "issue": "均值基线受极端值影响，与中位数偏差大导致误报",
        "strategy": "1. 改用中位数作为基线（尤其偏态分布数据）；2. 正常范围调整为[median×0.85, median×1.15]",
        "parameters": {"use_median_as_baseline": True, "median_range": (0.85, 1.15)}
    },
    "Q25": {
        "issue": "25%分位数异常，数据分布偏移导致误报",
        "strategy": "1. 改用分位数范围：[Q25-1.5IQR, Q75+1.5IQR]（IQR=Q75-Q25）；2. 每200步重新计算分位数",
        "parameters": {"use_quantile_range": True, "quantile_update_interval": 200}
    },
    "Q75": {
        "issue": "75%分位数异常，数据分布偏移导致误报",
        "strategy": "同Q25策略：基于IQR的动态范围+定期更新分位数",
        "parameters": {"use_quantile_range": True, "quantile_update_interval": 200}
    },
    "Prop Out Baseline": {
        "issue": "超出基线范围比例高，基线范围不合理",
        "strategy": "1. 当比例＞10%时，自动校准基线范围（扩大至[0.75·bl,1.25·bl]）；2. 用近300步数据重新计算基线",
        "parameters": {"prop_out_threshold": 0.1, "calibrated_range": (0.75, 1.25), "recalibrate_window": 300}
    },
    
    # 模式特征策略
    "Max Inc Run": {
        "issue": "长连续上升序列触发误报，模型过度敏感",
        "strategy": "1. 长上升序列（＞5步）需经2个连续窗口验证；2. 要求上升幅度逐步增大（每步增幅＞前步的80%）",
        "parameters": {"max_inc_run_threshold": 5, "validation_windows": 2, "amplitude_growth_threshold": 0.8}
    },
    "Max Dec Run": {
        "issue": "长连续下降序列触发误报，模型过度敏感",
        "strategy": "同上升序列策略：长下降序列需验证+幅度逐步增大",
        "parameters": {"max_dec_run_threshold": 5, "validation_windows": 2, "amplitude_growth_threshold": 0.8}
    },
    
    # 其他特征策略
    "Abs Error": {
        "issue": "预测绝对误差大，导致误报",
        "strategy": "1. 绝对误差＞基线10%时，启用备用模型（如XGBoost）二次预测；2. 两次预测均超范围才报警",
        "parameters": {"abs_error_threshold": 0.1, "use_secondary_model": True, "secondary_model": "XGBoost"}
    },
    "Rel Error": {
        "issue": "相对误差对小值放大，导致误报",
        "strategy": "1. 小值（如心率＜50）用绝对误差判断（＞10）；2. 大值用相对误差（＞15%）；3. 对接近0的值增加0.1的偏移量",
        "parameters": {"small_value_threshold": 50, "small_abs_error_threshold": 10, "large_rel_error_threshold": 0.15}
    },
    "Time Pos Norm": {
        "issue": "序列边缘（起始/结束）数据稳定性差导致误报",
        "strategy": "1. 起始段（前33%）和结束段（后33%）放宽阈值至±20%；2. 边缘区域报警延迟5步判断",
        "parameters": {"edge_ranges": (0.33, 0.66), "relaxed_range": (0.8, 1.2), "edge_delay_steps": 5}
    },
    "Is Extremum": {
        "issue": "局部极值点触发误报，单点异常非趋势",
        "strategy": "1. 局部极值需连续2步均为极值；2. 要求极值点超出范围的幅度＞基线的8%",
        "parameters": {"consecutive_extremum_threshold": 2, "extremum_amplitude_threshold": 0.08}
    },
    "Severity_num": {
        "issue": "高严重度误报频繁，模型需优化",
        "strategy": "1. 高严重度（2）误报自动记录特征模式，用于模型再训练（样本权重×2）；2. 触发人工审核流程",
        "parameters": {"retrain_weight": 2, "trigger_manual_review": True}
    }
}

def get_strategy(feature_name):
    """根据特征名称获取对应的优化策略"""
    if feature_name not in FALSE_ALERT_STRATEGIES:
        return {"error": f"未找到特征 {feature_name} 的策略"}
    return FALSE_ALERT_STRATEGIES[feature_name]