"""
自优化模块：通过闭环学习管道分析误报特征并推荐缓解策略。
支持18种预定义误报特征的提取，内置对应的优化策略，提升系统鲁棒性。
"""
from .false_alert_analyzer import FalseAlertAnalyzer
from .strategy_generator import get_strategy, apply_strategy

__all__ = [
    "FalseAlertAnalyzer",
    "get_strategy", "apply_strategy"
]