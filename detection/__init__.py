"""
报警模式模块：提供两种临床导向的报警模式，支持大鼠麻醉状态监测。
- 固定基线模式：适用于稳定状态，基于静态阈值判断异常。
- 动态基线模式：适用于趋势检测，随时间更新基线范围。
"""
from .base_analyzer import BaseAlarmAnalyzer
from .fixed_baseline import FixedBaselineAnalyzer
from .dynamic_baseline import DynamicBaselineAnalyzer
from .alarm_selector import AlarmModeSelector

__all__ = [
    "BaseAlarmAnalyzer",
    "FixedBaselineAnalyzer", "DynamicBaselineAnalyzer",
    "AlarmModeSelector"
]