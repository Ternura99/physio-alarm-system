"""
模型选择模块：实现基于Optuna和帕累托前沿分析的自动模型选择。
通过多目标优化（最小化误差、最大化鲁棒性）从候选模型中筛选最优解。
"""
from .optimizer import objective
from .pareto_analysis import get_pareto_front, plot_pareto_front
from .model_selector import ModelSelector

__all__ = [
    "objective",
    "get_pareto_front", "plot_pareto_front",
    "ModelSelector"
]