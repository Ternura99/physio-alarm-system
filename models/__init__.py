"""
深度学习模型模块：包含8种候选模型，覆盖循环神经网络、卷积神经网络和Transformer。
模型统一接口设计，支持快速替换和性能对比。
"""
from .rnn_models import LSTMModel, BiLSTMModel, GRUModel
from .cnn_models import CNNModel, CNNLSTMModel
from .transformer_model import TransformerModel
from .tcn_model import TCNModel
from .xgboost_model import XGBoostModel

__all__ = [
    "LSTMModel", "BiLSTMModel", "GRUModel",
    "CNNModel", "CNNLSTMModel",
    "TransformerModel", "TCNModel", "XGBoostModel"
]