import numpy as np
import xgboost as xgb


class XGBoostModel:
    """XGBoost回归模型，适配多变量时序预测"""
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            random_state=42
        )

    def fit(self, X, y):
        """
        X: 输入特征 (samples, seq_length*input_dim) → 展平的时序特征
        y: 标签 (samples, pred_length*input_dim) → 展平的预测目标
        """
        self.model.fit(X, y)

    def __call__(self, X):
        """预测接口，与PyTorch模型保持一致"""
        # X: (samples, seq_length, input_dim) → 展平为(samples, seq_length*input_dim)
        X_flat = X.reshape(X.shape[0], -1) if len(X.shape) == 3 else X
        y_pred = self.model.predict(X_flat)
        # 重塑为(samples, pred_length, input_dim)
        return y_pred.reshape(-1, self.pred_length, 3)

    def set_pred_length(self, pred_length):
        self.pred_length = pred_length  # 动态设置预测步长（与其他模型对齐）