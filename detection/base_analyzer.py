import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.data_utils import load_ocr_data  # 复用OCR数据加载工具
from model_selection.model_selector import ModelSelector  # 衔接模型选择模块


class BaseAlarmAnalyzer:
    def __init__(self, config):
        self.config = config
        self.variables = config["common"]["variables"]
        self.window_size = config["common"]["window_size"]
        self.forecast_horizon = config["common"]["forecast_horizon"]
        self.results_dir = config["common"]["results_dir"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载最优模型（从模型选择模块输出）
        self.model = self._load_best_model()
        
        # 初始化标准化器（复用OCR输出的标准化逻辑）
        self.scaler = StandardScaler()

    def _load_best_model(self):
        """加载模型选择模块输出的最优模型"""
        model_path = self.config["common"]["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"最优模型不存在: {model_path}")
        
        # 此处需根据模型选择模块的输出格式调整，示例：
        model_selector = ModelSelector(self.config["model_config"], self.device)
        best_model_info = torch.load(model_path)
        return best_model_info["model"].to(self.device)

    def load_monitoring_data(self, ocr_output_path):
        """加载OCR输出的标准化生理数据（衔接OCR模块）"""
        _, test_loader, self.scaler = load_ocr_data(
            ocr_output_path=ocr_output_path,
            seq_length=self.window_size,
            pred_length=self.forecast_horizon
        )
        return test_loader

    def predict(self, test_loader):
        """使用最优模型进行预测（复用模型预测逻辑）"""
        self.model.eval()
        predictions, true_values = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch).cpu().numpy()
                predictions.append(y_pred)
                true_values.append(y_batch.numpy())
        return np.concatenate(predictions), np.concatenate(true_values)

    def save_results(self, results, mode):
        """保存报警结果到对应目录"""
        mode_dir = os.path.join(self.results_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        # 保存CSV结果
        pd.DataFrame(results).to_csv(
            os.path.join(mode_dir, "alarm_summary.csv"),
            index=False
        )
        return mode_dir

    def analyze(self, test_loader):
        """抽象方法：子类需实现具体报警逻辑"""
        raise NotImplementedError("子类必须实现analyze方法")