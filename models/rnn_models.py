import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """基础LSTM模型，处理时序生理数据"""
    def __init__(self, input_dim=3, hidden_units=128, num_layers=1, dropout=0.1, pred_length=60):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_units, input_dim * pred_length)  # 输出维度：特征数×预测步长
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        out, _ = self.lstm(x)  # 取最后一层LSTM的输出
        out = self.fc(out[:, -1, :])  # 用最后一个时间步的特征做预测
        return out.view(out.size(0), self.pred_length, self.input_dim)  # 重塑为(batch, pred_length, input_dim)


class BiLSTMModel(nn.Module):
    """双向LSTM，增强时序特征提取能力"""
    def __init__(self, input_dim=3, hidden_units=128, num_layers=1, dropout=0.1, pred_length=60):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_units * 2, input_dim * pred_length)  # 双向需×2
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.fc(out[:, -1, :])  # 最后一个时间步的双向特征融合
        return out.view(out.size(0), self.pred_length, self.input_dim)


class GRUModel(nn.Module):
    """GRU模型，轻量版LSTM"""
    def __init__(self, input_dim=3, hidden_units=128, num_layers=1, dropout=0.1, pred_length=60):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_units, input_dim * pred_length)
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), self.pred_length, self.input_dim)