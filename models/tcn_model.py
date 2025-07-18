import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """去除填充，保持因果性（TCN核心组件）"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]  # 去除右侧填充


class TCNBlock(nn.Module):
    """TCN单个残差块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # 因果填充
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = self.chomp(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        # 残差连接
        if self.residual is not None:
            x = self.residual(x)
        return self.relu(out + x)


class TCNModel(nn.Module):
    """时间卷积网络（TCN），适合长时序因果关系捕捉"""
    def __init__(self, input_dim=3, num_channels=[32, 64, 128], kernel_size=3, dropout=0.1, pred_length=60):
        super().__init__()
        layers = []
        in_channels = input_dim
        for ch in num_channels:
            dilation = 2 **len(layers)  # 扩张因子：1,2,4...
            layers.append(TCNBlock(in_channels, ch, kernel_size, dilation, dropout))
            in_channels = ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], input_dim * pred_length)
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch, seq_length, input_dim) → (batch, input_dim, seq_length)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x[:, :, -1]  # 最后一个时间步输出
        x = self.fc(x)
        return x.view(x.size(0), self.pred_length, self.input_dim)