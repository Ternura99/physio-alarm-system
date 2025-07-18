import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """纯CNN模型，捕捉局部时序特征"""
    def __init__(self, input_dim=3, num_channels=[32, 64, 128], kernel_size=3, dropout=0.1, pred_length=60):
        super().__init__()
        self.conv_layers = nn.Sequential()
        in_channels = input_dim  # 输入通道数=特征数（3：血压、舒张压、心率）
        
        for i, ch in enumerate(num_channels):
            self.conv_layers.add_module(
                f'conv_{i}',
                nn.Sequential(
                    nn.Conv1d(in_channels, ch, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(ch),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_channels = ch
        
        self.fc = nn.Linear(num_channels[-1], input_dim * pred_length)
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim) → 转换为(batch, input_dim, seq_length)适配Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)  # 卷积层提取特征
        x = x[:, :, -1]  # 取最后一个时间步的特征
        x = self.fc(x)
        return x.view(x.size(0), self.pred_length, self.input_dim)


class CNNLSTMModel(nn.Module):
    """CNN+LSTM混合模型，先提取局部特征再捕捉时序依赖"""
    def __init__(self, input_dim=3, hidden_units=128, num_layers=1, kernel_size=3, dropout=0.1, pred_length=60):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_units, kernel_size, padding=kernel_size//2)
        self.lstm = nn.LSTM(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_units, input_dim * pred_length)
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch, seq_length, input_dim) → (batch, input_dim, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)  # CNN提取局部特征
        x = x.permute(0, 2, 1)  # 转换回(batch, seq_length, hidden_units)
        out, _ = self.lstm(x)  # LSTM捕捉时序依赖
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), self.pred_length, self.input_dim)