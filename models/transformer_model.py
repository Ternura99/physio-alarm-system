import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    """基于Transformer的时序预测模型，适合捕捉长距离依赖"""
    def __init__(self, input_dim=3, hidden_units=128, num_layers=1, num_heads=2, dropout=0.1, pred_length=60):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_units)  # 将输入特征映射到高维
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units * 4,
                dropout=dropout,
                batch_first=True  # 批次优先
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_units, input_dim * pred_length)
        self.pred_length = pred_length
        self.input_dim = input_dim

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # 映射到hidden_units维度
        x = self.transformer_encoder(x)  # 自注意力提取特征
        x = self.fc(x[:, -1, :])  # 最后一个时间步输出
        return x.view(x.size(0), self.pred_length, self.input_dim)