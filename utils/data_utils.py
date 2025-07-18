import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_ocr_data(ocr_output_path, seq_length=10, pred_length=60):
    """
    从OCR输出目录加载标准化生理数据，转换为模型输入格式
    :param ocr_output_path: OCR模块输出的标准化数据路径（如CSV文件）
    :param seq_length: 输入时序长度
    :param pred_length: 预测步长
    :return: 训练/测试数据加载器、标准化器
    """
    # 读取OCR输出的标准化数据（假设为CSV，包含'systolic_blood_pressure'等列）
    all_data = []
    for file in os.listdir(ocr_output_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(ocr_output_path, file))
            # 提取关键生理指标（与OCR模块输出对齐）
            df = df[['systolic_blood_pressure', 'diastolic_blood_pressure', 'pulse_rate']]
            all_data.append(df.values)
    
    # 合并所有数据（按时间顺序）
    data = np.concatenate(all_data, axis=0)
    # 标准化（与OCR的标准化保持一致）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 构建时序样本：输入序列→预测序列
    X, y = [], []
    for i in range(len(data_scaled) - seq_length - pred_length):
        X.append(data_scaled[i:i+seq_length])  # 输入：前seq_length个时间步
        y.append(data_scaled[i+seq_length:i+seq_length+pred_length])  # 输出：后pred_length个时间步
    
    X, y = np.array(X), np.array(y)
    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # 时序数据不打乱
    
    # 转换为PyTorch DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, scaler


def inverse_transform(scaler, data):
    """将标准化数据反转为原始尺度（用于结果解释）"""
    return scaler.inverse_transform(data.reshape(-1, 3)).reshape(data.shape)