import optuna
import torch
from ..utils.train_utils import train_model, evaluate_model


def objective(trial, model_name, model_factory, train_loader, test_loader, device, config):
    """Optuna目标函数，为不同模型搜索最优超参数"""
    # 超参数搜索空间（与模型类型关联）
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'hidden_units': trial.suggest_categorical('hidden_units', config['hidden_units']),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    
    # 针对特定模型扩展参数
    if model_name in ['Transformer']:
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
    if model_name in ['CNN', 'CNN-LSTM', 'TCN']:
        params['kernel_size'] = trial.suggest_int('kernel_size', 3, 7)
    if model_name in ['TCN', 'CNN']:
        params['num_channels'] = [params['hidden_units'] * (2**i) for i in range(params['num_layers'])]
    
    # 创建模型实例
    model = model_factory(model_name, params, config)
    
    # 训练与评估
    trained_model, _ = train_model(model, train_loader, device, lr=params['lr'], epochs=config['epochs'])
    metrics = evaluate_model(trained_model, test_loader, device)
    
    # 多目标优化：最小化MAE、RMSE，最大化R²（此处转为最小化1-R²）
    return metrics['MAE'], metrics['RMSE'], 1 - metrics['R²']