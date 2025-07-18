import os
import optuna
import torch
import pandas as pd
from ..models.rnn_models import LSTMModel, BiLSTMModel, GRUModel
from ..models.cnn_models import CNNModel, CNNLSTMModel
from ..models.transformer_model import TransformerModel
from ..models.tcn_model import TCNModel
from ..models.xgboost_model import XGBoostModel
from .optimizer import objective
from .pareto_analysis import get_pareto_front, plot_pareto_front


class ModelSelector:
    def __init__(self, config, device):
        self.config = config  # 模型配置（从model_config.yaml加载）
        self.device = device
        self.models = self._init_model_factory()  # 模型工厂：键为模型名，值为创建函数
        self.results_dir = config['results_dir']
        os.makedirs(self.results_dir, exist_ok=True)

    def _init_model_factory(self):
        """模型工厂：根据名称创建对应模型实例"""
        pred_length = self.config['pred_length']
        input_dim = self.config['input_dim']
        
        return {
            'LSTM': lambda params: LSTMModel(
                input_dim=input_dim,
                hidden_units=params['hidden_units'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'BiLSTM': lambda params: BiLSTMModel(
                input_dim=input_dim,
                hidden_units=params['hidden_units'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'GRU': lambda params: GRUModel(
                input_dim=input_dim,
                hidden_units=params['hidden_units'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'CNN': lambda params: CNNModel(
                input_dim=input_dim,
                num_channels=params['num_channels'],
                kernel_size=params['kernel_size'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'CNN-LSTM': lambda params: CNNLSTMModel(
                input_dim=input_dim,
                hidden_units=params['hidden_units'],
                num_layers=params['num_layers'],
                kernel_size=params['kernel_size'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'Transformer': lambda params: TransformerModel(
                input_dim=input_dim,
                hidden_units=params['hidden_units'],
                num_layers=params['num_layers'],
                num_heads=params['num_heads'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'TCN': lambda params: TCNModel(
                input_dim=input_dim,
                num_channels=params['num_channels'],
                kernel_size=params['kernel_size'],
                dropout=params['dropout'],
                pred_length=pred_length
            ),
            'XGBoost': lambda params: XGBoostModel()  # XGBoost参数在fit时处理
        }

    def select_best_model(self, train_loader, test_loader):
        """自动选择最优模型：遍历所有模型→超参数优化→帕累托分析→选择最优"""
        all_pareto_fronts = []
        best_models = []
        
        # 遍历8个模型，分别进行优化
        for model_name in self.models.keys():
            print(f"开始优化模型：{model_name}")
            # 创建Optuna研究（多目标优化）
            study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'])
            study.optimize(
                lambda trial: objective(
                    trial, model_name, self.models[model_name],
                    train_loader, test_loader, self.device, self.config
                ),
                n_trials=self.config['optuna_config']['n_trials'],
                show_progress_bar=True
            )
            
            # 提取帕累托前沿
            pareto_front = get_pareto_front(study.trials)
            all_pareto_fronts.append(pareto_front)
            
            # 保存当前模型的最优结果
            self._save_model_results(model_name, pareto_front)
            
            # 选择当前模型的最优解（帕累托前沿中综合指标最优的）
            best_trial = min(pareto_front, key=lambda t: (t.values[0] + t.values[1] + t.values[2]))
            best_model = self.models[model_name](best_trial.params)
            best_models.append({
                'model_name': model_name,
                'model': best_model,
                'params': best_trial.params,
                'metrics': {
                    'MAE': best_trial.values[0],
                    'RMSE': best_trial.values[1],
                    'R²': 1 - best_trial.values[2]
                }
            })
        
        # 可视化帕累托前沿
        plot_pareto_front(
            all_pareto_fronts, 
            list(self.models.keys()),
            os.path.join(self.results_dir, 'pareto_fronts.png')
        )
        
        # 从所有模型中选择综合最优模型（最小化MAE+RMSE+（1-R²））
        best_overall = min(
            best_models,
            key=lambda x: (x['metrics']['MAE'] + x['metrics']['RMSE'] + (1 - x['metrics']['R²']))
        )
        
        print(f"最优模型：{best_overall['model_name']}，指标：{best_overall['metrics']}")
        return best_overall

    def _save_model_results(self, model_name, pareto_front):
        """保存模型优化结果到文件"""
        results = []
        for trial in pareto_front:
            results.append({
                'model_name': model_name,
                'params': str(trial.params),
                'MAE': trial.values[0],
                'RMSE': trial.values[1],
                'R²': 1 - trial.values[2],
                'trial_number': trial.number
            })
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.results_dir, f'{model_name}_pareto_results.csv'), index=False)