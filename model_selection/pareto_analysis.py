import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def get_pareto_front(trials):
    """从Optuna试验结果中提取帕累托前沿（非支配解）"""
    # 按MAE排序，筛选非支配解
    trials_sorted = sorted(trials, key=lambda x: x.values[0])
    pareto_front = []
    min_rmse = float('inf')
    min_r2_loss = float('inf')
    
    for trial in trials_sorted:
        mae, rmse, r2_loss = trial.values
        # 若当前解在RMSE和R²损失上均优于之前的解，则加入帕累托前沿
        if rmse < min_rmse and r2_loss < min_r2_loss:
            pareto_front.append(trial)
            min_rmse = rmse
            min_r2_loss = r2_loss
    
    return pareto_front


def plot_pareto_front(pareto_fronts, model_names, save_path):
    """可视化不同模型的帕累托前沿，辅助模型选择"""
    plt.figure(figsize=(10, 6))
    colors = cm.rainbow(np.linspace(0, 1, len(model_names)))
    
    for i, (model_name, fronts) in enumerate(zip(model_names, pareto_fronts)):
        maes = [t.values[0] for t in fronts]
        rmses = [t.values[1] for t in fronts]
        plt.scatter(maes, rmses, color=colors[i], label=model_name, alpha=0.7)
    
    plt.xlabel('MAE')
    plt.ylabel('RMSE')
    plt.title('Pareto Fronts of Model Performance')
    plt.legend()
    plt.savefig(save_path)
    plt.close()