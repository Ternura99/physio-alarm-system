# 模型选择模块

## 功能说明
基于Optuna和帕累托前沿分析，从8种候选模型中自动选择最优模型，流程包括：
1. 超参数优化：使用Optuna搜索每种模型的最优超参数
2. 多目标评估：通过MAE、RMSE、R²等指标评估模型性能
3. 帕累托前沿分析：筛选在多个指标上表现最优的模型

## 核心组件
- `optimizer.py`：Optuna超参数优化逻辑
- `pareto_analysis.py`：帕累托前沿计算与可视化
- `model_selector.py`：自动模型选择主逻辑

## 使用方法
```python
# 示例代码
from model_selection import ModelSelector
from utils.data_utils import load_ocr_data
from utils.config_utils import load_config

# 加载数据（OCR输出的标准化数据）
train_loader, test_loader, _ = load_ocr_data("data/processed_data")

# 加载配置
config = load_config("configs/model_config.yaml")

# 初始化模型选择器
selector = ModelSelector(config)

# 自动选择最优模型
best_model = selector.select_best_model(train_loader, test_loader)

# 保存最优模型
import torch
torch.save(best_model, "results/model_selection/best_model.pth")

输出结果
模型性能对比：results/model_selection/model_comparison.csv
帕累托前沿图：results/model_selection/pareto_front.png
最优模型权重：results/model_selection/best_model.pth

两种模式对比
特性	固定基线模式	动态基线模式
基线更新频率	静态（一次初始化）	动态（每 N 步更新）
适用场景	稳定状态监测	趋势变化监测
误报分析	定时分析（每 200 步）	实时分析
正常范围定义	[0.85× 基线，1.15× 基线]	基于滑动窗口的动态范围
输出结果
报警记录：results/alarm_results/<mode>/alarm_summary.csv
可视化结果：results/alarm_results/<mode>/*.png