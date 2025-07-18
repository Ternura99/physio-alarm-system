
#### **(6) `optimization/README.md`**
```markdown
# 自优化模块

## 功能说明
通过分析误报特征，自动推荐优化策略，提升报警系统的准确性。核心功能：
1. 误报特征提取：支持18种预定义特征（趋势、波动性、分布等）
2. 优化策略推荐：为每种特征内置对应的缓解策略
3. 闭环学习：自动应用策略并更新系统参数

## 核心组件
- `false_alert_analyzer.py`：误报特征提取逻辑
- `strategy_generator.py`：优化策略生成与应用

## 18种误报特征与策略
| 类别         | 特征名称          | 优化策略简述                     |
|--------------|-------------------|----------------------------------|
| 趋势特征     | Slope             | 增加趋势平滑窗口，降低陡峭趋势权重 |
| 趋势特征     | Up Ratio          | 提高上升趋势报警阈值             |
| ...          | ...               | ...                              |
| 其他特征     | Severity_num      | 高严重度误报触发模型再训练       |

## 使用方法
```python
# 示例：分析误报并推荐策略
from optimization import FalseAlertAnalyzer
from utils.false_alert_utils import collect_false_alerts

# 收集误报数据
false_alerts = collect_false_alerts(true_values, predictions, limits)

# 初始化误报分析器
analyzer = FalseAlertAnalyzer()

# 分析误报特征并获取策略
feature_df = analyzer.analyze(false_alerts, true_values)

# 应用优化策略
updated_config = analyzer.apply_strategy(feature_df)

输出结果
误报特征分析：results/optimization_results/false_alert_features.csv
策略推荐：results/optimization_results/strategy_recommendations.csv
更新后的配置：configs/optimization_config.yaml（自动更新）