
#### **(3) `models/README.md`**
```markdown
# 深度学习模型模块

## 功能说明
实现8种候选模型，用于大鼠生理数据的预测与分析，包括循环神经网络、卷积神经网络和Transformer等主流架构，支持时序数据的趋势预测。

## 模型列表
| 模型名称       | 类型         | 适用场景                 |
|----------------|--------------|--------------------------|
| LSTM           | 循环神经网络 | 短期时序依赖预测         |
| BiLSTM         | 双向循环网络 | 需考虑前后文的时序预测   |
| GRU            | 循环神经网络 | 轻量版LSTM，效率更高     |
| CNN            | 卷积神经网络 | 局部特征提取             |
| CNN-LSTM       | 混合模型     | 局部特征+时序依赖联合建模 |
| TCN            | 时间卷积网络 | 长时序因果关系建模       |
| Transformer    | 注意力机制   | 长距离时序依赖建模       |
| XGBoost        | 传统机器学习 | 作为深度学习模型的对比基准 |

## 使用方法
```python
# 示例：初始化LSTM模型
from models import LSTMModel

# 初始化模型
model = LSTMModel(
    hidden_units=128,
    num_layers=2,
    dropout=0.2
)

# 模型输入输出格式
# 输入：(batch_size, seq_length, input_dim)  # input_dim=3（血压、心率等）
# 输出：(batch_size, pred_length, 3)  # 预测未来pred_length步的生理数据

依赖说明
PyTorch：深度学习框架
XGBoost：传统机器学习模型支持