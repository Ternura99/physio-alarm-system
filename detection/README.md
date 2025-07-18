
# 报警模式模块

## 功能说明
基于模型预测结果实现临床导向的报警功能，支持两种模式：
1. **固定基线模式**：适用于稳定状态，基于静态基线和正常范围判断异常
2. **动态基线模式**：适用于动态变化场景，随时间更新基线范围，检测趋势漂移

## 核心组件
- `base_analyzer.py`：报警模式基类，定义通用接口
- `fixed_baseline.py`：固定基线模式，含定时误报分析
- `dynamic_baseline.py`：动态基线模式，支持基线动态更新
- `alarm_selector.py`：人工选择报警模式的入口

## 使用方法
```python
# 示例：固定基线模式
from detection import AlarmModeSelector
from utils.config_utils import load_config

# 加载配置
config = load_config("configs/alarm_config.yaml")

# 初始化报警选择器
selector = AlarmModeSelector(config)

# 运行固定基线模式
results = selector.run(mode="fixed", ocr_data_path="data/processed_data")