# 大鼠麻醉监测智能报警系统

基于深度学习和OCR技术的大鼠麻醉生理数据监测系统，支持自动模型选择、多模式报警及误报自优化，为实验动物监测提供智能化解决方案。

## 系统架构
![系统架构图](paper_supplements/architecture_diagram.png)

系统包含4个核心模块：
1. **数据处理模块**：OCR提取生理数据并标准化
2. **模型选择模块**：从8种候选模型中自动筛选最优解
3. **报警模式模块**：支持固定基线（稳定状态）和动态基线（趋势检测）
4. **自优化模块**：分析误报特征并推荐优化策略

## 快速开始

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

运行流程
启动 OCR 服务：提取生理数据
bash
python scripts/run_ocr_service.py


模型自动选择：基于 OCR 输出数据训练并选择最优模型
bash
python scripts/run_model_pipeline.py

启动报警系统：选择报警模式（固定 / 动态基线）
bash
# 固定基线模式
python scripts/run_alarm_system.py --mode fixed
# 动态基线模式
python scripts/run_alarm_system.py --mode dynamic

目录说明
data/：输入数据与处理结果
data_processing/：OCR 数据提取与预处理
models/：深度学习模型实现
model_selection/：自动模型选择与超参数优化
detection/：两种报警模式实现
optimization/：误报分析与自优化策略
scripts/：系统启动脚本
论文相关
补充材料：
