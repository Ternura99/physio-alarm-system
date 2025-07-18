
# 数据处理模块（OCR）

## 功能说明
从大鼠麻醉实验图像中提取生理数据（血压、心率等），流程包括：
1. 目标检测：使用YOLO模型定位图像中的数值区域
2. 文本识别：通过PaddleOCR提取数值文本
3. 数据标准化：将识别结果转换为结构化生理数据，用于后续模型训练

## 核心组件
- `ocr_processor.py`：OCR核心逻辑，整合检测与识别
- `image_utils.py`：图像处理工具（检测框聚类、结果可视化）

## 使用方法
```python
# 示例代码
from data_processing import OCRProcessor
from utils.config_utils import load_config

# 加载配置
config = load_config("configs/ocr_config.yaml")

# 初始化OCR处理器
ocr_processor = OCRProcessor(config)

# 处理单张图像
result = ocr_processor.process_image(image_path="data/raw_images/sample.png")

# 处理图像文件夹
ocr_processor.process_folder(input_dir="data/raw_images", output_dir="data/processed_data")

输出格式
识别结果：data/processed_data/目录下的 CSV 文件，包含以下字段：
systolic_blood_pressure：收缩压
diastolic_blood_pressure：舒张压
pulse_rate：心率
参数配置
关键参数在configs/ocr_config.yaml中设置：
weights_path：YOLO 模型权重路径
img_dir：输入图像目录
output_dir：处理结果输出目录
聚类参数：distance_threshold、group_threshold等