"""
数据处理模块：负责通过OCR技术从大鼠麻醉实验图像中提取生理数据并标准化。
包含目标检测、文本识别、数据清洗与格式化功能，为后续模型训练提供输入。
"""
from .ocr_processor import OCRProcessor
from .image_utils import cluster_boxes, group_clusters, draw_results_on_image

__all__ = [
    "OCRProcessor",
    "cluster_boxes",
    "group_clusters",
    "draw_results_on_image"
]