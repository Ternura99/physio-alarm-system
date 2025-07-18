import cv2
import os
import numpy as np
import time
from typing import Dict, List, Any, Optional
from ultralytics import YOLO
from paddleocr import PaddleOCR
from .image_processor import ImageProcessor

class OCRProcessor:
    """大鼠麻醉实验图像OCR处理核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化OCR处理器"""
        # 配置参数
        self.config = config
        self.weights_path = config["weights_path"]
        self.img_dir = config["img_dir"]
        self.output_dir = config["output_dir"]
        self.y_threshold = config["y_threshold"]
        self.distance_threshold = config["distance_threshold"]
        self.group_threshold = config["group_threshold"]
        
        # 初始化模型
        self.model = YOLO(self.weights_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # 图像处理工具
        self.image_processor = ImageProcessor(
            y_threshold=self.y_threshold,
            group_threshold=self.group_threshold
        )
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储最新结果
        self.latest_results = {}

    def process_image(self, frame: np.ndarray, filename: Optional[str] = None) -> Dict[str, Any]:
        """处理单张图像，返回OCR结果"""
        result_data = {
            "filename": filename,
            "ocr_results": [],
            "timings": {},
            "annotated_image": None
        }
        
        start_time = time.time()  # 记录处理开始时间

        # 目标检测
        detection_start = time.time()
        results = self.model(frame)
        detection_end = time.time()
        result_data["timings"]["detection_time"] = round(detection_end - detection_start, 2)

        # 提取检测框
        original_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                original_boxes.append((x1, y1, x2, y2))

        # 聚类和分组
        clustering_start = time.time()
        merged_boxes = self.image_processor.cluster_boxes(original_boxes)
        grouped_boxes = self.image_processor.group_clusters(merged_boxes)
        clustering_end = time.time()
        result_data["timings"]["clustering_time"] = round(clustering_end - clustering_start, 2)

        # OCR识别
        ocr_start = time.time()
        for group_idx, group in enumerate(grouped_boxes, start=1):
            group_results = []
            for box in group:
                # 获取区域坐标并扩展
                x1, y1, x2, y2 = box
                roi_x1 = max(0, x1 - 20)
                roi_y1 = max(0, y1 - 20)
                roi_x2 = min(frame.shape[1], x2 + 20)
                roi_y2 = min(frame.shape[0], y2 + 20)

                # 裁剪区域并进行OCR识别
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                text = self.ocr.ocr(roi, cls=True)
                
                # 处理OCR结果
                text_results = []
                if text and len(text) > 0 and len(text[0]) > 0:
                    for line in text[0]:
                        if len(line) >= 2:
                            coords = [[float(p[0]), float(p[1])] for p in line[0]]
                            text_info = {
                                "text": line[1][0],
                                "confidence": float(line[1][1]),
                                "coords": coords
                            }
                            text_results.append(text_info)
            
                group_results.append({
                    "box": [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)],
                    "text_results": text_results
                })
            
            result_data["ocr_results"].append({
                "group_id": group_idx,
                "results": group_results
            })
        
        ocr_end = time.time()
        result_data["timings"]["ocr_time"] = round(ocr_end - ocr_start, 2)

        # 绘制结果并保存
        annotated_image = self.image_processor.draw_results_on_image(
            frame, grouped_boxes, result_data["ocr_results"]
        )
        
        if filename:
            output_filename = os.path.join(self.output_dir, f"annotated_{filename}")
            cv2.imwrite(output_filename, annotated_image)
            result_data["annotated_image"] = output_filename
        
        end_time = time.time()
        result_data["timings"]["total_processing_time"] = round(end_time - start_time, 2)
        
        return result_data

    def process_folder(self) -> None:
        """持续监控并处理指定文件夹中的所有图像"""
        print(f"开始监控文件夹: {self.img_dir}")
        
        while True:
            try:
                # 获取所有图像文件
                image_files = [
                    f for f in os.listdir(self.img_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ]
                
                if not image_files:
                    print(f"警告: 文件夹 {self.img_dir} 中没有找到图像文件")
                    time.sleep(5)
                    continue
                
                # 按文件名排序
                image_files.sort()
                
                # 处理未处理的图像
                unprocessed_files = [f for f in image_files if f not in self.latest_results]
                
                if not unprocessed_files:
                    print(f"所有图像都已处理完成，等待新图像...")
                    time.sleep(5)
                    continue
                
                print(f"找到 {len(unprocessed_files)} 张新图像待处理")
                
                # 处理每张图像
                for filename in unprocessed_files:
                    try:
                        image_path = os.path.join(self.img_dir, filename)
                        frame = cv2.imread(image_path)
                        
                        if frame is None:
                            print(f"无法读取图像: {image_path}")
                            continue
                        
                        # 处理图像
                        result = self.process_image(frame, filename=filename)
                        self.latest_results[filename] = result
                        
                        # 打印识别文本
                        ocr_text = []
                        for group in result["ocr_results"]:
                            for item in group["results"]:
                                for text_item in item["text_results"]:
                                    ocr_text.append(text_item["text"])
                        
                        if ocr_text:
                            print(f"{filename}: {' '.join(ocr_text)}")
                        else:
                            print(f"{filename}: 无识别结果")
                    
                    except Exception as e:
                        print(f"处理图像 {filename} 时出错: {str(e)}")
            
            except Exception as e:
                print(f"处理文件夹时出错: {str(e)}")
                time.sleep(5)