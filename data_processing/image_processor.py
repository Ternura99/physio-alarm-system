import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class ImageProcessor:
    """图像处理辅助类"""
    
    def __init__(self, y_threshold: int = 20, group_threshold: int = 1000):
        """初始化图像处理工具"""
        self.y_threshold = y_threshold
        self.group_threshold = group_threshold

    def cluster_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """基于纵坐标差距对检测框进行聚类"""
        clusters = []
        for box in boxes:
            merged = False
            for cluster in clusters:
                existing_box = cluster[0]
                existing_y1, existing_y2 = existing_box[1], existing_box[3]
                current_y1, current_y2 = box[1], box[3]

                y1_diff = abs(existing_y1 - current_y1)
                y2_diff = abs(existing_y2 - current_y2)

                if y1_diff < self.y_threshold and y2_diff < self.y_threshold:
                    # 合并框
                    x1 = min(existing_box[0], box[0])
                    y1 = min(existing_box[1], box[1])
                    x2 = max(existing_box[2], box[2])
                    y2 = max(existing_box[3], box[3])
                    cluster[0] = (x1, y1, x2, y2)
                    merged = True
                    break
            if not merged:
                clusters.append([box])

        return [cluster[0] for cluster in clusters]

    def group_clusters(self, merged_boxes: List[Tuple[int, int, int, int]]) -> List[List[Tuple[int, int, int, int]]]:
        """将聚类后的框分组"""
        groups = []
        for box in merged_boxes:
            merged = False
            for group in groups:
                last_box = group[-1]
                center_x = (last_box[0] + last_box[2]) / 2
                center_y = (last_box[1] + last_box[3]) / 2
                current_center_x = (box[0] + box[2]) / 2
                current_center_y = (box[1] + box[3]) / 2
                distance = np.sqrt((center_x - current_center_x)**2 + (center_y - current_center_y)**2)
                
                if distance < self.group_threshold:
                    group.append(box)
                    merged = True
                    break
            if not merged:
                groups.append([box])
                
        return groups

    def draw_results_on_image(self, image: np.ndarray, grouped_boxes: List[List[Tuple[int, int, int, int]]], 
                              ocr_results: List[Dict[str, Any]]) -> np.ndarray:
        """在图像上绘制检测结果和OCR识别结果"""
        annotated_image = image.copy()
        
        for group_idx, (group, group_result) in enumerate(zip(grouped_boxes, ocr_results)):
            # 为每个组使用不同的颜色
            color = (0, 255, 0)  # 绿色
            if group_idx % 3 == 1:
                color = (0, 0, 255)  # 红色
            elif group_idx % 3 == 2:
                color = (255, 0, 0)  # 蓝色
                
            # 绘制每个区域框
            for i, (box, detection) in enumerate(zip(group, group_result["results"])):
                # 绘制矩形框
                cv2.rectangle(annotated_image, 
                             (detection["box"][0], detection["box"][1]), 
                             (detection["box"][2], detection["box"][3]), 
                             color, 2)
                
                # 提取识别文本
                texts = []
                for text_item in detection["text_results"]:
                    texts.append(f"{text_item['text']} ({text_item['confidence']:.2f})")
                
                text = ", ".join(texts) if texts else "No Text"
                
                # 在框上方绘制文本
                cv2.putText(annotated_image, 
                           text, 
                           (detection["box"][0], detection["box"][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
        return annotated_image