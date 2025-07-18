import threading
from api.ocr_api import app
from data_processing.ocr_processor import OCRProcessor
import yaml
import os

if __name__ == '__main__':
    # 加载配置
    with open("configs/ocr_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化OCR处理器
    ocr_processor = OCRProcessor(config["ocr_config"])
    
    # 启动文件夹处理线程
    folder_thread = threading.Thread(target=ocr_processor.process_folder, daemon=True)
    folder_thread.start()
    
    # 启动API服务
    print("启动Web服务器，支持以下接口:")
    print("- http://localhost:5000/              查看API服务状态")
    print("- http://localhost:5000/latest        获取最新处理结果（仅文本）")
    print("- http://localhost:5000/status        查看处理进度")
    print("- http://localhost:5000/results       获取所有处理结果（仅文本，分页）")
    print("- http://localhost:5000/text          获取所有识别文本（简单列表）")
    print("- http://localhost:5000/image/<name>  查看指定的标注图片")
    print("- http://localhost:5000/history       查询OCR历史记录（支持分页和搜索）")
    print("- http://localhost:5000/history/search?keyword=关键词  搜索OCR历史记录")
    print(f"处理后的图片将保存在 {os.path.abspath(ocr_processor.output_dir)} 目录")
    print(f"OCR结果将同时保存到MySQL数据库")
    
    app.run(host='0.0.0.0', port=5000, debug=False)