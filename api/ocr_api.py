from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional
from data_processing.ocr_processor import OCRProcessor
from utils.db_helpers import DatabaseHelper
import yaml

app = Flask(__name__)
CORS(app)

# 加载配置
with open("configs/ocr_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化OCR处理器和数据库助手
ocr_processor = OCRProcessor(config["ocr_config"])
db_helper = DatabaseHelper(config["db_config"])

@app.route('/status', methods=['GET'])
def get_status() -> jsonify:
    """获取处理状态"""
    total_images = len([
        f for f in os.listdir(ocr_processor.img_dir) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    processed_images = len(ocr_processor.latest_results)
    
    return jsonify({
        'success': True,
        'data': {
            'total_images': total_images,
            'processed_images': processed_images,
            'processing_progress': f"{processed_images}/{total_images}",
            'progress_percentage': round(processed_images / max(total_images, 1) * 100, 2),
            'processed_files': list(ocr_processor.latest_results.keys())
        },
        'message': '获取处理状态成功'
    })

@app.route('/results', methods=['GET'])
def get_all_results() -> jsonify:
    """获取所有处理结果（仅文本）"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # 按处理时间排序
    sorted_results = sorted(
        ocr_processor.latest_results.items(), 
        key=lambda x: os.path.getmtime(os.path.join(ocr_processor.img_dir, x[0])) 
        if os.path.exists(os.path.join(ocr_processor.img_dir, x[0])) else 0,
        reverse=True
    )
    
    # 分页处理
    total = len(sorted_results)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    
    current_page_results = []
    for i in range(start, end):
        if i < len(sorted_results):
            filename, result = sorted_results[i]
            
            # 提取文本
            texts = []
            for group in result["ocr_results"]:
                for item in group["results"]:
                    for text_item in item["text_results"]:
                        texts.append(text_item["text"])
            
            current_page_results.append({
                'filename': filename,
                'text': ' '.join(texts) if texts else '无识别结果'
            })
    
    return jsonify({
        'success': True,
        'data': {
            'results': current_page_results,
            'pagination': {
                'total': total,
                'page': page,
                'per_page': per_page,
                'pages': (total + per_page - 1) // per_page
            }
        },
        'message': '获取处理结果成功'
    })

@app.route('/ocr', methods=['POST'])
def ocr_detect() -> jsonify:
    """OCR识别API接口"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': '请求格式错误，需要提供image字段'
            }), 400
        
        # 解码Base64图片数据
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': '图片解码失败'
            }), 400
        
        # 生成唯一文件名
        filename = f"api_{int(time.time())}.jpg"
        
        # 处理图片
        result = ocr_processor.process_image(image, filename)
        
        # 保存到数据库
        all_texts = []
        for group in result["ocr_results"]:
            for item in group["results"]:
                for text_item in item["text_results"]:
                    all_texts.append(text_item["text"])
        
        text_result = ' '.join(all_texts) if all_texts else '无识别结果'
        db_helper.save_to_db(filename, text_result, result["timings"]["total_processing_time"])
        
        return jsonify({
            'success': True,
            'data': result,
            'message': '处理成功',
            'image_url': f"/image/{os.path.basename(result['annotated_image'])}"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理失败: {str(e)}'
        }), 500

@app.route('/image/<filename>', methods=['GET'])
def get_image(filename: str) -> send_file:
    """获取处理后的标注图片"""
    try:
        file_path = os.path.join(ocr_processor.output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': f'图片不存在: {filename}'
            }), 404
        
        return send_file(file_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取图片失败: {str(e)}'
        }), 500

@app.route('/history', methods=['GET'])
def get_history() -> jsonify:
    """获取OCR历史记录"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    search = request.args.get('search', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    
    results = db_helper.get_history(
        page=page, 
        per_page=per_page, 
        search=search, 
        start_date=start_date, 
        end_date=end_date
    )
    
    return jsonify({
        'success': True,
        'data': results,
        'message': f'共获取到 {len(results["history"])} 条历史记录'
    })

@app.route('/history/search', methods=['GET'])
def search_history() -> jsonify:
    """搜索OCR历史记录"""
    keyword = request.args.get('keyword', '')
    
    if not keyword:
        return jsonify({
            'success': False,
            'message': '请提供搜索关键词'
        }), 400
    
    results = db_helper.search_history(keyword)
    
    return jsonify({
        'success': True,
        'data': results['results'],
        'message': f'共找到 {results["count"]} 条匹配记录'
    })

@app.route('/', methods=['GET'])
def index() -> jsonify:
    """API服务状态信息"""
    return jsonify({
        'success': True,
        'message': 'OCR服务器已启动',
        'endpoints': {
            'GET /': '获取服务状态',
            'GET /latest': '获取最新处理结果（仅文本）',
            'GET /image/<filename>': '获取处理后的标注图片',
            'POST /ocr': '手动提交图片进行OCR识别',
            'GET /status': '查看处理进度',
            'GET /results': '获取所有处理结果（仅文本，分页）',
            'GET /text': '获取所有识别文本（简单列表）',
            'GET /history': '查询OCR历史记录（支持分页和搜索）',
            'GET /history/search': '搜索OCR历史记录'
        }
    })

if __name__ == '__main__':
    # 初始化数据库
    try:
        db_helper.init_db()
        print("数据库初始化成功")
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        print("请检查数据库配置并重新启动")
        exit(1)
    
    app.run(host='0.0.0.0', port=5000, debug=False)