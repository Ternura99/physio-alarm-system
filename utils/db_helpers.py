import pymysql
from datetime import datetime
from typing import Dict, Any

class DatabaseHelper:
    """数据库操作辅助类"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """初始化数据库连接"""
        self.db_config = db_config

    def init_db(self) -> None:
        """初始化数据库表"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                # 创建ocr_history表
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ocr_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    text_result TEXT,
                    processing_time FLOAT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
            conn.commit()
        finally:
            conn.close()

    def save_to_db(self, filename: str, text_result: str, processing_time: float) -> None:
        """保存OCR结果到数据库"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                sql = "INSERT INTO ocr_history (filename, text_result, processing_time) VALUES (%s, %s, %s)"
                cursor.execute(sql, (filename, text_result, processing_time))
            conn.commit()
        except Exception as e:
            print(f"保存到数据库失败: {str(e)}")
        finally:
            conn.close()

    def get_history(self, page: int = 1, per_page: int = 10, 
                    search: str = '', start_date: str = '', end_date: str = '') -> Dict[str, Any]:
        """获取OCR历史记录"""
        conn = pymysql.connect(**self.db_config)
        try:
            # 构建SQL查询
            sql = "SELECT * FROM ocr_history WHERE 1=1"
            params = []
            
            if search:
                sql += " AND (filename LIKE %s OR text_result LIKE %s)"
                search_param = f"%{search}%"
                params.extend([search_param, search_param])
            
            if start_date:
                sql += " AND DATE(created_at) >= %s"
                params.append(start_date)
            
            if end_date:
                sql += " AND DATE(created_at) <= %s"
                params.append(end_date)
            
            sql += " ORDER BY created_at DESC"
            
            with conn.cursor() as cursor:
                # 获取总记录数
                count_sql = f"SELECT COUNT(*) as total FROM ({sql}) as t"
                cursor.execute(count_sql, params)
                total = cursor.fetchone()['total']
                
                # 添加分页限制
                sql += " LIMIT %s OFFSET %s"
                offset = (page - 1) * per_page
                params.extend([per_page, offset])
                
                # 执行最终查询
                cursor.execute(sql, params)
                results = cursor.fetchall()
                
                # 格式化日期时间
                for result in results:
                    if 'created_at' in result and result['created_at']:
                        result['created_at'] = result['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'history': results,
                'pagination': {
                    'total': total,
                    'page': page,
                    'per_page': per_page,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        finally:
            conn.close()

    def search_history(self, keyword: str) -> Dict[str, Any]:
        """搜索OCR历史记录"""
        conn = pymysql.connect(**self.db_config)
        try:
            with conn.cursor() as cursor:
                sql = "SELECT * FROM ocr_history WHERE filename LIKE %s OR text_result LIKE %s ORDER BY created_at DESC LIMIT 50"
                cursor.execute(sql, [f'%{keyword}%', f'%{keyword}%'])
                results = cursor.fetchall()
                
                # 格式化日期时间
                for result in results:
                    if 'created_at' in result and result['created_at']:
                        result['created_at'] = result['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                'results': results,
                'count': len(results)
            }
        finally:
            conn.close()