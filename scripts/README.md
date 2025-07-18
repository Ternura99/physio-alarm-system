# 启动脚本

## 功能说明
提供系统各模块的启动入口，简化运行流程。

## 脚本列表
| 脚本名称               | 功能                          | 用法示例                                  |
|------------------------|-------------------------------|-------------------------------------------|
| `run_ocr_service.py`   | 启动OCR服务，提取生理数据      | `python run_ocr_service.py`               |
| `run_model_pipeline.py`| 启动模型选择流程              | `python run_model_pipeline.py`            |
| `run_alarm_system.py`  | 启动报警系统，支持模式选择    | `python run_alarm_system.py --mode fixed` |
| `run_full_pipeline.py` | 启动完整流程（OCR→模型→报警） | `python run_full_pipeline.py`             |

## 注意事项
- 运行前请确保配置文件（`configs/`目录下）正确设置路径和参数
- 首次运行需初始化数据库（自动执行，见`utils/db_helpers.py`）