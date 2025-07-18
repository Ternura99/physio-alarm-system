# scripts/run_full_pipeline.py
"""
大鼠麻醉监测系统 - 完整流程自动化脚本
执行从OCR数据提取到模型选择、报警分析及自优化的全流程
"""

import os
import argparse
import logging
from datetime import datetime
import pandas as pd
import torch
import optuna

# 导入各模块组件
from data_processing import OCRProcessor
from models import LSTMModel, BiLSTMModel, GRUModel, CNNModel, CNNLSTMModel, TransformerModel, TCNModel, XGBoostModel
from model_selection import ModelSelector
from detection import AlarmModeSelector
from optimization import FalseAlertAnalyzer
from utils import (
    load_ocr_data, train_model, evaluate_model, 
    collect_false_alerts, analyze_false_alert_features,
    plot_alarm_results, plot_model_performance,
    load_config, save_config
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FullPipeline")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="大鼠麻醉监测系统 - 完整流程")
    parser.add_argument("--ocr-config", default="configs/ocr_config.yaml", help="OCR配置文件路径")
    parser.add_argument("--model-config", default="configs/model_config.yaml", help="模型配置文件路径")
    parser.add_argument("--alarm-config", default="configs/alarm_config.yaml", help="报警配置文件路径")
    parser.add_argument("--optimization-config", default="configs/optimization_config.yaml", help="优化配置文件路径")
    parser.add_argument("--input-dir", default="data/raw_images", help="输入图像目录")
    parser.add_argument("--output-dir", default="results", help="结果输出目录")
    parser.add_argument("--alarm-mode", default="fixed", choices=["fixed", "dynamic"], help="报警模式")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna优化试验次数")
    return parser.parse_args()


def main():
    """执行完整流程"""
    args = parse_args()
    logger.info(f"启动完整流程，参数: {args}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "ocr_results"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "model_selection"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "alarm_results", args.alarm_mode), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "optimization_results"), exist_ok=True)
    
    # 加载配置
    ocr_config = load_config(args.ocr_config)
    model_config = load_config(args.model_config)
    alarm_config = load_config(args.alarm_config)
    optimization_config = load_config(args.optimization_config)
    
    # ==================== 1. OCR数据处理 ====================
    logger.info("开始OCR数据处理...")
    ocr_processor = OCRProcessor(ocr_config)
    ocr_results = ocr_processor.process_folder(
        input_dir=args.input_dir, 
        output_dir=os.path.join(args.output_dir, "ocr_results")
    )
    logger.info(f"OCR处理完成，共处理 {len(ocr_results)} 张图像")
    
    # 保存OCR结果
    ocr_output_path = os.path.join(args.output_dir, "ocr_results", "ocr_processed_data.csv")
    pd.DataFrame(ocr_results).to_csv(ocr_output_path, index=False)
    logger.info(f"OCR结果已保存至: {ocr_output_path}")
    
    # ==================== 2. 模型选择 ====================
    logger.info("开始模型选择流程...")
    # 加载OCR处理后的数据
    train_loader, test_loader, scaler = load_ocr_data(
        data_path=ocr_output_path,
        seq_length=model_config["seq_length"],
        pred_length=model_config["pred_length"],
        batch_size=model_config["batch_size"]
    )
    
    # 定义候选模型
    model_classes = {
        "lstm": LSTMModel,
        "bilstm": BiLSTMModel,
        "gru": GRUModel,
        "cnn": CNNModel,
        "cnnlstm": CNNLSTMModel,
        "transformer": TransformerModel,
        "tcn": TCNModel,
        "xgboost": XGBoostModel
    }
    
    # 初始化模型选择器
    selector = ModelSelector(model_config)
    
    # 优化并选择最佳模型
    best_model, best_params, study = selector.select_best_model(
        model_classes=model_classes,
        train_loader=train_loader,
        test_loader=test_loader,
        n_trials=args.n_trials,
        device=args.device
    )
    
    # 保存最佳模型
    model_output_path = os.path.join(args.output_dir, "model_selection", "best_model.pth")
    torch.save(best_model.state_dict(), model_output_path)
    logger.info(f"最佳模型已保存至: {model_output_path}")
    
    # 保存优化结果
    study_df = study.trials_dataframe()
    study_df.to_csv(os.path.join(args.output_dir, "model_selection", "optimization_results.csv"), index=False)
    logger.info(f"模型优化结果已保存")
    
    # 评估最佳模型
    metrics = evaluate_model(best_model, test_loader, device=args.device)
    logger.info(f"最佳模型评估结果: {metrics}")
    
    # 生成模型性能可视化
    plot_model_performance(
        metrics=metrics,
        output_path=os.path.join(args.output_dir, "model_selection", "model_performance.png")
    )
    
    # ==================== 3. 报警分析 ====================
    logger.info(f"开始{args.alarm_mode}报警模式分析...")
    # 预测测试数据
    predictions, true_values = selector.predict_with_best_model(
        best_model, test_loader, device=args.device, scaler=scaler
    )
    
    # 初始化报警选择器
    alarm_selector = AlarmModeSelector(alarm_config)
    
    # 执行报警分析
    alarm_results = alarm_selector.run(
        mode=args.alarm_mode,
        predictions=predictions,
        true_values=true_values,
        model=best_model,
        data_loader=test_loader,
        scaler=scaler
    )
    
    # 保存报警结果
    alarm_output_path = os.path.join(
        args.output_dir, "alarm_results", args.alarm_mode, "alarm_summary.csv"
    )
    pd.DataFrame(alarm_results).to_csv(alarm_output_path, index=False)
    logger.info(f"报警结果已保存至: {alarm_output_path}")
    
    # 生成报警可视化
    plot_alarm_results(
        predictions=predictions,
        true_values=true_values,
        alarms=alarm_results,
        output_path=os.path.join(args.output_dir, "alarm_results", args.alarm_mode, "alarm_visualization.png")
    )
    
    # ==================== 4. 误报分析与优化 ====================
    logger.info("开始误报分析与优化...")
    # 收集误报
    limits = (alarm_config["lower_limit"], alarm_config["upper_limit"])
    false_alerts = collect_false_alerts(true_values, predictions, limits)
    logger.info(f"共检测到 {len(false_alerts)} 个误报")
    
    # 分析误报特征
    feature_df = analyze_false_alert_features(
        false_alerts=false_alerts,
        true_vals=true_values,
        pre_window=optimization_config["pre_window"]
    )
    
    # 保存误报特征分析
    feature_output_path = os.path.join(
        args.output_dir, "optimization_results", "false_alert_features.csv"
    )
    feature_df.to_csv(feature_output_path, index=False)
    logger.info(f"误报特征分析已保存至: {feature_output_path}")
    
    # 应用优化策略
    false_alert_analyzer = FalseAlertAnalyzer(optimization_config)
    updated_config = false_alert_analyzer.apply_strategy(feature_df)
    
    # 保存更新后的配置
    updated_config_path = os.path.join(args.output_dir, "optimization_results", "updated_config.yaml")
    save_config(updated_config, updated_config_path)
    logger.info(f"更新后的配置已保存至: {updated_config_path}")
    
    logger.info("完整流程执行完毕!")


if __name__ == "__main__":
    main()