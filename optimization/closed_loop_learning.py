# optimization/closed_loop_learning.py
"""
大鼠麻醉监测系统 - 闭环学习优化
实现从误报分析到策略应用的完整闭环流程
"""

import os
import logging
import pandas as pd
import torch
from .false_alert_analyzer import FalseAlertAnalyzer
from .false_alert_strategies import apply_strategy
from models import get_model_by_name
from utils import load_config, save_config, train_model, evaluate_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ClosedLoopLearning")


class ClosedLoopOptimizer:
    """闭环学习优化器，实现从误报到策略应用的完整流程"""
    
    def __init__(self, config_path="configs/optimization_config.yaml"):
        """初始化闭环优化器"""
        self.config = load_config(config_path)
        self.analyzer = FalseAlertAnalyzer(self.config)
        self.model_config = load_config("configs/model_config.yaml")
        self.alarm_config = load_config("configs/alarm_config.yaml")
        
    def run(self, false_alerts, true_values, model, data_loader, device="cpu"):
        """执行闭环优化流程"""
        logger.info("开始闭环优化流程...")
        
        # 1. 分析误报特征
        logger.info("分析误报特征...")
        feature_df = self.analyzer.analyze(
            false_alerts=false_alerts,
            true_values=true_values,
            pre_window=self.config.get("pre_window", 100)
        )
        
        if feature_df.empty:
            logger.info("未检测到误报，无需优化")
            return self.config, None
        
        # 2. 确定需要优化的特征（按出现频率排序）
        logger.info("确定需要优化的特征...")
        feature_counts = {}
        for col in feature_df.columns:
            if col.endswith("_strategy") and not col.startswith("Time Pos"):
                feature_name = col.replace("_strategy", "")
                count = len(feature_df[~feature_df[col].isna()])
                if count > 0:
                    feature_counts[feature_name] = count
        
        # 只优化出现频率最高的前3个特征
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(f"待优化特征: {top_features}")
        
        # 3. 应用优化策略，更新配置
        logger.info("应用优化策略...")
        updated_config = self.config.copy()
        for feature_name, _ in top_features:
            updated_config = apply_strategy(feature_name, updated_config)
        
        # 4. 如果需要模型再训练（如Severity_num策略），则执行再训练
        if "retrain_weight" in updated_config.get("other_features", {}):
            logger.info("检测到需要模型再训练的策略，开始再训练...")
            retrain_weight = updated_config["other_features"]["retrain_weight"]
            
            # 为高严重度误报样本增加权重
            sample_weights = np.ones(len(true_values))
            for alert in false_alerts:
                if alert.get("severity", 0) >= 2:  # 高严重度
                    idx = alert["index"]
                    sample_weights[idx] *= retrain_weight
            
            # 再训练模型
            model_type = self.model_config.get("model_type", "lstm")
            new_model = get_model_by_name(model_type, self.model_config)
            new_model = train_model(
                model=new_model,
                train_loader=data_loader,
                epochs=self.model_config.get("retrain_epochs", 5),
                sample_weights=sample_weights,
                device=device
            )
            
            # 评估新模型
            metrics = evaluate_model(new_model, data_loader, device=device)
            logger.info(f"再训练后模型性能: {metrics}")
            
            # 保存新模型
            model_path = "results/optimization_results/optimized_model.pth"
            torch.save(new_model.state_dict(), model_path)
            logger.info(f"优化后的模型已保存至: {model_path}")
        else:
            new_model = None
        
        # 5. 保存更新后的配置
        config_path = "configs/optimized_config.yaml"
        save_config(updated_config, config_path)
        logger.info(f"优化配置已保存至: {config_path}")
        
        logger.info("闭环优化流程完成!")
        return updated_config, new_model