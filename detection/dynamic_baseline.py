import numpy as np
from .base_analyzer import BaseAlarmAnalyzer
from utils.false_alert_utils import collect_false_alerts
from utils.visualization_utils import plot_alarm_results


class DynamicBaselineAnalyzer(BaseAlarmAnalyzer):
    def __init__(self, config):
        super().__init__(config)
        self.segment_length = config["dynamic_baseline"]["segment_length"]
        self.normal_range_ratio = config["dynamic_baseline"]["normal_range"]

    def _compute_dynamic_limits(self, true_values):
        """按段计算动态基线和范围"""
        num_segments = len(true_values) // self.segment_length + 1
        limits = []
        for i in range(num_segments):
            start = i * self.segment_length
            end = min((i + 1) * self.segment_length, len(true_values))
            segment = true_values[start:end]
            if len(segment) == 0:
                continue
            # 本段基线为均值
            baseline = np.mean(segment)
            # 本段正常范围
            lower = baseline * self.normal_range_ratio[0]
            upper = baseline * self.normal_range_ratio[1]
            limits.append((lower, upper))
        return limits

    def analyze(self, test_loader):
        """动态基线报警分析"""
        # 1. 预测与真实值获取
        predictions, true_values = self.predict(test_loader)
        predictions = predictions.reshape(-1)
        true_values = true_values.reshape(-1)

        # 2. 计算动态基线范围（每段不同）
        dynamic_limits = self._compute_dynamic_limits(true_values)
        print(f"动态基线共 {len(dynamic_limits)} 段，每段长度 {self.segment_length}")

        # 3. 报警判断（按段匹配范围）
        false_alerts = []
        for i, (p, t) in enumerate(zip(predictions, true_values)):
            # 确定当前点所属段
            seg_idx = min(i // self.segment_length, len(dynamic_limits) - 1)
            lower, upper = dynamic_limits[seg_idx]
            # 判断误报
            if (p < lower or p > upper) and (lower <= t <= upper):
                false_alerts.append({
                    "index": i,
                    "type": "false_positive",
                    "predicted": p,
                    "true": t,
                    "limits": (lower, upper)
                })
            elif (lower <= p <= upper) and (t < lower or t > upper):
                false_alerts.append({
                    "index": i,
                    "type": "false_negative",
                    "predicted": p,
                    "true": t,
                    "limits": (lower, upper)
                })

        # 4. 可视化与结果保存
        mode_dir = self.save_results({
            "variable": self.variables[0],
            "num_segments": len(dynamic_limits),
            "segment_length": self.segment_length,
            "false_positives": len([a for a in false_alerts if a["type"] == "false_positive"]),
            "false_negatives": len([a for a in false_alerts if a["type"] == "false_negative"]),
            "total_samples": len(true_values)
        }, "dynamic_baseline")

        # 绘制动态基线结果
        plot_alarm_results(
            true_values,
            predictions,
            dynamic_limits,
            self.variables[0],
            mode_dir,
            is_fixed=False
        )

        return {
            "mode": "dynamic_baseline",
            "dynamic_limits": dynamic_limits,
            "false_alerts": false_alerts
        }