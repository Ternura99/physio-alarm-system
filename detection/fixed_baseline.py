import numpy as np
from .base_analyzer import BaseAlarmAnalyzer
from utils.false_alert_utils import collect_false_alerts, analyze_false_alert_features
from utils.visualization_utils import plot_alarm_results


class FixedBaselineAnalyzer(BaseAlarmAnalyzer):
    def __init__(self, config):
        super().__init__(config)
        self.baseline_ratio = config["fixed_baseline"]["baseline_ratio"]
        self.normal_range_ratio = config["fixed_baseline"]["normal_range"]
        self.analysis_interval = config["fixed_baseline"]["false_alert_analysis_interval"]

    def _compute_baseline(self, true_values):
        """基于初始数据计算固定基线和范围"""
        # 取前N%数据计算基线
        baseline_data = true_values[:int(len(true_values) * self.baseline_ratio)]
        baseline = np.mean(baseline_data)
        # 计算正常范围（基线±比例）
        lower = baseline * self.normal_range_ratio[0]
        upper = baseline * self.normal_range_ratio[1]
        return baseline, (lower, upper)

    def analyze(self, test_loader):
        """固定基线报警分析（含定时误报特征提取）"""
        # 1. 预测与真实值获取
        predictions, true_values = self.predict(test_loader)
        # 展平数据（适配单变量分析）
        predictions = predictions.reshape(-1)
        true_values = true_values.reshape(-1)

        # 2. 计算固定基线和范围
        baseline, limits = self._compute_baseline(true_values)
        print(f"固定基线: {baseline:.2f}, 正常范围: [{limits[0]:.2f}, {limits[1]:.2f}]")

        # 3. 报警判断与结果收集
        results = []
        false_alerts = collect_false_alerts(true_values, predictions, limits)

        # 4. 定时进行误报特征提取（每间隔analysis_interval步）
        total_steps = len(true_values)
        for interval in range(0, total_steps, self.analysis_interval):
            end = min(interval + self.analysis_interval, total_steps)
            # 提取当前区间的误报
            interval_false = [a for a in false_alerts if interval <= a["index"] < end]
            if interval_false:
                # 分析误报特征
                false_features = analyze_false_alert_features(
                    interval_false,
                    true_values,
                    pre_window=self.config["common"]["window_size"]
                )
                # 保存区间误报特征
                false_features.to_csv(
                    os.path.join(self.results_dir, "fixed_baseline", f"false_alert_features_interval_{interval}.csv"),
                    index=False
                )
                print(f"已分析区间 [{interval}-{end}] 的误报特征，共 {len(interval_false)} 条")

        # 5. 可视化与结果保存
        mode_dir = self.save_results({
            "variable": self.variables[0],  # 可扩展多变量
            "baseline": baseline,
            "lower_limit": limits[0],
            "upper_limit": limits[1],
            "false_positives": len([a for a in false_alerts if a["type"] == "false_positive"]),
            "false_negatives": len([a for a in false_alerts if a["type"] == "false_negative"]),
            "total_samples": len(true_values)
        }, "fixed_baseline")

        # 绘制单变量结果（可循环扩展多变量）
        plot_alarm_results(
            true_values,
            predictions,
            limits,
            self.variables[0],
            mode_dir,
            is_fixed=True
        )

        return {
            "mode": "fixed_baseline",
            "results": results,
            "false_alerts": false_alerts
        }