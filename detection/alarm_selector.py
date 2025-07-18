import yaml
from .fixed_baseline import FixedBaselineAnalyzer
from .dynamic_baseline import DynamicBaselineAnalyzer


class AlarmModeSelector:
    def __init__(self, config_path="configs/alarm_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def select_mode(self, mode):
        """根据用户选择返回对应报警分析器"""
        if mode not in ["fixed", "dynamic"]:
            raise ValueError("模式必须为 'fixed' 或 'dynamic'")
        
        # 初始化对应模式分析器
        if mode == "fixed":
            return FixedBaselineAnalyzer(self.config)
        else:
            return DynamicBaselineAnalyzer(self.config)

    def run(self, mode, ocr_output_path):
        """运行选定模式的报警分析"""
        analyzer = self.select_mode(mode)
        # 加载OCR输出数据（衔接OCR模块）
        test_loader = analyzer.load_monitoring_data(ocr_output_path)
        # 执行分析
        results = analyzer.analyze(test_loader)
        print(f"{mode} 模式报警分析完成，结果已保存至 {analyzer.results_dir}")
        return results


# 示例：人工选择模式的交互入口
if __name__ == "__main__":
    # 实际使用时可改为命令行参数或UI输入
    user_mode = input("请选择报警模式（fixed/dynamic）: ").strip().lower()
    selector = AlarmModeSelector()
    # OCR输出数据路径（衔接OCR模块的processed_data）
    ocr_data_path = "data/processed_data"
    selector.run(user_mode, ocr_data_path)