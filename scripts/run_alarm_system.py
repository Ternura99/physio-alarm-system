import argparse
from detection.alarm_selector import AlarmModeSelector


def main():
    # 解析命令行参数（支持人工选择模式）
    parser = argparse.ArgumentParser(description="大鼠麻醉监测报警系统")
    parser.add_argument("--mode", type=str, required=True, choices=["fixed", "dynamic"],
                        help="选择报警模式：fixed（固定基线）或 dynamic（动态基线）")
    parser.add_argument("--ocr-path", type=str, default="data/processed_data",
                        help="OCR输出的标准化数据路径")
    args = parser.parse_args()

    # 运行报警系统
    selector = AlarmModeSelector()
    selector.run(args.mode, args.ocr_path)


if __name__ == "__main__":
    main()