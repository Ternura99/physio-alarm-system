import yaml
import torch
from utils.data_utils import load_ocr_data
from model_selection.model_selector import ModelSelector


def main():
    # 加载配置
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    # 1. 加载OCR输出的标准化数据（衔接OCR模块）
    print("从OCR输出加载数据...")
    train_loader, test_loader, scaler = load_ocr_data(
        ocr_output_path=config['model_config']['ocr_output_path'],
        seq_length=config['model_config']['seq_length'],
        pred_length=config['model_config']['pred_length']
    )
    
    # 2. 初始化模型选择器并自动选择最优模型
    print("开始模型自动选择与优化...")
    model_selector = ModelSelector(config['model_config'], device)
    best_model_info = model_selector.select_best_model(train_loader, test_loader)
    
    # 3. 保存最终最优模型（供后续报警模块使用）
    print(f"最优模型已选出：{best_model_info['model_name']}")
    # （可选：保存模型权重）
    torch.save(best_model_info['model'].state_dict(), 
               f"results/model_selection/best_{best_model_info['model_name']}.pth")


if __name__ == '__main__':
    main()