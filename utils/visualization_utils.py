import matplotlib.pyplot as plt
import os


def plot_alarm_results(true_values, predictions, limits, variable, mode_dir, is_fixed=True):
    """绘制报警结果（支持固定/动态基线）"""
    plt.figure(figsize=(12, 6))
    # 指标颜色映射
    colors = {
        'pulse_rate': ('#376439', '#81b095', '#cfeadf'),
        'systolic_blood_pressure': ('#832440', '#c87d98', '#ebcce2'),
        'diastolic_blood_pressure': ('#b7282e', '#dc917b', '#fee3ce')
    }
    true_color, pred_color, range_color = colors.get(variable, ('black', 'gray', 'lightgray'))

    # 绘制真实值和预测值
    plt.plot(true_values, label="True Values", color=true_color)
    plt.plot(predictions, label="Predicted Values", color=pred_color)

    # 绘制基线范围
    if is_fixed:
        # 固定基线：单一线和范围
        baseline = (limits[0] + limits[1]) / 2  # 基线为范围中点
        plt.axhline(baseline, linestyle='--', color='blue', label="Baseline")
        plt.fill_between(
            range(len(true_values)),
            limits[0], limits[1],
            color=range_color, alpha=0.3, label="Normal Range"
        )
    else:
        # 动态基线：每段不同范围
        for i, (lower, upper) in enumerate(limits):
            start = i * len(true_values) // len(limits)
            end = (i + 1) * len(true_values) // len(limits)
            plt.axhspan(lower, upper, xmin=start/len(true_values), xmax=end/len(true_values),
                        color=range_color, alpha=0.3, label="Normal Range" if i == 0 else "")

    # 标记误报点
    false_pos = [(i, p) for i, (p, t) in enumerate(zip(predictions, true_values))
                 if (p < limits[0] or p > limits[1]) and (limits[0] <= t <= limits[1])]
    false_neg = [(i, p) for i, (p, t) in enumerate(zip(predictions, true_values))
                 if (limits[0] <= p <= limits[1]) and (t < limits[0] or t > limits[1])]
    if false_pos:
        fp_indices, fp_values = zip(*false_pos)
        plt.scatter(fp_indices, fp_values, marker='^', color='red', label='False Positive', zorder=10)
    if false_neg:
        fn_indices, fn_values = zip(*false_neg)
        plt.scatter(fn_indices, fn_values, marker='x', color='blue', label='False Negative', zorder=10)

    plt.title(f"{variable} Alarm Results")
    plt.xlabel("Time Step")
    plt.ylabel(variable)
    plt.legend()
    # 保存图片
    img_path = os.path.join(mode_dir, f"{variable}_alarm_plot.png")
    plt.savefig(img_path)
    plt.close()
    return img_path