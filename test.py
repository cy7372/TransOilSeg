import torch
import sys
import os
import dancher_tools as dt

# 确保当前路径在sys.path中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 类别映射
class_names = {
    0: "Background",
    1: "Oil Spill",
    2: "Look-alike",
    3: "Ship",
    4: "Land"
}

# 输出格式对齐的宽度
column_width = 15  # 每个项的宽度

def main():
    # 解析参数
    args = dt.utils.get_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据加载器 (测试集)
    _, test_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)
    
    # 获取预定义指标
    metrics = dt.utils.get_metrics(args)

    # 加载模型权重
    model.load(model_dir=args.model_save_dir, mode=args.load_mode, specified_path=args.weights)

    # 定义损失函数
    criterion = dt.utils.get_loss(args)

    # 配置模型
    model.compile(optimizer=None, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始评估模型
    avg_loss, avg_metrics, per_class_avg_metrics = model.evaluate(data_loader=test_loader,class_names=args.ds['class_name'])

    # # 动态生成表头
    # header = f"{'Class':<15}"
    # for metric_name in metrics.keys():
    #     header += f"{metric_name:<15}"
    
    # print(header)
    # print("-" * len(header))

    # # 打印每个类别的评价指标
    # for cls, class_name in class_names.items():
    #     # 构建每个类别的输出字符串，并保证每个字段对齐
    #     class_metrics_str = f"{class_name:<15}"
    #     class_metrics_str += "".join([f"{class_metrics.get(cls, 0):<15.4f}"
    #                                   for metric_name, class_metrics in per_class_avg_metrics.items()])
    #     print(class_metrics_str)


if __name__ == '__main__':
    main()
