import torch
import sys
import os
import dancher_tools as dt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class_names = {
    0: "Background",
    1: "Oil Spill",
    2: "Look-alike",
    3: "Ship",
    4: "Land"
}
def main():
    # 解析参数
    args = dt.utils.get_config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取数据加载器
    train_loader, val_loader = dt.utils.get_dataloaders(args)

    # 初始化模型
    model = dt.utils.get_model(args, device)
    
    metrics = dt.utils.get_metrics(args)

    # 加载或迁移模型权重
    dt.utils.load_weights(model, args)
    
    # 定义损失函数和优化器
    criterion = dt.utils.get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 判断是否需要使用置信学习
    if args.conf_threshold is not None:
        train_loader = dt.utils.apply_CL(args.conf_threshold, model, train_loader, device)

    # 通过 `compile` 方法配置模型
    model.compile(optimizer=optimizer, criterion=criterion, metrics=metrics, loss_weights=args.loss_weights)

    # 开始训练模型
    model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        model_save_dir=args.model_save_dir,
        patience=args.patience,
        delta=args.delta,
        save_interval=args.save_interval,
        class_names=args.ds['class_name']
    )

    # 评估模型性能
    
    # 开始评估模型
    avg_loss, avg_metrics, per_class_avg_metrics = model.evaluate(data_loader=val_loader,class_names=args.ds['class_name'])


if __name__ == '__main__':
    main()
