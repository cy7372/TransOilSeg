# inference.py

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from utils.general import set_seed, setup_args, model_loader
from utils.loss import DiceLoss, FocalLoss, GPLoss, CombinedLoss
from segmentation_models_pytorch.losses import JaccardLoss
from utils.datasets import load_dataset, RandomGenerator
from utils.general import test_dataset

def main():
    config_file = 'configs/M4D_shared.ini'
    args = setup_args(config_file=config_file)

    set_seed(42)
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = args.deterministic

    print("=" * 80)
    print("Running inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_loader(args, device)

    # 加载损失函数（如仅推理可省略）
    criterion = CombinedLoss(
        losses=[
            DiceLoss(args.num_classes),
            JaccardLoss("multiclass"),
            FocalLoss(args.num_classes),
            GPLoss()
        ],
        weights=[0.25, 0.25, 0.25, 0.25]
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    net.load(optimizer=optimizer, mode=2, model_dir=args.exp_path, specified_path=args.model_path)
    net.eval()

    infer_transform = transforms.Compose([
        RandomGenerator(output_size=[args.img_size, args.img_size])
    ])

    for dataset_name in args.datasets:
        print(f"\nEvaluating dataset: {dataset_name}")
        db_test = load_dataset(
            base_dir=args.test_paths[dataset_name],
            transform=infer_transform
        )
        miou = test_dataset(net, args, db_test, device=device)
        print(f"[{dataset_name}] mIoU: {miou:.2f}%")

if __name__ == "__main__":
    main()
