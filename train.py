import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

from utils.general import set_seed, setup_args, model_loader
from utils.loss import DiceLoss, BCELoss, CombinedLoss
from segmentation_models_pytorch.losses import JaccardLoss
from utils.datasets import load_dataset, RandomGenerator
from utils.general import test_dataset

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        image = batch['image'].to(device)
        label = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss

def main():
    config_file = 'configs/M4D_shared.ini'
    args = setup_args(config_file=config_file)

    set_seed(42)
    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = args.deterministic

    print("="*80)
    print("Running training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_loader(args, device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    net.load(optimizer=optimizer, mode=2, model_dir=args.exp_path, specified_path=args.model_path)
    if hasattr(net, 'last_epoch'):
        start_epoch = net.last_epoch + 1
    else:
        start_epoch = 0

    criterion = CombinedLoss(
        [
            JaccardLoss("multiclass"),   # outputs logits, targets long
            nn.CrossEntropyLoss(),
            DiceLoss(args.num_classes)
        ],
        weights=[0.5, 0.25, 0.25]
    )

    train_transform = transforms.Compose([
        RandomGenerator(output_size=[args.img_size, args.img_size])
    ])
    val_transform = transforms.Compose([
        RandomGenerator(output_size=[args.img_size, args.img_size])
    ])

    train_dataset = load_dataset(
        base_dir=args.train_paths[args.datasets[0]],
        transform=train_transform
    )
    val_dataset = load_dataset(
        base_dir=args.test_paths[args.datasets[0]],
        transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    for epoch in range(start_epoch, args.max_epochs + 1):
        print(f"\nEpoch [{epoch}/{args.max_epochs}]")

        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss = validate(net, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")

        # 每个 epoch 都保存一次
        net.save(epoch, optimizer=optimizer, model_dir=args.exp_path, mode=0)
        scheduler.step()

    print("Training Finished!")

if __name__ == "__main__":
    main()
