import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import copy
from PIL import Image
import os
from torch.nn import functional as F
from tqdm import tqdm
# utils/general.py

import os
import argparse
import configparser
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_args(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    args = argparse.Namespace()

    args.model_name = config.get('exp_set', 'model_name')
    args.batch_size = config.getint('exp_set', 'batch_size')
    args.max_epochs = config.getint('exp_set', 'max_epochs')
    args.img_size = config.getint('exp_set', 'img_size')
    args.name_classes = config.get('exp_set', 'name_classes').split(',')
    args.ds_path = config.get('exp_set', 'ds_path')
    args.datasets = config.get('exp_set', 'dataset').split(',')
    args.name = config.get('exp_set', 'name', fallback=None)
    args.model_path = config.get('exp_set', 'model_path', fallback=None)
    args.transfer = config.get('exp_set', 'transfer', fallback=False)
    if args.transfer:
        args.source = config.get('exp_set', 'source')
        args.target = config.get('exp_set', 'target')

    args.n_gpu = config.getint('train_set', 'n_gpu')
    args.deterministic = config.getboolean('train_set', 'deterministic')
    args.base_lr = config.getfloat('train_set', 'base_lr')
    args.vit_patches_size = 16
    args.num_classes = len(args.name_classes)

    args.train_paths = {}
    args.test_paths = {}
    for dataset in args.datasets:
        args.train_paths[dataset] = os.path.join(args.ds_path, dataset, 'train', 'npz')
        args.test_paths[dataset] = os.path.join(args.ds_path, dataset, 'test', 'npz')

    args.exp = f'TOS_{args.name}'
    exp_path = os.path.join("./Results", args.exp)
    os.makedirs(exp_path, exist_ok=True)
    args.exp_path = exp_path

    return args

def model_loader(args, device):
    from networks.model_config import get_model_config
    from networks.TransOilSeg import TransOilSeg as TOS

    if args.model_name.lower() == 'transoilseg':
        config = get_model_config()
        config.n_classes = args.num_classes
        config.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size)
        )
        model = TOS(config, img_size=args.img_size, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    return model.to(device)



from .metrics import fast_hist


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def create_color_mask(prediction):
    """将预测结果映射成可视化彩色mask，适用于最多5类（根据你的色表可自定义）"""
    colors = np.array([
        [0, 0, 0],       # 类0
        [255, 0, 0],     # 类1
        [0, 255, 0],     # 类2
        [0, 0, 255],     # 类3
        [255, 255, 0],   # 类4
    ], dtype=np.uint8)
    # 如果类别数多于5，可以扩展colors
    color_mask = colors[prediction % len(colors)]
    return color_mask

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    """
    image, label: Tensor, 需要shape为 [1, C, H, W]
    net: 已加载权重的模型
    classes: 类别数
    patch_size: 推理时输入网络的patch大小
    test_save_path, case: 如需保存可视化mask用
    """
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _, x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        image = zoom(image, (1, patch_size[0]/x, patch_size[1]/y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
    if x != patch_size[0] or y != patch_size[1]:
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    else:
        prediction = out

    hist = fast_hist(label.flatten(), prediction.flatten(), classes)
    metric_list = [calculate_metric_percase(prediction == i, label == i) for i in range(1, classes)]

    if test_save_path and case:
        color_mask = create_color_mask(prediction.astype(np.uint8))
        Image.fromarray(color_mask).save(os.path.join(test_save_path, f"{case}_pred.png"))

    return metric_list, hist

def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    iu[np.isnan(iu)] = 0
    return iu

def test_dataset(model, args, dataset, device, input_size=None, test_save_path=None):
    """
    dataset: 已构造好的Dataset对象
    args: 含有 num_classes, name_classes 等参数
    """
    model.to(device).eval()
    hists = np.zeros((args.num_classes, args.num_classes))
    # 支持传入 test_save_path 实现自动mask保存
    for sampled_batch in tqdm(
        torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False),
        desc="Evaluating dataset"
    ):
        image = sampled_batch["image"].to(device)
        label = sampled_batch["label"].to(device)
        case_name = sampled_batch["case_name"][0] if "case_name" in sampled_batch else None

        _, hist = test_single_volume(
            image, label, model, args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path, case=case_name
        )
        hists += hist

    mIoUs = per_class_iu(hists)
    for ind_class in range(args.num_classes):
        if hasattr(args, "name_classes") and args.name_classes:
            print(f"===> {args.name_classes[ind_class]}:\t mIoU-{mIoUs[ind_class] * 100:.2f}%")
        else:
            print(f"===> class {ind_class}:\t mIoU-{mIoUs[ind_class] * 100:.2f}%")

    overall_mIoU = np.nanmean(mIoUs) * 100
    print(f"===> mIoU: {overall_mIoU:.2f}%")
    return overall_mIoU


def gradient_penalty(netD, real_data, fake_data, l=10):
    batch_size = real_data.size(0)
    alpha = real_data.new_empty((batch_size, 1, 1, 1)).uniform_(0, 1)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=real_data.new_ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * l

    return 


# def get_image_gradients(image):
#     """Returns image gradients (dy, dx) for each color channel.
#     Both output tensors have the same shape as the input: [b, c, h, w].
#     Places the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
#     That means that dy will always have zeros in the last row,
#     and dx will always have zeros in the last column.

#     This can be used to implement the anisotropic 2-D version of the
#     Total Variation formula:
#         https://en.wikipedia.org/wiki/Total_variation_denoising
#     (anisotropic is using l1, isotropic is using l2 norm)

#     Arguments:
#         image: Tensor with shape [b, c, h, w].
#     Returns:
#         Pair of tensors (dy, dx) holding the vertical and horizontal image
#         gradients (1-step finite difference).
#     Raises:
#       ValueError: If `image` is not a 3D image or 4D tensor.
#     """

#     image_shape = image.shape

#     if len(image_shape) == 3:
#         # The input is a single image with shape [height, width, channels].
#         # Calculate the difference of neighboring pixel-values.
#         # The images are shifted one pixel along the height and width by slicing.
#         dx = image[:, 1:, :] - image[:, :-1, :] #pixel_dif2, f_v_1-f_v_2
#         dy = image[1:, :, :] - image[:-1, :, :] #pixel_dif1, f_h_1-f_h_2

#     elif len(image_shape) == 4:
#         # Return tensors with same size as original image
#         #adds one pixel pad to the right and removes one pixel from the left
#         right = F.pad(image, [0, 1, 0, 0])[..., :, 1:]
#         #adds one pixel pad to the bottom and removes one pixel from the top
#         bottom = F.pad(image, [0, 0, 0, 1])[..., 1:, :]

#         #right and bottom have the same dimensions as image
#         dx, dy = right - image, bottom - image

#         #this is required because otherwise results in the last column and row having
#         # the original pixels from the image
#         dx[:, :, :, -1] = 0 # dx will always have zeros in the last column, right-left
#         dy[:, :, -1, :] = 0 # dy will always have zeros in the last row,    bottom-top
#     else:
#       raise ValueError(
#           'image_gradients expects a 3D [h, w, c] or 4D tensor '
#           '[batch_size, c, h, w], not %s.', image_shape)

#     return dy, dx

# def denormalize(x, min_max=(-1.0, 1.0)):
#     '''
#         Denormalize from [-1,1] range to [0,1]
#         formula: xi' = (xi - mu)/sigma
#         Example: "out = (x + 1.0) / 2.0" for denorm
#             range (-1,1) to (0,1)
#         for use with proper act in Generator output (ie. tanh)
#     '''
#     out = (x - min_max[0]) / (min_max[1] - min_max[0])
#     if isinstance(x, torch.Tensor):
#         return out.clamp(0, 1) 
#     elif isinstance(x, np.ndarray):
#         return np.clip(out, 0, 1)
#     else:
#         raise TypeError("Got unexpected object type, expected torch.Tensor or \
#         np.ndarray")