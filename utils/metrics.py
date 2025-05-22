import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import copy
from PIL import Image
import os
from torch.nn import functional as F
from torch.autograd import Variable
import cv2

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
         
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
 
    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
 
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score
 
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n) 
 
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)
 
def per_class_PA(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)
 
def compute_mIoU(pred, label, num_classes): 
    
    hist = np.zeros((num_classes, num_classes))
    if len(label.flatten()) != len(pred.flatten()): 
        print(
            'Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(
                len(label.flatten()), len(pred.flatten())))

    #------------------------------------------------#
    #   对一张图片计算21×21的hist矩阵，并累加
    #------------------------------------------------#
    hist += fast_hist(label.flatten(), pred.flatten(),num_classes) 
    # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值

    #------------------------------------------------#
    mIoUs   = per_class_iu(hist)
    mPA     = per_class_PA(hist)
    Oil_mIoU = round(mIoUs[1] * 100, 2)
    mIoU=round(np.nanmean(mIoUs) * 100, 2)
   
    return mIoU,Oil_mIoU