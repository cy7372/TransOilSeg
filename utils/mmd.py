#!/usr/bin/env python
# encoding: utf-8
import sys
import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    # try:    
    # total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size()[1:]))
    expand_list = [int(total.size(0))] + [int(i) for i in total.size()]
    total0 = total.unsqueeze(0).expand(expand_list)
        # print(source.size())
        # print(target.size())
        # sys.exit()
        # RuntimeError: expand(torch.cuda.FloatTensor{[1, 32, 5, 224, 224]}, size=[32, 32, 5]): the number of sizes provided (3) must be greater or equal to the number of dimensions in the tensor (5)
        # total.size(0) = 32
        # total.size(1) = 5
        # total.unsqueeze(0).size() = [1, 32, 5, 224, 224]
        # total.size() = [32, 5, 224, 224]
        # source.size() = [16, 5, 224, 224]

    total1 = total.unsqueeze(1).expand(expand_list)
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

