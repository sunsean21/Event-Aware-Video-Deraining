import cv2
import math
import torch
import torch.nn as nn
import numpy as np
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path


def rgb2ycbcr(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    :parame img: uint8 or float ndarray
    '''
    in_im_type = im.dtype
    im = im.astype(np.float64)
    if in_im_type != np.uint8:
        im *= 255.
    # convert
    if only_y:
        rlt = np.dot(im, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im, np.array([[65.481,  -37.797, 112.0  ],
                                      [128.553, -74.203, -93.786],
                                      [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if in_im_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.

    return rlt.astype(in_im_type)

def rgb2ycbcrTorch(im, only_y=True):
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    im_temp = im.permute([0,2,3,1]) * 255.0  # N x H x W x C --> N x H x W x C, [0,255]
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calc_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def batch_psnr(img, imclean, border=0, ycbcr=False):
    '''
    input: n x c x w x h 
    '''
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)

    Img = img_as_ubyte(img.data.numpy())
    Iclean = img_as_ubyte(imclean.data.numpy())
    PSNR = 0
    h, w = Iclean.shape[2:]
    for i in range(Img.shape[0]):
        PSNR += calc_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (PSNR/Img.shape[0])

def batch_ssim(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img_as_ubyte(img.data.numpy())
    Iclean = img_as_ubyte(imclean.data.numpy())
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calc_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (SSIM/Img.shape[0])
