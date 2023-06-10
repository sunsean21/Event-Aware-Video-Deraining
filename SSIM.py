import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import cv2
import utils.LightFlowNet as LightFlowNet
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, seq_len, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

class seq_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.backwarp_tenGrid = {}

    @staticmethod
    def tv1_norm3d(x, weight=[1.0, 1.0, 1.0]):
        '''
        Tv norm.
        :param x: B x 3 x num_frame x p x p
        :param weight: list with length 3
        '''
        B, C, N = x.shape[:3]
        x_tv = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().sum() * weight[0]
        y_tv = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().sum() * weight[1]
        z_tv = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().sum() * weight[2]
        tv_loss = (x_tv + y_tv + z_tv) / (B*C*N)
        return tv_loss
    
    def calc_psnr(self, im1, im2, border=0):
        mse = (im1 - im2).square().mean()
        return 20 * torch.log10(255.0 / mse.sqrt())

    def align_loss(self, imgs):
        seq_len = imgs.shape[2]
        loss = 0
        for i in range(seq_len-1):
            img1 = imgs[:,:,i]
            img2 = imgs[:,:,i+1]
            tenOutput21 = LightFlowNet.estimate(img1, img2)
            warped21 = LightFlowNet.backwarp(img2, tenOutput21)
            Mask1 = warped21!=0
            loss += ((Mask1*img1 - warped21).abs()).mean()
            tenOutput12 = LightFlowNet.estimate(img2, img1)
            warped12 = LightFlowNet.backwarp(img1, tenOutput12)
            Mask2 = warped12!=0
            loss += ((Mask2*img2 - warped12).abs()).mean()
        return loss / (2*(seq_len-1))
           
           
    ''' 
    def forward(self, imgs1, imgs2, if_tv_loss=False):
        (_, channel, seq_len, _, _) = imgs1.size()

        if channel == self.channel and self.window.data.type() == imgs1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if imgs1.is_cuda:
                window = window.cuda(imgs1.get_device())
            window = window.type_as(imgs1)
            
            self.window = window
            self.channel = channel
        # return _ssim(imgs1[:,:,0,:,:], imgs2[:,:,0,:,:], window, self.window_size, channel, self.size_average)
        seq_loss = None
        seq_mse = None
        for i in range(seq_len):
            if seq_loss is None:
                seq_loss = _ssim(imgs1[:,:,i,:,:], imgs2[:,:,i,:,:], window, self.window_size, channel, self.size_average)
                # seq_psnr = self.calc_psnr(imgs1[:,:,i,:,:], imgs2[:,:,i,:,:])
                seq_mse = ((imgs1[:,:,i,:,:] - imgs2[:,:,i,:,:])**2).mean().sqrt()
            else:
                seq_loss = seq_loss + _ssim(imgs1[:,:,i,:,:], imgs2[:,:,i,:,:], window, self.window_size, channel, self.size_average)
                # seq_psnr += self.calc_psnr(imgs1[:,:,i,:,:], imgs2[:,:,i,:,:])
                seq_mse = seq_mse + ((imgs1[:,:,i,:,:] - imgs2[:,:,i,:,:])**2).mean().sqrt()
        
        #tv_loss = self.tv0_norm3d(imgs2) / 255. if if_tv_loss == 1 else 0
        align_loss = self.align_loss(imgs2)
        print(seq_loss.item() / seq_len - 1, 0.01 / (seq_mse.item() / seq_len), -align_loss.item())
        
        return seq_loss / seq_len - 1 - align_loss# + 0.01 / (seq_mse / seq_len) + 0.01 / align_loss #+ seq_psnr / seq_len / 45
    '''
    def forward(self, imgs1, imgs2, if_sa_loss=False):
        align_loss = 0
        seq_mse = ((imgs1 - imgs2)**2).mean().sqrt()
        if if_sa_loss:
            align_loss = self.align_loss(imgs2)
            print(seq_mse.item(), 0.1 * align_loss.item())
        return - seq_mse - 0.1 * align_loss
     
