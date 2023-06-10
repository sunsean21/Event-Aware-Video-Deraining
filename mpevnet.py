import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import utils.LightFlowNet as LightFlowNet


class MPEVNet(nn.Module):
    def __init__(self, recurrent_iter=[2, 2, 2], use_GPU=True, opt_dim=32):
        super().__init__()
        class self_atten(nn.Module):
            def __init__(self, in_chans=256):
                super().__init__()
                # self-attention
                self.dv = in_chans
                self.dk = 4
                self.query_conv = nn.Conv3d(in_chans, self.dk, (3, 3, 3), 1, (1, 1, 1))
                self.key_conv = nn.Conv3d(in_chans, self.dk, (3, 3, 3), 1, (1, 1, 1))
                self.softmax  = nn.Softmax(dim=-1)
                self.gamma = nn.Parameter(torch.zeros(1))
                self.value_conv = nn.Conv3d(in_chans, self.dv, (3, 3, 3), 1, (1, 1, 1))
            def forward(self, e):
                # self-attention
                batch_size, _, seq_size_event, row, col = e.size()
                
                query  = self.query_conv(e).view(batch_size,-1,seq_size_event*row*col).permute(0,2,1)
                key =  self.key_conv(e).view(batch_size,-1,seq_size_event*row*col)
                energy =  torch.bmm(query, key) 
                attention = self.softmax(energy) # n x N x N
                value = self.value_conv(e).view(batch_size,-1, seq_size_event*row*col) 
         
                out = torch.bmm(value, attention.permute(0,2,1)) # n x c x N
                out = out.view(batch_size, -1, seq_size_event, row, col) 
         
                e = self.gamma * out + e
                return e
        class atten(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.d = d
                self.conv_event = nn.Sequential(
                    nn.Conv3d(2, self.d, (3, 3, 3), 1, (1, 1, 1), bias=False),
                    nn.BatchNorm3d(num_features=self.d),
                    nn.PReLU(),
                    nn.Conv3d(self.d, self.d, (3, 3, 3), 1, (1, 1, 1), bias=False),
                    nn.BatchNorm3d(num_features=self.d),
                    nn.PReLU(),
                )

                self.conv_events_down = nn.ModuleList()
                din = self.d
                for i in range(2):
                    dout = din * 2
                    self.conv_events_down.append(
                        nn.Sequential(
                            nn.Conv3d(din, dout, (3, 3, 3), 1, (1, 1, 1), bias=False),
                            nn.BatchNorm3d(num_features=dout),
                            nn.PReLU(),
                        )
                    )
                    din = dout

                self.self_atten = self_atten(dout)

                self.conv_events_up = nn.ModuleList()
                for i in range(2):
                    dout = din // 2
                    self.conv_events_up.append(
                        nn.Sequential(
                            nn.Conv3d(din, dout, (3, 3, 3), 1, (1, 1, 1), bias=False),
                            nn.BatchNorm3d(num_features=dout),
                            nn.PReLU(),
                        )
                    )
                    din = dout

                self.conv_event_out = nn.Sequential(
                    nn.Conv3d(self.d, self.d, (3, 3, 3), 1, (1, 1, 1)),
                    nn.Tanh(),
                )
                #self.deconv1_event = nn.ConvTranspose3d(self.d, self.d, (1, 2, 2), (1, 2, 2),)
                #self.deconv2_event = nn.ConvTranspose3d(self.d, self.d, (1, 2, 2), (1, 2, 2),)
               
            def forward(self, event):
                e0 = self.conv_event(event)
                e1, ind1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2), return_indices=True)(e0) # 32, /2 
                e2 = self.conv_events_down[0](e1) # 64, /2
                e3, ind3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2), return_indices=True)(e2) # 64, /4 
                e4 = self.conv_events_down[1](e3) # 128, /4
                e, ind5 = nn.MaxPool3d((1, 2, 2), (1, 2, 2), return_indices=True)(e4) # 128, /8
                                
                e = self.self_atten(e)
                e_save = e.detach() 
                eu1 = nn.MaxUnpool3d((1, 2, 2), (1, 2, 2))(e, ind5, output_size=e4.shape) + e4 # 128, /4
                eu2 = self.conv_events_up[0](eu1) # 64, /4
                eu3 = nn.MaxUnpool3d((1, 2, 2), (1, 2, 2))(eu2, ind3, output_size=e2.shape) + e2 # 64, /2
                eu4 = self.conv_events_up[1](eu3) # 32, /2
                eu5 = nn.MaxUnpool3d((1, 2, 2), (1, 2, 2))(eu4, ind1, output_size=e0.shape) + e0 # 32, /1
                
                e_out = self.conv_event_out(eu5)
                return e_out, e_save

        class ResBlock(nn.Module):
            def __init__(self, din, n_block=5):
                super().__init__()
                self.n_block = n_block
                self.res_convs = nn.ModuleList()
                self.prelus = nn.ModuleList()
                for i in range(self.n_block):
                    self.res_convs.append(
                        nn.Sequential(
                            nn.Conv3d(din, din, (3, 3, 3), 1, 1),
                            nn.PReLU(),
                            nn.Conv3d(din, din, (3, 3, 3), 1, 1),
                            nn.PReLU()
                        )
                    )
                    self.prelus.append(nn.PReLU())

            def forward(self, x):
                for i in range(self.n_block):
                    resx = x 
                    x = self.prelus[i](self.res_convs[i](x) + resx) 
                return x
                    
        class ConvLSTM(nn.Module):
            def __init__(self, din):
                super().__init__()
                self.conv_i = nn.Sequential(
                    nn.Conv3d(din + din, din, (3, 3, 3), 1, 1),
                    nn.Sigmoid()
                    )
                self.conv_f = nn.Sequential(
                    nn.Conv3d(din + din, din, (3, 3, 3), 1, 1),
                    nn.Sigmoid()
                    )
                self.conv_g = nn.Sequential(
                    nn.Conv3d(din + din, din, (3, 3, 3), 1, 1),
                    nn.Tanh()
                    )
                self.conv_o = nn.Sequential(
                    nn.Conv3d(din + din, din, (3, 3, 3), 1, 1),
                    nn.Sigmoid()
                    )
            def forward(self, x, h, c):
                x = torch.cat((x, h), 1) 
                i = self.conv_i(x) 
                f = self.conv_f(x) 
                g = self.conv_g(x) 
                o = self.conv_o(x) 
                c = f * c + i * g 
                h = o * torch.tanh(c)
                return h, c

        
        self.din = opt_dim
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv3d(6, self.din, (3, 3, 3), 1, 1),
            nn.PReLU()
            )
        self.conv = nn.Sequential(
            nn.Conv3d(self.din, 3, (3, 3, 3), 1, 1),
            )
        self.convLSTM = ConvLSTM(self.din)
        self.resBlock = ResBlock(self.din)
        self.atten = atten(self.din)
        #self.lightFlowNet = LightFlowNet.LightFlowNet()
        

    def forward_patch(self, input, event, iteration, patch_modes, level, x=None, h=None, c=None, e=None):
        '''
        input: n x c x s x w x h
        output: n x c x s x w x h
        '''
        patch_mode = patch_modes[level]
        patches_input = self.patch_divide(input, patch_mode)
        patches_event = self.patch_divide(event, patch_mode)
#        warpped_l = self.patch_divide(self.warpped_l, patch_mode)
#        warpped_r = self.patch_divide(self.warpped_r, patch_mode)
        batch_size, _, seq_size, row, col = patches_input.size()
        if x is None and iteration != 0:
            x = patches_input
        if h is None and iteration != 0:
            h = Variable(torch.zeros(batch_size, self.din, seq_size, row, col))
            c = Variable(torch.zeros(batch_size, self.din, seq_size, row, col))
            if self.use_GPU:
                h = h.cuda()
                c = c.cuda()
        e_save = None
        for itr in range(iteration):
            #print(patches_input.shape, x.shape, level)
            x = torch.cat((patches_input, x,), 1) 
#            x = torch.cat((patches_input, x, warpped_l, warpped_r), 1)
            x = self.conv0(x) 

            e, e_save = self.atten(patches_event, )
            x = x + x * (e[:, :, :-1, :, :] + e[:, :, 1:, :, :])

            h, c = self.convLSTM(x, h, c)
            x = h # 571085

            if itr + 1 == iteration:
                next_patch_mode = patch_modes[min(len(patch_modes)-1, level+1)]
                x, h, c, e = self.patch_merge(x, h, c, e, patch_mode)
                x = self.patch_divide(x, next_patch_mode)
                h = self.patch_divide(h, next_patch_mode)
                c = self.patch_divide(c, next_patch_mode)
                e = self.patch_divide(e, next_patch_mode) 
                patches_input = self.patch_divide(input, next_patch_mode)
                
            x = self.resBlock(x)
            x = self.conv(x) 

            x = x + patches_input 
        torch.cuda.empty_cache()
        return x, h, c, e, e_save

    def patch_divide(self, input, patch_mode):
        if input is None:
            return None
        batch_size, _, seq_size, row, col = input.size()
        rows = row // patch_mode[0]
        cols = col // patch_mode[1]
        N = batch_size * patch_mode[0] * patch_mode[1]
        patches = input.reshape(batch_size,-1,seq_size,patch_mode[0],rows,patch_mode[1],cols).permute(0,3,5,1,2,4,6).reshape(N,-1,seq_size,rows,cols)
        return patches

    def patch_merge(self, x, h, c, e, patch_mode):
        N, _, seq_size, rows, cols = x.size()
        row = rows * patch_mode[0]
        col = cols * patch_mode[1]
        batch_size = N // patch_mode[0] // patch_mode[1]
        x = x.reshape(batch_size, patch_mode[0], patch_mode[1], -1, seq_size, rows, cols).permute(0,3,4,1,5,2,6).reshape(batch_size, -1, seq_size, row, col)
        h = h.reshape(batch_size, patch_mode[0], patch_mode[1], -1, seq_size, rows, cols).permute(0,3,4,1,5,2,6).reshape(batch_size, -1, seq_size, row, col)
        c = c.reshape(batch_size, patch_mode[0], patch_mode[1], -1, seq_size, rows, cols).permute(0,3,4,1,5,2,6).reshape(batch_size, -1, seq_size, row, col)
        if e is not None:
            e = e.reshape(batch_size, patch_mode[0], patch_mode[1], -1, seq_size+1, rows, cols).permute(0,3,4,1,5,2,6).reshape(batch_size, -1, seq_size+1, row, col)
        return x, h, c, e
    
    def get_warp(self, input, if_left=True):
        warpped = torch.zeros_like(input)
        seq_size = input.shape[2]
        if if_left:
            for i in range(1, seq_size):
                flow = LightFlowNet.estimate(input[:,:,i], input[:,:,i-1])
                warpped[:,:,i] = LightFlowNet.backwarp(input[:,:,i-1], flow)
        else:
            for i in range(0, seq_size-1):
                flow = LightFlowNet.estimate(input[:,:,i], input[:,:,i+1])
                warpped[:,:,i] = LightFlowNet.backwarp(input[:,:,i+1], flow)
        return warpped

    def forward(self, input, event,):
        patch_modes = [(2, 2), (1, 2), (1, 1)]
#        self.warpped_l = self.get_warp(input, True) 
#        self.warpped_r = self.get_warp(input, False)
        x, h, c, e = None, None, None, None
        if_pad_rows=False; if_pad_cols=False
        N, chn, seq_size, rows, cols = input.size()
        if rows % 2 != 0:
            if_pad_rows=True
            input = torch.cat((input, torch.zeros(N, chn, seq_size, 1, cols).to(input)), 3)
            event = torch.cat((event, torch.zeros(N, chn-1, seq_size+1, 1, cols).to(event)), 3)
            rows+=1
        if cols % 2 != 0:
            if_pad_cols=True
            input = torch.cat((input, torch.zeros(N, chn, seq_size, rows, 1).to(input)), 4)
            event = torch.cat((event, torch.zeros(N, chn-1, seq_size+1, rows, 1).to(event)), 4)
            cols+=1
        for level, iteration in enumerate(self.iteration):
            x, h, c, e, e_save = self.forward_patch(input, event, iteration, patch_modes, level, x, h, c, e)
        if if_pad_rows:
            x = x[:,:,:,:-1,:]
        if if_pad_cols:
            x = x[:,:,:,:,:-1]
        return x#, e_save

