import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, cv2
import h5py 
from pathlib import Path
import random
from PIL import Image

working_dir = "~/proj1/evnet/data"
event_dir = "~/proj1/evnet/data/event/Dataset_Training_Synthetic"
image_dir = "~/proj1/evnet/data/image/Dataset_Training_Synthetic"

train_subdirs = ["t"+str(i) for i in range(1,9)]

def normalize(data):
    '''
    Assume data is within [0,255]
    '''
    return data / 255.

def Im2Patch(img, win, stride=1):
    '''
    input: c x w x h
    output: c x win x win x n_patch
    '''
    k = 0
    endc, endw, endh = img.shape
    w_steps = list(range(0, endw - win + 1, stride)) + [endw - stride]
    h_steps = list(range(0, endh - win + 1, stride)) + [endh - stride]
    TotalPatNum = len(w_steps) * len(h_steps) # n_patch
    Y = np.zeros([endc, win, win, TotalPatNum], np.float32)

    for i in w_steps:
        for j in h_steps:
            Y[:, :, :, k] = img[:, i:i+win, j:j+win]
            k = k + 1
    return Y


def preprocess(patch_size, stride, seq_crop_len=7, seq_crop_stride=7, data_path=working_dir):
    print('process training data')
    
    save_target_path = os.path.join(data_path, 'train_target_evnet_s7.h5')
    save_input_path = os.path.join(data_path, 'train_input_evnet_s7.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    N_raintypes = 3
    for ii in train_subdirs: # there are 8 videos in NTURain training
        for id_, img in enumerate(sorted([i for i in (image_dir / Path(ii+"_GT")).glob("*.jpg")])):
            gt = str(img)
            target = cv2.imread(gt)
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride) # c x win x win x n_patch
            
            input_patches_raintypes = np.empty((N_raintypes,)+target_patches.shape, target_patches.dtype)
            for jj in range(N_raintypes): # 3 synthetic rainy ones for each rain-free video
                rain = str(img.parents[1] / Path(ii+"_Rain_0"+str(jj+1)) / img.name)
                input_img = cv2.imread(rain)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
                print("target file: %s # samples: %d" % (rain, target_patches.shape[3]))
                input_patches_raintypes[jj, ] = input_patches # 3 x c x win x win x n_patch

            if id_ == 0:
                seq_target_patches = target_patches[np.newaxis, ] # 1 x c x win x win x n_patch
                seq_input_patches_raintypes = input_patches_raintypes[np.newaxis,] # 1 x 3 x c x win x win x n_patch
            else:
                seq_target_patches = np.concatenate((seq_target_patches, target_patches[np.newaxis, ]), axis=0) # s x c x win x win x n_patch
                seq_input_patches_raintypes = np.concatenate((seq_input_patches_raintypes, input_patches_raintypes[np.newaxis, ]), axis=0) # s x 3 x c x win x win x n_patch
        for n in range(seq_target_patches.shape[-1]): 
            for jj in range(N_raintypes):
                for ss in list(range(0, seq_target_patches.shape[0] - seq_crop_len + 1, seq_crop_stride)) + [seq_target_patches.shape[0] - seq_crop_len]:
                    target_data = seq_target_patches[ss:ss+seq_crop_len, :, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = seq_input_patches_raintypes[ss:ss+seq_crop_len, jj, :, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)
                    train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


def event_preprocess(patch_size, stride, seq_crop_len=7, seq_crop_stride=7, data_path=working_dir):
    print('process training data')
    
    save_target_path = os.path.join(data_path, 'train_target_evnet_s7_{}.h5'.format(patch_size))
    save_input_path = os.path.join(data_path, 'train_input_evnet_s7_{}.h5'.format(patch_size))
    save_event_path = os.path.join(data_path, 'train_event_evnet_s7_{}.h5'.format(patch_size))

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    event_h5f = h5py.File(save_event_path, 'w')

    train_num = 0
    N_raintypes = 3
    for ii in train_subdirs: # there are 8 videos in NTURain training
        files = sorted([i for i in (image_dir / Path(ii+"_GT")).glob("*.jpg")])
        for id_, img in enumerate(files):
            target = cv2.imread(str(img))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride) # c x win x win x n_patch
            
            input_patches_raintypes = np.empty((N_raintypes,)+target_patches.shape, target_patches.dtype) # 3 x c x win x winc x n_patch
            event_patches_raintypes = np.empty((N_raintypes,)+target_patches.shape, target_patches.dtype) # 3 x c x win x winc x n_patch
            
            for jj in range(N_raintypes): # 3 synthetic rainy ones for each rain-free video
                rain = str(img.parents[1] / Path(ii+"_Rain_0"+str(jj+1)) / img.name)
                input_img = cv2.imread(rain)
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)
                print("target file: %s # samples: %d" % (rain, target_patches.shape[3]))
                input_patches_raintypes[jj, ] = input_patches # 3 x c x win x win x n_patch
                if id_+1 != len(files):
                    event = os.path.join(event_dir, (ii+"_Rain_0"+str(jj+1)), str(img.stem)+".npy")
                    print(event)
                    event_img = np.load(event)
                    
                    event_img = np.float32(event_img)
                    event_patches = Im2Patch(event_img.transpose(2, 0, 1), win=patch_size, stride=stride)
                    event_patches_raintypes[jj, ] = event_patches # 3 x c x win x win x n_patch
            if id_ == 0:
                seq_target_patches = target_patches[np.newaxis, ] # 1 x c x win x win x n_patch
                seq_input_patches_raintypes = input_patches_raintypes[np.newaxis,] # 1 x 3 x c x win x win x n_patch
                seq_event_patches_raintypes = event_patches_raintypes[np.newaxis,] # 1 x 3 x c x win x win x n_patch
                seq_event_patches_raintypes = np.concatenate((np.zeros(seq_event_patches_raintypes.shape, np.float32), seq_event_patches_raintypes), 0)
            else:
                seq_target_patches = np.concatenate((seq_target_patches, target_patches[np.newaxis, ]), axis=0) # s x c x win x win x n_patch
                seq_input_patches_raintypes = np.concatenate((seq_input_patches_raintypes, input_patches_raintypes[np.newaxis, ]), axis=0) # s x 3 x c x win x win x n_patch
                if id_+1 != len(files):
                    seq_event_patches_raintypes = np.concatenate((seq_event_patches_raintypes, event_patches_raintypes[np.newaxis, ]), axis=0) # s x 3 x c x win x win x n_patch
                else:
                    seq_event_patches_raintypes = np.concatenate((seq_event_patches_raintypes, np.zeros((1,) + seq_event_patches_raintypes.shape[1:], np.float32)), axis=0)
        print((seq_event_patches_raintypes[:,:,1,:,:,:]!=0).any())
        for n in range(seq_target_patches.shape[-1]): 
            for jj in range(N_raintypes):
                for ss in list(range(0, seq_target_patches.shape[0] - seq_crop_len + 1, seq_crop_stride)) + [seq_target_patches.shape[0] - seq_crop_len]:
                    target_data = seq_target_patches[ss:ss+seq_crop_len, :, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = seq_input_patches_raintypes[ss:ss+seq_crop_len, jj, :, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    event_data = seq_event_patches_raintypes[ss:ss+seq_crop_len+1, jj, :, :, :, n].copy()
                    event_h5f.create_dataset(str(train_num), data=event_data)
                    train_num += 1
                    if np.isnan(input_data).any():
                        print("\n\n\ninput_data has nan at",n, jj, ss)
                    if np.isnan(event_data).any():
                        print("\n\n\nevent_data has nan at",n, jj, ss)
                    if np.isnan(target_data).any():
                        print("\n\n\ntarget_data has nan at",n, jj, ss)

    target_h5f.close()
    input_h5f.close()
    event_h5f.close()
    print('training set, # samples %d\n' % train_num)

def event_preprocess_RainVIDSS(patch_size, stride, seq_crop_len=7, seq_crop_stride=7, data_path=working_dir, if_sb=False, image_dir="~/proj1/evnet/data/image/dataset_RainVIDSS/", event_dir="~/proj1/evnet/data/event/RainVIDSS/train/rainy"):
    print('process training data')
    suffex_sb = "_sb" if if_sb else "" 
    save_target_path = os.path.join(data_path, 'train_target_RainVIDSS_s7_{}{}.h5'.format(patch_size, suffex_sb))
    save_input_path = os.path.join(data_path, 'train_input_RainVIDSS_s7_{}{}.h5'.format(patch_size, suffex_sb))
    save_event_path = os.path.join(data_path, 'train_event_RainVIDSS_s7_{}{}.h5'.format(patch_size, suffex_sb))

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')
    event_h5f = h5py.File(save_event_path, 'w')

    train_num = 0
    gt_path = os.path.join(image_dir, "train", "gt")
    img_path = os.path.join(image_dir, "train", "rainy")
    evt_path = os.path.join(event_dir)
    train_subdirs = os.listdir(gt_path)
    print(train_subdirs)
    for ii in train_subdirs: # there are 208 videos in RainVIDSS training
        files = sorted([i for i in (Path(gt_path) / Path(ii)).glob("*.jpg")])
        events = sorted([i for i in (Path(evt_path) / Path(ii)).glob("*.npy")])
        for id_, img in enumerate(files):
            print("Reading {}, {}".format(img, os.path.join(img_path, ii, img.name)))
            target = cv2.imread(str(img))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])
            target_img = target
            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride) # c x win x win x n_patch
            
            image = cv2.imread(os.path.join(img_path, ii, img.name))
            b, g, r = cv2.split(image)
            image = cv2.merge([r, g, b])
            image_img = image
            image_img = np.float32(normalize(image_img))
            image_patches = Im2Patch(image_img.transpose(2,0,1), win=patch_size, stride=stride) # c x win x win x n_patch
            
            if id_ + 1 < len(files):
                event_img = np.load(events[id_])
                event_img = np.float32(event_img)
                event_patches = Im2Patch(event_img.transpose(2,0,1), win=patch_size, stride=stride) # c x win x win x n_patch
            
            if id_ == 0:
                seq_target_patches = target_patches[np.newaxis, ] # 1 x c x win x win x n_patch
                seq_input_patches = image_patches[np.newaxis,] # 1 x c x win x win x n_patch
                seq_event_patches = event_patches[np.newaxis,] # 1 x c x win x win x n_patch
                seq_event_pad = np.zeros_like(seq_event_patches)
                seq_event_patches = np.concatenate((seq_event_pad.copy(), seq_event_patches), 0)
            else:
                seq_target_patches = np.concatenate((seq_target_patches, target_patches[np.newaxis, ]), axis=0) # s x c x win x win x n_patch
                seq_input_patches = np.concatenate((seq_input_patches, image_patches[np.newaxis, ]), axis=0) # s x c x win x win x n_patch
                if id_+1 != len(files):
                    seq_event_patches = np.concatenate((seq_event_patches, event_patches[np.newaxis, ]), axis=0) # s x c x win x win x n_patch
                else:
                    seq_event_patches = np.concatenate((seq_event_patches, seq_event_pad.copy()), axis=0)
        print((seq_event_patches[:,1,:,:,:]!=0).any())
        print(seq_target_patches.shape, seq_input_patches.shape, seq_event_patches.shape)
        for n in range(seq_target_patches.shape[-1]): 
            for ss in list(range(0, seq_target_patches.shape[0] - seq_crop_len + 1, seq_crop_stride)) + [seq_target_patches.shape[0] - seq_crop_len]:
                target_data = seq_target_patches[ss:ss+seq_crop_len, :, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = seq_input_patches[ss:ss+seq_crop_len, :, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)

                event_data = seq_event_patches[ss:ss+seq_crop_len+1, :, :, :, n].copy()
                event_h5f.create_dataset(str(train_num), data=event_data)
                train_num += 1
                if np.isnan(input_data).any():
                    print("\n\n\ninput_data has nan at",n, ss)
                if np.isnan(event_data).any():
                    print("\n\n\nevent_data has nan at",n, ss)
                if np.isnan(target_data).any():
                    print("\n\n\ntarget_data has nan at",n, ss)
                if if_sb == True:
                    if train_num > 50:
                        target_h5f.close()
                        input_h5f.close()
                        event_h5f.close()
                        return


    target_h5f.close()
    input_h5f.close()
    event_h5f.close()
    print('training set, # samples %d\n' % train_num)

# def collate_fn(batch):
#     seq_lens = [item[0].shape[0] for item in batch]
#     seq_input = []
#     seq_gt = []
#     for item in batch:
#         seq_input.append(item[0])
#         seq_gt.append(item[1])
#     seq_input_pad = pad_sequence(seq_input, batch_first=True)
#     seq_gt_pad = pad_sequence(seq_gt, batch_first=True)
#     print(seq_lens)
#     return {
#         'rain': seq_input_pad.transpose(2,1),
#         'gt': seq_gt_pad.transpose(2,1),
#         'lens': seq_lens,
#     }

class NTURainData(Dataset):
    def __init__(self, data_path=working_dir, if_event=False, data_name='NTURain'):
        super().__init__()
        self.data_path = data_path
        if data_name == 'NTURain':
            self.data_name = "evnet"
        else:   
            self.data_name = "RainVIDSS"
        target_path = os.path.join(self.data_path, 'train_target_{}_s7_128_sb.h5'.format(self.data_name))
        
        target_h5f = h5py.File(target_path, 'r')
        
        self.keys = list(target_h5f.keys())
        # random.shuffle(self.keys)
        target_h5f.close()
        
        # input_path = os.path.join(self.data_path, 'train_input_evnet_s5_100.h5')
        # input_h5f = h5py.File(input_path, 'r')
        # input_h5f.close()

        self.if_event = if_event
        

    def __len__(self, ):
        return len(self.keys)

    def __getitem__(self, index):
        
        target_path = os.path.join(self.data_path, 'train_target_{}_s7_128_sb.h5'.format(self.data_name))
        input_path = os.path.join(self.data_path, 'train_input_{}_s7_128_sb.h5'.format(self.data_name))

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key]).transpose(1,0,2,3) # c x s x win x win
        input = np.array(input_h5f[key]).transpose(1,0,2,3)
        # print(target.shape, input.shape)
        target_h5f.close()
        input_h5f.close()

        if self.if_event:
            event_path = os.path.join(self.data_path, 'train_event_{}_s7_128_sb.h5'.format(self.data_name))
            event_h5f = h5py.File(event_path, 'r')
            event = np.array(event_h5f[key]).transpose(1,0,2,3)
            assert (event[1]==0).all()
            event = event[[0, 2]]
            event_h5f.close()
            if np.isnan(event).any():
                print("event has nan")
            return torch.tensor(input).to(torch.float32), torch.tensor(target).to(torch.float32), torch.tensor(event).to(torch.float32)
        # input_min = input.reshape(3,-1).min(1)[:,np.newaxis,np.newaxis,np.newaxis]
        # input_max = input.reshape(3,-1).max(1)[:,np.newaxis,np.newaxis,np.newaxis]
        # input_pro = (( (input - input_min) / (input_max-input_min) ) - 0.5) / 0.5
        # event_min = event.reshape(3,-1).min(1)[:,np.newaxis,np.newaxis,np.newaxis]
        # event_max = event.reshape(3,-1).max(1)[:,np.newaxis,np.newaxis,np.newaxis]
        # event_pro = (( (event - event_min) / (event_max-event_min) ) - 0.5) / 0.5
        # target_min = target.reshape(3,-1).min(1)[:,np.newaxis,np.newaxis,np.newaxis]
        # target_max = target.reshape(3,-1).max(1)[:,np.newaxis,np.newaxis,np.newaxis]
        # target_pro = (( (target - target_min) / (target_max-target_min) ) - 0.5) / 0.5
        if np.isnan(input).any():
            print("input has nan")
        if np.isnan(target).any():
            print("target has nan")
        return torch.tensor(input).to(torch.float32), torch.tensor(target).to(torch.float32)
