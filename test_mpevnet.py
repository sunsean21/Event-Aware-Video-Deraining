import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from mpevnet import *
import time 
from pathlib import Path
from skimage import img_as_float32, img_as_ubyte

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/mpevNet/s7_i123_100_sa_resume/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="~/proj1/evnet/data/image/Dataset_Testing_Synthetic/", help='path to training data')
parser.add_argument("--event_path", type=str, default="~/proj1/evnet/data/event/Dataset_Testing_Synthetic/", help='path to event data')
parser.add_argument("--save_path", type=str, default="~/proj1/evnet/output/mpevNet/out_s7_i123_100_sa_resume_e", help='path to save results')

parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=str, default='1,2,3', help='number of recursive stages')
parser.add_argument("--seq_len", type=int, default=7, help='length of sequence for test')
parser.add_argument("--model_epoch", type=int, default=20, help='the epoch of model for test')
parser.add_argument("--dim_feature", type=int, default=16, help='dimension of hidden features')


opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
opt.save_path = opt.save_path + str(opt.model_epoch)

def main():

#    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = MPEVNet([int(i) for i in opt.recurrent_iter.split(',')], opt.use_GPU, opt_dim=opt.dim_feature)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    if opt.model_epoch == 0:
        model.load_state_dict(torch.load(Path(opt.logdir) / 'net_latest.pth'))
    else:
        model.load_state_dict(torch.load(Path(opt.logdir) / 'net_epoch{}.pth'.format(opt.model_epoch)))
    model.eval()

    time_test = 0
    count = 0
    psnr_all_y = []
    ssim_all_y = []
    rain_types = ["a"+str(ii) if i < 4 else "b"+str(ii) for i, ii in enumerate(list(range(1,5))*2)] 
    for catagory in rain_types:
        test_subdir = catagory + "_Rain"
        print("testing "+ test_subdir)
        os.makedirs(Path(opt.save_path) / Path(test_subdir), exist_ok=True)
        os.makedirs(Path(opt.save_path) / Path(test_subdir) / Path("event"), exist_ok=True)
        files = [i for i in sorted((Path(opt.data_path) / Path(test_subdir)).glob("*.jpg"))]
        update_seq_img = True
        img_paths = []
        seq_id = 0
        seq_event = None
        for jj, img_path in enumerate(files):

            # input image
            y = cv2.imread(str(img_path))
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y)) # 1 x c x w x h

            if opt.use_GPU:
                y = y.cuda()

            im_gt = img_as_float32(cv2.imread(os.path.join(opt.data_path, catagory + "_GT", img_path.name), flags=cv2.IMREAD_COLOR)[:, :, ::-1]).transpose([2,0,1])
            if jj + 1 != len(files):
                event_path = os.path.join(opt.event_path, catagory + "_Rain", img_path.stem+".npy")
                event_img = np.load(event_path)
                event_img = torch.Tensor(np.float32(event_img).transpose(2, 0, 1)[np.newaxis,]) # 1 x c x w x h

            img_paths.append(img_path)
            
            if update_seq_img:
                seq_img = y.unsqueeze(2) # 1 x c x s(1) x h x w
            
            if update_seq_img:
                seq_img = y.unsqueeze(2) # 1 x c x s(1) x h x w
            
            if update_seq_img:
                seq_img = y.unsqueeze(2) # 1 x c x s(1) x h x w
            
            if update_seq_img:
                seq_img = y.unsqueeze(2) # 1 x c x s(1) x h x w
            
            if update_seq_img:
                seq_img = y.unsqueeze(2) # 1 x c x s(1) x h x w
                if seq_event is None:
                    seq_event = torch.zeros(seq_img.shape).to(seq_img.dtype)
                else:
                    seq_event = seq_event[:,:,-1:,:,:]
                update_seq_img = False
            else:
                temp = y.unsqueeze(2) # 1 x c x 1 x h x w
                seq_img = torch.cat((seq_img, temp), dim=2)  #  1 x c x s x h x w
            if opt.use_GPU:
                seq_event = seq_event.cuda()
                event_img = event_img.cuda()
            if jj + 1 != len(files):
                seq_event = torch.cat((seq_event, event_img.unsqueeze(2)), 2)
            else:
                _, c, _, h, w = seq_event.shape
                seq_event = torch.cat((seq_event, torch.zeros((1,c,1,h,w)).to(seq_event.dtype).to(seq_event.device)),2)    

            if jj == 0:
                seq_gt = torch.from_numpy(im_gt[np.newaxis,])  # 1 x c x s x h x w
            else:
                temp = torch.from_numpy(im_gt[np.newaxis, ]) # 1 x c x h x w
                seq_gt = torch.cat((seq_gt, temp), dim=0)             #  s x c x h x w
            if ( (jj + 1) % opt.seq_len == 0 ) or jj + 1 == len(files): 
                update_seq_img = True
                with torch.no_grad(): #
                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    start_time = time.time()
                    ''' 
                    print(seq_event.shape,)
                    w_mid = seq_img.size(-1) // 4
                    h_mid = seq_img.size(-2) // 4
                    out = torch.zeros_like(seq_img)
                    evt = torch.zeros_like(seq_img)
                    for i_w in  range(4):
                        for i_h in range(4):
                            out[:,:,:,i_h*h_mid:(i_h+1)*h_mid,i_w*w_mid:(i_w+1)*w_mid] = model(seq_img[:,:,:,i_h*h_mid:(i_h+1)*h_mid,i_w*w_mid:(i_w+1)*w_mid], seq_event[:,[0,2],:,i_h*h_mid:(i_h+1)*h_mid,i_w*w_mid:(i_w+1)*w_mid]) # 1 x c x s x h x w//2
                    '''
                    out, e_save0 = model(seq_img, seq_event[:,[0,2]]) # 1 x c x s x h x w
                     
                    out = torch.clamp(out, 0., 1.).squeeze(0).transpose(1,0) # s x c w x h

                    if opt.use_GPU:
                        torch.cuda.synchronize()
                    end_time = time.time()
                    dur_time = end_time - start_time
                    time_test += dur_time

                    print(img_path, ': ', dur_time)
                    count += 1
                if opt.use_GPU:
                    out = out.data.cpu().numpy()   #back to cpu
                else:
                    out = out.data.numpy()

                for ss in range(out.shape[0]):
                    save_out = out[ss,].transpose(1, 2, 0)
                    b, g, r = cv2.split(save_out)
                    save_out = cv2.merge([r, g, b])

                    cv2.imwrite(os.path.join(opt.save_path, test_subdir, img_paths[ss].name), img_as_ubyte(save_out))
                    e_save = e_save0[:,:,ss,].cpu().squeeze(0).data.numpy()
                    e_save = e_save.mean(0, keepdims=True).transpose(1,2,0)
                    #e_save = np.clip(e_save, 0., 1.)
                    #e_save = 1.0/(1+np.exp(-1*e_save))
                    e_save = (e_save - e_save.min()) / (e_save.max() - e_save.min())
                    e_save = (e_save*255.).astype(np.uint8)
                    e_save = cv2.applyColorMap(e_save, cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(opt.save_path, test_subdir, "event", img_paths[ss].name), e_save)

                img_paths = []
                
                if seq_id == 0:
                    seq_out = torch.from_numpy(out)
                else:
                    seq_out = torch.cat((seq_out, torch.from_numpy(out)), 0)    
                seq_id += 1
            
        print('Avg. time:', time_test/count)
        print(seq_out.shape, seq_gt.shape)
        seq_out = seq_out[2:-2,]
        seq_gt = seq_gt[2:-2,]
        psnrm_y = batch_psnr(seq_out, seq_gt, ycbcr=True)
        psnr_all_y.append(psnrm_y)
        ssimm_y = batch_ssim(seq_out, seq_gt, ycbcr=True)
        ssim_all_y.append(ssimm_y)
        print('Type:{:s}, PSNR:{:5.2f}, SSIM:{:6.4f}'.format(test_subdir, psnrm_y, ssimm_y))

    mean_psnr_y = sum(psnr_all_y) / len(rain_types)
    mean_ssim_y = sum(ssim_all_y) / len(rain_types)
    print('MPSNR:{:5.2f}, MSSIM:{:6.4f}'.format(mean_psnr_y, mean_ssim_y))

if __name__ == "__main__":
    main()
