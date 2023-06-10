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
parser.add_argument("--logdir", type=str, default="logs/mpevNet/s7_i123_100_RainVIDSS/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="~/proj1/evnet/data/others/", help='path to training data')
parser.add_argument("--event_path", type=str, default="~/proj1/evnet/data/event/", help='path to event data')
parser.add_argument("--save_path", type=str, default="~/proj1/evnet/output/others/mpevNet/out_s7_i123_100_RainVIDSS_e", help='path to save results')

parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="2", help='GPU id')
parser.add_argument("--recurrent_iter", type=str, default='1,2,3', help='number of recursive stages')
parser.add_argument("--seq_len", type=int, default=7, help='length of sequence for test')
parser.add_argument("--model_epoch", type=int, default=22, help='the epoch of model for test')
parser.add_argument("--dim_feature", type=int, default=16, help='dimension of hidden features')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
opt.save_path = opt.save_path + str(opt.model_epoch)

def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = MPEVNet([int(i) for i in opt.recurrent_iter.split(',')], opt.use_GPU, opt_dim=opt.dim_feature)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    if opt.model_epoch == 0:
        model.load_state_dict(torch.load(str(Path(opt.logdir) / 'net_latest.pth')))
    else:
        model.load_state_dict(torch.load(str(Path(opt.logdir) / 'net_epoch{}.pth'.format(opt.model_epoch))))
    model.eval()

    time_test = 0
    count = 0
    psnr_all_y = []
    ssim_all_y = []
#    rain_types = ["ra1", "ra2", "ra3", "ra4", "rb1", "rb2", "rb3"] 
    seq_names = os.listdir(opt.data_path)
    for seq_name in seq_names:
        #if "ori" in seq_name or "yard" in seq_name or 'balcony' in seq_name or 'garden' in seq_name or 'ground' in seq_name or 'tree' in seq_name or '_Rain' in seq_name:
        #   continue
        test_subdir = seq_name
        print("testing "+ test_subdir)
        os.makedirs(Path(opt.save_path) / Path(test_subdir), exist_ok=True)
       # os.makedirs(Path(opt.save_path) / Path(test_subdir) / Path("event"), exist_ok=True)
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

            if jj + 1 != len(files):
                event_path = os.path.join(opt.event_path, seq_name, "%0.5d" % (int(img_path.stem))+".npy")
                event_img = np.load(event_path)
                event_img = torch.Tensor(np.float32(event_img).transpose(2, 0, 1)[np.newaxis,]) # 1 x c x w x h
            if seq_name in ['rb'+str(id_)+'_Rain' for id_ in range(1,4)]:
                event_img = event_img[:,:,:360,:]

            img_paths.append(img_path)
            
            if update_seq_img:
                seq_img = y.unsqueeze(2) # 1 x c x s(1) x h x w
                print(seq_img.shape)
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
                print(seq_event.shape, event_img.shape, )
                seq_event = torch.cat((seq_event, event_img.unsqueeze(2)), 2)
            else:
                _, c, _, h, w = seq_event.shape
                seq_event = torch.cat((seq_event, torch.zeros((1,c,1,h,w)).to(seq_event.dtype).to(seq_event.device)),2)    


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
                    out, _ = model(seq_img, seq_event[:,[0,2]]) # 1 x c x s x h x w
                    
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
                    b, g, r = cv2.split((255.*save_out).astype(np.uint8))
                    save_out = cv2.merge([r, g, b])

                    cv2.imwrite(os.path.join(opt.save_path, test_subdir, img_paths[ss].name), save_out)
#                    cv2.imwrite(os.path.join(opt.save_path, test_subdir, "event", img_paths[ss].name), (evt.squeeze(0)[:,ss,:,:].permute(1,2,0).cpu().numpy()*255.).astype(np.uint8))

                img_paths = []
                
                if seq_id == 0:
                    seq_out = torch.from_numpy(out)
                else:
                    seq_out = torch.cat((seq_out, torch.from_numpy(out)), 0)    
                seq_id += 1
            
        print('Avg. time:', time_test/count)


if __name__ == "__main__":
    main()

# Type:a1_Rain, PSNR:35.13, SSIM:0.9667
# Type:a2_Rain, PSNR:33.04, SSIM:0.9577
# Type:a3_Rain, PSNR:34.32, SSIM:0.9573
# Type:a4_Rain, PSNR:38.73, SSIM:0.9782
# Type:b1_Rain, PSNR:35.38, SSIM:0.9639
# Type:b2_Rain, PSNR:38.24, SSIM:0.9740
# Type:b3_Rain, PSNR:37.80, SSIM:0.9729
# Type:b4_Rain, PSNR:37.14, SSIM:0.9676
# 36.2225, 0.9672875

# Epoch: 23, test: MPSNR:36.09, MSSIM:0.9655

