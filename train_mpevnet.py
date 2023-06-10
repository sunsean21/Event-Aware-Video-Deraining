import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from SSIM import seq_SSIM
from mpevnet import *

torch.cuda.empty_cache()
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description="MPEVNet_train")
parser.add_argument("--preprocess", action="store_true", default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/mpevNet/s7_i6_100_ev_sa2", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="dataset/train/RainVIDSS",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=str, default='0,0,6', help='number of recursive stages')
parser.add_argument("--sa_loss_epoch", type=int, default=2, help='when (in which epoch) to use self-alignment loss')
parser.add_argument("--sb", type=int, default=0, help='small batch for quick test or not')
parser.add_argument("--data_name", type=str, default="NTURain", help='data name')
parser.add_argument("--dim_feature", type=int, default=16, help='dimension of hidden features')
opt = parser.parse_args()
print(opt)
if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_train = dataLoader_evnet.NTURainData(data_path=opt.data_path, if_event=True, data_name=opt.data_name)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    model = MPEVNet(recurrent_iter=[int(i) for i in opt.recurrent_iter.split(',')], use_GPU=opt.use_gpu, opt_dim=opt.dim_feature)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = seq_SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=1, threshold=1e-3)  # learning rates
    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
        optimizer.load_state_dict(torch.load(os.path.join(opt.save_path, 'param_epoch%d.pth' % initial_epoch)))
    # start training
    step = initial_epoch * dataset_train.__len__() / opt.batch_size
    prev_psnr = 0
    if_sa_loss = True
    for epoch in range(initial_epoch, opt.epochs):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        epoch_loss = 0
        epoch_psnr = 0
        ## epoch training start
        for i, (input_train, target_train, event_train) in enumerate(loader_train, 0):
            if opt.sb == 1 and i > 10:
                break
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            # input_train, target_train = Variable(input_train), Variable(target_train)
            if opt.use_gpu:
                input_train, target_train, event_train = input_train.cuda(), target_train.cuda(), event_train.cuda()
            input_train.requires_grad_()
            event_train.requires_grad_()
            # print(input_train.requires_grad, event_train.requires_grad, target_train.requires_grad)
            out_train = model(input_train, event_train)
            # print(out_train.requires_grad)
            if (epoch >= opt.sa_loss_epoch and prev_psnr >= 34):
                if_sa_loss = True
                prev_psnr = 0
            pixel_metric = criterion(target_train, out_train, if_sa_loss)
            loss = -pixel_metric
            
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2., norm_type=2)
            optimizer.step()


            torch.cuda.empty_cache()
            # training curve
            model.eval()
            with torch.no_grad():
                out_train = out_train.detach()#model(input_train, event_train)
                out_train = torch.clamp(out_train, 0., 1.)
                psnr_train = batch_PSNR(out_train, target_train, 1.)
                epoch_psnr += psnr_train
                epoch_loss += pixel_metric.item()
                print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                      (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                    writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)
                step += 1
            #if i == 10:
            #    break
        ## epoch training end
        print("The average loss for epoch {} is {}".format(epoch, epoch_loss / len(loader_train)))
        print("The average psnr for epoch {} is {}".format(epoch, epoch_psnr / len(loader_train)))
        scheduler.step(epoch_loss / len(loader_train))   
        prev_psnr = epoch_psnr / len(loader_train)
        # log the images
        model.eval()
        with torch.no_grad():
            out_train = out_train.detach()#model(input_train, event_train)
            out_train = torch.clamp(out_train, 0., 1.)
            print(target_train.data.shape)
            _, dim_c, dim_s, dim_w, dim_h = target_train.data.shape
            im_target = utils.make_grid(target_train.data.transpose(2,1).reshape(-1, dim_c, dim_w, dim_h), nrow=dim_s, normalize=True, scale_each=True)
            im_input = utils.make_grid(input_train.data.transpose(2,1).reshape(-1, dim_c, dim_w, dim_h), nrow=dim_s, normalize=True, scale_each=True)
            im_event = utils.make_grid(event_train.data.transpose(2,1).reshape(-1, 2, dim_w, dim_h), nrow=dim_s+1, normalize=True, scale_each=True)
            im_derain = utils.make_grid(out_train.data.transpose(2,1).reshape(-1, dim_c, dim_w, dim_h), nrow=dim_s, normalize=True, scale_each=True)
            writer.add_image('clean image', im_target, epoch+1)
            writer.add_image('rainy image', im_input, epoch+1)
            writer.add_image('event image', im_event, epoch+1)
            writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        torch.save(optimizer.state_dict(), os.path.join(opt.save_path, 'param_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))
            torch.save(optimizer.state_dict(), os.path.join(opt.save_path, 'param_epoch%d.pth' % (epoch+1)))
            writer.add_scalar('Epoch loss', epoch_loss / len(loader_train), epoch)
            writer.add_scalar('Epoch psnr', epoch_psnr / len(loader_train), epoch)


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        elif opt.data_path.find('NTURain') != -1:
            dataLoader_evnet.event_preprocess(patch_size=128, stride=128, seq_crop_len=7, seq_crop_stride=7)
        elif opt.data_path.find('RainVIDSS') != -1:
            dataLoader_evnet.event_preprocess_RainVIDSS(patch_size=128, stride=128, seq_crop_len=7, seq_crop_stride=7) 
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
