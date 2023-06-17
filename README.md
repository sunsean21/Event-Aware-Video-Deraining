
# Event-Aware Video Deraining via Multi-Patch Progressive Learning

[[Paper]](https://ieeexplore.ieee.org/document/10122854) 

# Requirment

Python 3.8 (previous version should be fine as well)
Numpy
PyTorch 1.9 (previous version should be fine as well)
torchvision 0.10.1

# Data preparation

## NTURain

1. Download the NTURain dataset from https://github.com/hotndy/SPAC-SupplementaryMaterials

2. Extract all clips from .rar files

3. The extracted files should have the following structure:

├── Dataset_Testing_RealRain

    ├── ra1_Rain

    ...

    ├── rb3_Rain  

├── Dataset_Testing_Synthetic

    ├── a1_Rain

    ├── a1_GT

    ...

    ├── b4_Rain

    ├── b4_GT 

├── Dataset_Training_Synthetic

    ├── t1_Rain_01

    ...

    ├── t8_Rain_03 


## RainVID&SS

It is available in https://pan.baidu.com/s/1zuGkbKzQjnliujqDMHl1og?pwd=q0r1#list/path=%2Fdataset_event_derain.

## Event

1. Follow the steps in https://github.com/uzh-rpg/rpg_esim to install the event-camera simulator, ESIM.

2. Simulate event for all NTURain clips. A script in utils/make_event_NTURain.sh is helpful. You can follow the instruction in https://github.com/uzh-rpg/rpg_esim/wiki/Simulating-events-from-a-video as well. 

3. The obtained event streams are in .bag format. They should be extracted and stored in npy or jpg format. A script in utils/event2img.sh is helpful.

4. To follow the silence and anonymity policy, we have not shared our pre-computed event files online.

Or you can download the pre-generated event of NTURain in: https://pan.baidu.com/s/1PxHjHSOAW5Q04rsBbWp-UQ?pwd=2aol (PIN: 2aol)

# Train

1. Modify the confiurations in train_mpevnet.sh

2. Since we borrow the reimplementation of lightflownet3 from https://github.com/lhao0301/pytorch-liteflownet3 and https://github.com/NVIDIA/flownet2-pytorch, you should follow their step of installing correlation_package.

3. run the code 

```
bash train_mpevnet.sh
```

If it is the first time to run the code, you should add "--preprocess" argument to get pre-processed .h5 file.

# Test for NTURain

1. Modify the confiurations in test_mpevnet.sh

2. run the code 

```
bash test_mpevnet.sh
```

# Test for real video

1. Modify the confiurations in test_mpevnet_others.sh

2. run the code 

```
bash test_mpevnet_others.sh
```

# BibTex

```
@ARTICLE{10122854,
  author={Sun, Shangquan and Ren, Wenqi and Li, Jingzhi and Zhang, Kaihao and Liang, Meiyu and Cao, Xiaochun},
  journal={IEEE Transactions on Image Processing}, 
  title={Event-Aware Video Deraining via Multi-Patch Progressive Learning}, 
  year={2023},
  volume={32},
  number={},
  pages={3040-3053},
  doi={10.1109/TIP.2023.3272283}}
```