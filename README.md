
# Event-Aware Video Deraining via Multi-Patch Progressive Learning

[[IEEE Paper]](https://ieeexplore.ieee.org/document/10122854) [Google Drive Paper](https://drive.google.com/file/d/19VseYjSmdTs_tKxJT1JISRJtB-MiIR60/view)

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

```
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

```


## RainVID&SS

It is available in [Baidu Disk](https://pan.baidu.com/s/1zuGkbKzQjnliujqDMHl1og?pwd=q0r1#list/path=%2Fdataset_event_derain) or [Google Drive](https://drive.google.com/drive/folders/1uXlk7WuI1md_vYeXQJBm3ctgmziu3-xO?usp=sharing).

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

# Visual Results

|    Models    |                     NTURain                       |                      RainVIDSS                       |  Real-World|
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| MSCSC | [Google Drive](https://drive.google.com/file/d/1TG1TmY1-1q4ZPuLnBPd8t7F7zpxf0_zH/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1f__8xDHPnXFQa0ObbA0qwmD_-Zqc7btG/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1KTU-fl-ttUt0Jf9L1HHjB5DB-mGR7lZH/view?usp=sharing) |
| SLDNet | [Google Drive](https://drive.google.com/file/d/1D3OpTigvXii8g4p2fycBmI9P9sUtwz5C/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1e3LxGKr0UpYxsB2WnbkjUyIZPtJ2MMvI/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1PzZM05WTcGoVUrM7A736ovMwZo6juz2V/view?usp=sharing) |
| S2VD | [Google Drive](https://drive.google.com/file/d/1k2RLW6WGiiM0tR3Xc8MFkDUOJ6SGeC6V/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1sPvRdkUYH-98iMDV4Rk3fKyTDx20rRIc/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1GmrBPvfN0k619mDP0XYfVmpTkP_jFGqn/view?usp=sharing) |
| MFGAN | [Google Drive](https://drive.google.com/file/d/1sRW2g3KnjlKAd2mXngiATBv1NDzmXbgT/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1fP7bye3D24PzGjsL2O2XBrOEdc8ie9g8/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/1DYT8vlOVLsuspimOI5PDFpQIWXuiuxa2/view?usp=sharing) |
| Ours | [Google Drive](https://drive.google.com/file/d/17sfbWY3c5Xdjaf34JNuMIIRiB8WXHzxi/view?usp=sharing) | [Google Drive](https://drive.google.com/file/d/16-dAVx2z8JXVcAPD1ZirnyWi47gEG-QX/view?usp=sharing) | [Google Drive]() |

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
