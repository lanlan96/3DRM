# VoteNet with Relation Module
## Introduction
This repository is the VoteNet equipped with relation module code implementation on ScanNet dataset and sunrgbd dataset. The main parts of the code are based on [VoteNet](https://github.com/facebookresearch/votenet).

## Installation
Please follow the [Installation](https://github.com/facebookresearch/votenet#installation) and [Data preparation](https://github.com/facebookresearch/votenet#data-preparation) structions in VoteNet.

## Train and Test

### Train and test on ScanNet
To train a model on Scannet data, you can simply run (it takes around 4 hours to convergence with one TITAN V GPU):
```
CUDA_VISIBLE_DEVICES=0 python train_with_rn.py --dataset scannet --log_dir log_scannet --num_point 40000
```
To test the trained model with its checkpoint:
```
python eval_with_rn.py --dataset scannet --checkpoint_path log_scannet/checkpoint.tar --dump_dir eval_scannet --num_point 40000 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```

### Train and test on Sunrgbd
To train a new VoteNet model on SUN RGB-D data:
```
CUDA_VISIBLE_DEVICES=0 python train_with_rn.py --dataset sunrgbd --log_dir log_sunrgbd
```
To test the trained model with its checkpoint:
```
python eval_with_rn.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal
```

## Acknowledgemets
This code largely benefits from excellent works [VoteNet](https://github.com/facebookresearch/votenet) and [cgnl-network.pytorch](https://github.com/KaiyuYue/cgnl-network.pytorch) repositories, please also consider cite [VoteNet](https://arxiv.org/pdf/1904.09664.pdf) and [CGNL](https://arxiv.org/pdf/1810.13125.pdf) if you use this code.