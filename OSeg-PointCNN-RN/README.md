# OSeg-PointCNN with Relation Module

## Introduction
This repository is the detection framework OSeg-PointCNN equipped with the relation module code implementation on S3DIS dataset. Note that OSeg-PointCNN is a 3D object detection framework of which performance is relatively lower than the State-of-the-art methods.

## Installation
We implement OSeg-PointCNN-RN with Tensorflow 1.12.0, CUDA9.0 and cudnn7.05, please follow the following command to install the dependencies:
    
    pip install -r requirements.txt

## Data Preparation  
Organize the data and directories like folloing before starting to training and evaluation.

├── data  
│   ├── local_data  
│ │ ├── Area_1  
│ │ ├── Area_2    
│ │ └── ...  
│   ├── split_for_local  
│ │ ├── train_files_for_area_1.txt     
│ │ ├── val_files_for_area_1.txt   
│ │ └── ...  
│ ├── proposal05    
│ ├── data_release     
│ ├── Stanford3dDataset_v1.2_Aligned_Version      
├── models   
│ │ ├── train   
│ │ ├── eval  
├── OSeg-PointCNN-RN      
      
Please follow the following steps to produce required data and files.
* Download [S3DIS](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1) dataset, and follow the over-segmentation method used in [VDRAE](https://github.com/yifeishi/HierarchyLayout/tree/master/prerprocess) to generate proposals. Save them in directory: data/local_data/. We take the node segments as the initial proposals. Organize the generated proposals according to the areas like the examples.
* Design a deep nerual network (e.g PointCNN) to select proposals with good objectness (IoU>0.5).
* Use the python [script](./preprocess_data/cal_proposal_05.py) cal_proposal_05.py to save the context points corresponding to each proposal in data/proposal05. Before starting, remember to down-sample each scene points (e.g open3d.voxel_down_sample) in S3DIS to accelerate the processing speed and change path to your own.
* use the python [script](./preprocess_data/train_test_split.py) to generate the train and val filelist txt files for each area, or you can produce them yourself by combining selected proposals in other five areas together as _train_ txt and proposals in the left area as _val_ txt for each area.


## Train

Run train.sh to start training. We perform a 6-fold cross validation across six areas. If you want to get the mean AP across six areas, you need to train on six areas. Before training, make sure the data is prepared.

    ./train.sh -a $AREA_ID -g $GPU_ID -b $BATCH_SIZE
For example, train on Area_1 with batch_size=8:

    ./train.sh -a 1 -g 0 -b 8
    

## Evaluate
Change the path (_args.load_ckpt_ at line 61 in eval.py) for your trained model. Run eval.sh to start training.
    
    ./eval.sh -a AREA_ID -g $GPU_ID -b $BATCH_SIZE

For example, evaluate on Area_1 with batch_size=8:

    ./eval.sh -a 1 -g 0 -b 8

