#!/usr/bin/env bash

stage=
gpu=0
batchsize=8
debug="False"
remote="False"
area=1
path="../models/test/"

usage() { echo "train/val 3d detection with -s stage -g gpu -b batchsize -d debug -r remote -a area -p path options"; }

while getopts s:g:b:d:r:a:p:h opt; do
  case $opt in
  s)
    stage=$(($OPTARG))
    ;;
  g)
    gpu=${OPTARG}
    ;;
  b)
    batchsize=$(($OPTARG))
    ;;
  d)
    debug=$OPTARG
    ;;
  r)
    remote=$OPTARG
    ;;
  a)
    area=$(($OPTARG))
    ;;
  a)
    path=$(($OPTARG))
    ;;
  h)
    usage; exit;;
  esac
done

echo "stage:" $stage
echo "gpu:" $gpu
echo "batchsize:" $batchsize
echo "debug:" $debug
echo "remote:" $remote
echo "area:" $area


python train.py -db $debug -r $remote --log "log.txt" -m "pointcnn_feature" -s $path  --setting "s3dis_x3_l4_rn" --batch_size $batchsize --area $area --gpu $gpu

