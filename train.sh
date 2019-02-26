#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR='/media/3T_disk/my_datasets/imdb_wiki/wiki'

NETWORK=m1
DATASET=imdb
MODELDIR='./models'
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model-$NETWORK-$DATASET"
LOGFILE="$MODELDIR/log_m1_imdb"
PRETRAINED="./models/model,0"

CUDA_VISIBLE_DEVICES='0' python -u train.py --data-dir $DATA_DIR --prefix $PREFIX --pretrained "$PRETRAINED"  --network $NETWORK  --per-batch-size 128  --lr 0.01 --lr-steps '10000' --ckpt 2 --verbose 500 --multiplier 0.25 > "$LOGFILE" 2>&1 &