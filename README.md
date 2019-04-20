
## The MXNet Implementation of Face Age and Gender Estimation

This repository contains a lightweight model for face age and gender estimation. The model is very small and efficient about 1MB size, 10ms on single CPU core. Gender accuracy 96% on validation set and 4.1 age MAE.  

Two methods ([MTCNN](https://github.com/deepinx/mtcnn-face-detection) and [ESSH](https://github.com/deepinx/enhanced-ssh-mxnet)) are both provided in this repository for face detection and alignment. For easy cases, the results of two methods are almost the same, however, on hard cases the ESSH method has a much better detection results. You can use ``python test.py --det 0`` to choose MTCNN method, or use ``python test.py --det 1`` for ESSH.

## Environment

This repository has been tested under the following environment:

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (==1.3.0)

## Installation

1.  Prepare the environment.

2.  Clone the repository.
    
3.  Type  `make`  to build necessary cxx libs.


## Training

You can download rec format file directly in here: [BaiduCloud](https://pan.baidu.com/s/112tf6HQy3Yvo6F9L4jZopg) or [GoogleDrive](https://drive.google.com/open?id=1ztT0XM3aVUHIBCe8H1ch9rJMoS49PTql) , or package it youself according to the following steps

+ Download IMDB-WIKI dataset (face only) from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/.
+ Unzip them under *`./data`* or others path.
+ Pre-process each images in dataset and package it in a rec file.

You can first train on imdb, then fine tune on wiki dataset. Train MobileNet 0.25X on a GPU such as GTX-1080Ti according to the following command
```
CUDA_VISIBLE_DEVICES='0' python -u train.py --data-dir $DATA_DIR --prefix './models/model' --network m1 --multiplier 0.25 --per-batch-size 128 --lr 0.01 --lr-steps '10000' --ckpt 2
```
Instead, you can edit *`train.sh`* and run *`sh ./train.sh`* to train your models.

## Testing

  -  Download the ESSH model from [BaiduCloud](https://pan.baidu.com/s/1sghM7w1nN3j8-UHfBHo6rA) or [GoogleDrive](https://drive.google.com/open?id=1eX_i0iZxZTMyJ4QccYd2F4x60GbZqQQJ) and place it in *`./ssh-model/`*.

  -  You can use `python test.py` to test the pre-trained models or your own models.
 

## Results

Results of face age and gender estimation (inferenced from model MobileNet 0.25X) are shown below.

<div align=center><img src="https://raw.githubusercontent.com/deepinx/age-gender-estimation/master/sample-images/detection%20result_test1_22.02.2019.png" width="750"/></div>

## License

MIT LICENSE

## Acknowledgment

The code is adapted based on an intial fork from the [insightface](https://github.com/deepinsight/insightface) repository.

