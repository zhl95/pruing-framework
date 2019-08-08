# A unified framework for various pruning methods. Fully object oriented, high extensibility.


## Table of Contents

- [Requirements](#requirements)
- [Quick start](#Quick start)
- [Architecture](#Architecture)
  - [Trainer](#Trainer)
  - [Methodology](#Methodology)
  - [Layer variants](#Layer variants)
  - [Network](#Network)
  - [Config and Args](#Config and Args)
- [Added pruning methods](#Added pruning methods)
  - [Norm based non-structural method](#Norm based non-structural method)
  - [Norm-based filter pruning](#Norm-based filter pruning)
  - [Geomeric median-based filter pruning](#Geomeric median-based filter pruning)
- [Create a new pruning methods](#Create a new pruning methods)
- [Notes](#notes)
  - [Torchvision Version](#torchvision-version)
  - [Why use 100 epochs for training](#why-use-100-epochs-for-training)
  - [Process of ImageNet dataset](#process-of-imagenet-dataset)
  - [FLOPs Calculation](#flops-calculation)
- [Citation](#citation)




## Requirements
- Python 3.6
- PyTorch 1.0.1 tested, 0.4.0 and above should also work.
- TorchVision 0.2.2

## Quick start
 -source source.sh to activate the env. update env name to your existing or create it.
 -python run1.py 


## Architecture

#### Trainer
training related class and methods are defined in train3.py, inlcuding train(), validate(), try_resume(). 


#### Methodology
Pruning methods are defined in functions/, for example, mask.py is the method for norm and geo-median filter pruning.

#### Layer variants
Here new layers is derived from basic Pytorch layers such as Conv2D and Linear. Override the original forward and backward function
as needed. In this way, there is no need to explicitly load and rewrite data using buffer at the top main().

#### Network
Networks are defined in models/, import desired layer variants before constructing the network.

## Config and Args
Configurations are fined in config_.py with easydict, which can be updated by arguments. Args parser is in top main file: run1.py.


## Added pruning methods
Here I added several popular pruning methods.

#### Norm based non-structural method

#### Norm-based filter pruning

#### Geomeric median-based filter pruning

## Create a new pruning methods

## Notes

#### Torchvision Version
We use the torchvision of 0.3.0. If the version of your torchvision is 0.2.0, then the `transforms.RandomResizedCrop` should be `transforms.RandomSizedCrop` and the `transforms.Resize` should be `transforms.Scale`.

#### Why use 100 epochs for training
This can improve the accuracy slightly.

#### Process of ImageNet dataset
We follow the [Facebook process of ImageNet](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).
Two subfolders ("train" and "val") are included in the "/path/to/ImageNet2012".
The correspding code is [here](https://github.com/he-y/filter_similarity/blob/master/pruning_imagenet.py#L136-L137).

#### FLOPs Calculation
Refer to the [file](https://github.com/he-y/soft-filter-pruning/blob/master/utils/cifar_resnet_flop.py).



## Citation
For Filter Pruning via Geometric Median:
![i1](https://github.com/he-y/filter-pruning-geometric-median/blob/master/functions/explain.png)
```
@inproceedings{he2019filter,
  title     = {Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration},
  author    = {He, Yang and Liu, Ping and Wang, Ziwei and Hu, Zhilan and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```
