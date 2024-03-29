
# MRLA-Net

Title: Cross-Layer Retrospective Retrieving via Layer Attention (accepted by ICLR-2023, [paper](http://arxiv.org/abs/2302.03985))


MRLA-base Module:

<img src="figures/MRLA_eq6_ver2.png" width="600" alt="MRLA-base"/><br/>

    
MRLA-light Module:

<img src="figures/MRLA_eq8.png" width="600" alt="MRLA-light"/><br/>


## Installation

### Base environment for ImageNet Classification

1. Create a conda virtual environment and activate it.
    ```shell
    conda create -n mrla python=3.7 -y
    conda activate mrla
    ```

2. PyTorch versions 1.4, 1.5.x, 1.6, 1.7.x, and 1.8 are supported
    ```shell
    # CUDA 11.1
    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
    
    # CUDA 10.1
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

    # CUDA 10.2
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
    ```

### timm

Please follow the official installation.

### MMDetection

Please follow the official installation.

### DeiT/CeiT

Please follow their github to install PyTorch 1.7.0+ and torchvision 0.8.1+ and pytorch-image-models 0.3.2


## Quick Start

### Train with ResNet on ImageNet-1K

#### - Use single node or multi node with multiple GPUs

Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.

For example, to train ResNet-50 with MRLA-light

  ```bash
  python train.py '/imagenet' -a resnet50_mrlal -b 256 --epochs 100 --warmup-epochs 3 --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:12300' --world-size 1 --rank 0 --workers 10
  ```

#### - Specify single GPU or multiple GPUs

For example, to train ResNet-50 with MRLA using 2 GPUs

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python train.py '/imagenet' -a resnet50_mrlal -b 256 --epochs 100 --warmup-epochs 3 --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:12300' --world-size 1 --rank 0 --workers 10
  ```

### Testing

To evaluate the best model

  ```bash
  python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 --resume {path to the best model} -e {imagenet-folder with train and val folders}
  ```
  
### MMDetection

We provide models and config files for MMDetection. Put the files into the same folder as in this repository, e.g., put 'resnet_mrlal.py' in './mmdetection/mmdet/models/backbones/', and import the model in the __init__.py file.

Note that the config files of the latest version of MMDetection are a little different from previous one. Specifically, use 'init_cfg=' instead of 'pretrained=' to load the pretrained weights.

To train a faster_rcnn with our MRLA on ResNet-50 using 2 GPUs (batch=16, samples_per_gpu=8),

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python tools/train.py configs/faster_rcnn/faster_rcnn_r50mrlal_fpn_1x_coco.py --cfg-options data.samples_per_gpu=8
  ```

### Train with EfficientNet on ImageNet-1K

Please install pytorch-image-models (timm) first. There would be some differences between different verions of timm. My version is timm==0.4.9

Put the files in timm folder into the same folder of pytorch-image-models

To train a EfficientNet-B0 with our MRLA using 2 GPUs,

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 train.py '/imagenet' --model efficientnet_mrlal_b0 -b 384 --lr .048 --epochs 350 --sched step --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --aa rand-m9-mstd0.5 --amp --remode pixel --reprob 0.2
  ```


### Train with DeiT on ImageNet-1K

To train DeiT-T with MRLA, batch size of 4x256 on 4 GPUs

  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_mrlal_tiny_patch16_224 --batch-size 256 --data-path '/imagenet' 
  ``` 

