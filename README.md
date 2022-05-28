# Learning with Knowledge Distillation for Fine Grained Image Classification

Fine-grained Image Classification (FGIC) is one of the challenging tasks in Computer Vision. Many recent methodologies including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have tried to solve this problem. In this study we show the effectiveness of using both CNNs and ViTs hand in hand to produce state of the art results on challenging FGIC datasets. We show that by using DeiT as student model and ConvNext as teacher model in knowledge distillation settings, we achieve top1 and top5 accuracies of 92.52\% and 99.15\% respectively on combined CUB + Stanford Dogs datasets. On a more challenging dataset named FoodX-251 we achieved top1 and top5 accuracies of 74.71\% and 93\% respectively.  

This repository contains the PyTorch based training and evaluation codes for reproducing main results of our project. Specifically, we provide instructions on:
<ol>
  <li>
    Creating a teacher model by finetuning a (ImageNet1k) pretrained ConvNeXt model on a fine-grained image classification dataset. 
  </li>
    <li>
    Creating a student model by finetuning (ImageNet1k) pretrained DeiT (Data-efficient image Transformer) model on a fine-grained image classification dataset. 
  </li>
    <li>
    Using Knowledge Distillation to further push forward the performance of DeiT student model by distilling knowledge from ConvNext teacher model for fine-grained image classification.
    </li>
  </ol>

<b> Note </b>: For part 1, refer to [our this repo](https://github.com/MUKhattak/ConvNext_FGVC).

## Technical Report 
Complete technical report can be viewed [here](https://github.com/MUKhattak/DeiT_ConvNeXt_KnowledgeDistillation/blob/deit_convnext/FGVC_report.pdf).

## Requirements and Installation
We have tested this code on Ubuntu 20.04 LTS with Python 3.8. Follow the instructions below to setup the environment and install the dependencies.
 ```shell
 conda create -n fgvcdeit python=3.8
 conda activate fgvcdeit
 # Install torch and torchvision
 pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
 # Install pytorch-image models (timm)
 pip3 install timm==0.3.2
 ```
 
  Now clone this repository:
  ```shell
 git clone https://github.com/MUKhattak/DeiT_ConvNeXt_KnowledgeDistillation.git
 cd DeiT_ConvNeXt_KnowledgeDistillation/
```
 
 
 ## Datasets
We provide support for 3 FGVC datasets for our experiments on KD distillation with DeiT and ConvNeXt.

<b> CUB Dataset </b>

Download CUB dataset from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view). Extract the file. Our code expects the dataset folder to have the following structure:

```
CUB_dataset_root_folder/
    └─ images
    └─ image_class_labels.txt
    └─ train_test_split.txt
    └─ ....
```
<b> FoodX Dataset </b>

Download FoodX dataset from [here](https://github.com/karansikka1/iFood_2019). After extracting the files, the root folder should have following structure:

```
FoodX_dataset_root_folder/
    └─ annot
        ├─ class_list.txt
        ├─ train_info.csv
        ├─ val_info.csv
    └─ train_set
        ├─ train_039992.jpg
        ├─ ....
    └─ val_set
        ├─ val_005206.jpg
        ├─ ....
```

<b> Stanford Dogs Dataset </b>

Download the dataset from [here](http://vision.stanford.edu/aditya86/ImageNetDogs/). The root folder should have following structure:

```
dog_dataset_root_folder/
    └─ Images
        ├─ n02092339-Weimaraner
            ├─ n02092339_107.jpg
            ├─ ....
        ├─ n02101388-Brittany_spaniel
            ├─ ....
        ├─ ....
    └─ splits
        ├─ file_list.mat
        ├─ test_list.mat
        ├─ train_list.mat

```

## Training and Evaluation 


 
