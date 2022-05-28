# Learning with Knowledge Distillation for Fine Grained Image Classification

Fine-grained Image Classification (FGIC) is one of the challenging tasks in Computer Vision. Many recent methodologies including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) have tried to solve this problem. In this study we show the effectiveness of using both CNNs and ViTs hand in hand to produce state of the art results on challenging FGIC datasets. We show that by using DeiT as student model and ConvNext as teacher model in knowledge distillation settings, we achieve top1 and top5 accuracies of 92.52\% and 99.15\% respectively on combined CUB + Stanford Dogs datasets. On a more challenging dataset named FoodX-251 we achieved top1 and top5 accuracies of 74.71\% and 92.99\% respectively.  

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
  </ul>

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
 git clone [https://github.com/MUKhattak/OD-Satellite-iSAID.git](https://github.com/MUKhattak/DeiT_ConvNeXt_KnowledgeDistillation.git)
 cd DeiT_ConvNeXt_KnowledgeDistillation/
```
 
