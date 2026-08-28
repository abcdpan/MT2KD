# MT²KD: Multi‑Task Learning for Multi‑Temperature Knowledge Distillation




![Framework of MT²KD](D:\papers\code_submit\MT2KD\MT2KD\1.png)

![Framework of MT²KD](D:\papers\code_submit\MT2KD\MT2KD\2.png)


This repo:
**(1) covers the implementation of the following ICLR 2020 paper:**
"Contrastive Representation Distillation" (CRD). [Paper](http://arxiv.org/abs/1910.10699), [Project Page](http://hobbitlong.github.io/CRD/).
<p></p>

## Installation

Python 3.9+

Torch 2.0.0+

## Running

1. Parameter Description
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
2. Train teacher classifications:

   (T=3 and T=6) 

   ```
   python train_teacherCls.py --dataset cifar100 --path_t ./scripts/save/resnet32x4_vanilla/ckpt_epoch_240.pth --cls_channels 256 --cls_size 1 --kd_T 3 
   ```
   ```
   python train_teacherCls.py --dataset cifar100 --path_t ./scripts/save/resnet32x4_vanilla/ckpt_epoch_240.pth --cls_channels 256 --cls_size 1 --kd_T 6 
   ```
   
3. Train student:
   ```
   python train_student_muilti_to_muilti.py --dataset cifar100 --path_t ./scripts/save/resnet32x4_vanilla/ckpt_epoch_240.pth --cls_channels 256 --cls_size 1 --path_cls1  MT2KD/save/classifier/models/Classifier_T_resnet32x4_cifar100_T3_1/classifier_best.pth --path_cls2 MT2KD/save/classifier/models/Classifier_T_resnet32x4_cifar100_T6_1/classifier_best.pth --model_s resnet8x4_multiCls --distill kd --cls 0.1 --div 9.0 -b 0
   ```# MT2KD
