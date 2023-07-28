# DenseMP: Unsupervised Dense Pre-training for Few-shot Medical Image Segmentation

Official PyTorch implementation for the paper:

> **DenseMP: Unsupervised Dense Pre-training for Few-shot Medical Image Segmentation**
>
> Zhaoxin Fan, Puquan Pan, Zeren Zhang, Ce Chen, Tianyang Wang, Siyang Zheng, Min Xu
>
> <a href='https://arxiv.org/abs/2307.09604'><img src='https://img.shields.io/badge/arXiv-2307.09604-red'></a>


<p align="center">
<img src="./figure/overview.png" width="90%" />
</p>

> Few-shot medical image semantic segmentation is of paramount importance in the domain of medical image analysis. However, existing methodologies grapple with the challenge of data scarcity during the training phase, leading to over-fitting. To mitigate this issue, we introduce a novel Unsupervised Dense Few-shot Medical Image Segmentation Model Training Pipeline (DenseMP) that capitalizes on unsupervised dense pre-training. DenseMP is composed of two distinct stages: (1) segmentation-aware dense contrastive pre-training, and (2) few-shot-aware superpixel guided dense pre-training. These stages collaboratively yield a pre-trained initial model specifically designed for few-shot medical image segmentation, which can subsequently be fine-tuned on the target dataset. Our proposed pipeline significantly enhances the performance of the widely recognized few-shot segmentation model, PA-Net, achieving state-of-the-art results on the Abd-CT and Abd-MRI datasets. Code will be released after acceptance.

#### To-Do List

- [ ] Uploading trained Model (stage1/stage2)

## &#x1F527; Usage
### Stage 1: Segmentation-aware Dense Contrastive Pre-training
#### Pre-step
1. Create a conda virtual environment and activate it.
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

2. Installing
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
cd DenseCL
pip install -r requirements.txt
python setup.py develop
```

1. (Optinal) Install `google` and `future tensorboard`
```
pip install google
pip install future tensorboard
```

#### Datasets

- ROCO: [Radiology Objects in COntext (ROCO)](https://github.com/razorx89/roco-dataset)
  Here we only use data from `/roco-dataset/data/train/radiology/`. And we implement the data structure as follows
  ```
  Stage1
  ├── openselfsup
  ├── benchmarks
  ├── configs
  ├── data
  │   ├── roco
  │   │   ├── ....jpg
  │   │   ├── ....jpg
  ```


#### Models

- Download the pre-trained backbones from [ResNet101 pretrained by ImageNet](https://cloudstor.aarnet.edu.au/plus/s/4sugyvuBOiMXXnC/download) and put the pth file under `/Stage1/initmodel/`
- Download our trained Stage1 backbone from [gdrive]() and put them under `Stage2/pretrained_model/densecl/radiology_medical_image/`, for Stage 2 training.


#### Scripts
- `cd ./Stage1`
- **Step 1** *Training*
  Train the shared backbone used in Stage 2 with the following command line.
  `bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]`
  Optional arguments are:
  - `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
  - `--pretrained ${PRETRAIN_WEIGHTS}`: Load pretrained weights for the backbone.
  - 
- **Step 2** *Extracting Backbone Weights*
  ```
  WORK_DIR=work_dirs/selfsup/densecl/densecl_roco_50ep/
  CHECKPOINT=${WORK_DIR}/epoch_50.pth
  WEIGHT_FILE=${WORK_DIR}/extracted_densecl_roco_50ep.pth

  python tools/extract_backbone_weights.py ${CHECKPOINT} ${WEIGHT_FILE}
  ```

### Stage 2: Few-shot-aware Superpixel Guided Dense Pre-training
#### Pre-step: Installing dependencies
```
cd ./Stage2
pip install -r requirements.txt
```
#### Datasets
- Abdominal MRI:  [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)
- Abdominal CT: [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

For pre-processed data of Abd-MRI and Abd-CT, please download from [gdrive]()

#### Models

- Download the pre-trained backbones of Stage 1 from [gdrive]() and put them under `Stage2/pretrained_model/densecl/radiology_medical_image/`. 
- We provide trained DenseMP [models]() for performance evaluation. 

#### Scripts
The results will be saved in `exps`folder; remember to make sure the target`logs`folder exist, or the logs may not be saved.
- `cd ./Stage2`
- **Step 1** *Training*

  ```
  bash examples/mri_ct_s1_l0/02_DenseCL_Medical/train_ssl_abdominal_mri_densecl_5_fold.sh | tee logs/mri_ct_s1_l0/02_DenseCL_Medical_Train_5k_Mix_0
  ```

- **Step 2** *Testing*

  ```
  bash examples/mri_ct_s1_l0/02_DenseCL_Medical/test_ssl_abdominal_mri_densecl_5_fold.sh | tee logs/mri_ct_s1_l0/02_DenseCL_Medical_Test_5k_Mix_0
  ```

- **Step 3** *Finetuning* 
  After obtaining the best model from Step 1&2, remember the modified the `sh`file in order to finetune the model to obtain the best performance.
  ```
  bash examples/mri_ct_s1_l0/02_DenseCL_Medical/ft_test_ssl_abdominal_mri_densecl_5_fold.sh | tee logs/mri_ct_s1_l0/02_DenseCL_Medical_Ft_Train_5k_Mix_0
  ```
  after training, we evaluate the performance of the method
  ```
  bash examples/mri_ct_s1_l0/02_DenseCL_Medical/ft_test_ssl_abdominal_mri_densecl_5_fold.sh | tee logs/mri_ct_s1_l0/02_DenseCL_Medical_Ft_Test_5k_Mix_0
  ```

#### Performance

Performance comparison with the state-of-the-art approaches (*i.e.*, [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation) in terms of **average** **Dice** across all folds. 

1. ##### Setting 1

   | Method     | Abd-CT                   | Abd-MRI                  |
   | ---------- | ------------------------ | ------------------------ |
   | SSL-ALPNet | 73.35                    | 78.84                    |
   | DenseMP    | 72.84 <sub>(-0.51)</sub> | 79.83 <sub>(+0.99)</sub> |

2. ##### Setting 2

   | Method     | Abd-CT                   | Abd-MRI                  |
   | ---------- | ------------------------ | ------------------------ |
   | SSL-ALPNet | 63.02                    | 73.02                    |
   | DenseMP    | 68.39 <sub>(+5.37)</sub> | 76.86 <sub>(+3.84)</sub> |

#### Visualization

<p align="middle">
    <img src="figure/visualization.png">
</p>


### Supporting Experiments
Detail comming soon

### References

This repo is mainly built based on [DenseCL](https://github.com/WXinlong/DenseCL), [SSL_ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation). Thanks for their great work!


### BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@article{fan2023densemp,
  title={DenseMP: Unsupervised Dense Pre-training for Few-shot Medical Image Segmentation},
  author={Fan, Zhaoxin and Pan, Puquan and Zhang, Zeren and Chen, Ce and Wang, Tianyang and Zheng, Siyang and Xu, Min},
  journal={arXiv preprint arXiv:2307.09604},
  year={2023}
}
```
