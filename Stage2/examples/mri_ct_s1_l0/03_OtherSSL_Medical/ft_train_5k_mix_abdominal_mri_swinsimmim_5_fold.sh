#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=1
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="03_SwinSimMIM_Medical_Ft_Train_5k"
DATASET='CHAOST2'
NWORKER=0

ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)
# ALL_SCALE=("MIDDLE") # config of pseudolabels

### Use L/R kidney as testing classes
# LABEL_SETS=0
# EXCLU='[2,3]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
LABEL_SETS=0
EXCLU='[]'

###### Training configs ######
NSTEP=5100
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=1000 # interval for saving snapshot
LOG_INTERVAL=30 # interval for logging
SEED='1234'

###### Validation configs ######
SUPP_ID='[4]' #  # using the additionally loaded scan as support

# declare -A dict
# RELOAD_PATHS=(["0"]="exps/mri_ct_s1_l1/02_DenseCL_Medical_Train_MIDDLE_1/mySSL_train_densecl_5_fold_CHAOST2_Superpix_lbgroup1_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_1_1shot/1/snapshots/75000.pth" ["1"]="exps/mri_ct_s1_l1/02_DenseCL_Medical_Train_MIDDLE_1/mySSL_train_densecl_5_fold_CHAOST2_Superpix_lbgroup1_scale_MIDDLE_vfold1_CHAOST2_Superpix_sets_1_1shot/1/snapshots/100000.pth" ["2"]="exps/mri_ct_s1_l1/02_DenseCL_Medical_Train_MIDDLE_1/mySSL_train_densecl_5_fold_CHAOST2_Superpix_lbgroup1_scale_MIDDLE_vfold2_CHAOST2_Superpix_sets_1_1shot/1/snapshots/75000.pth" ["3"]="exps/mri_ct_s1_l1/02_DenseCL_Medical_Train_MIDDLE_1/mySSL_train_densecl_5_fold_CHAOST2_Superpix_lbgroup1_scale_MIDDLE_vfold3_CHAOST2_Superpix_sets_1_1shot/1/snapshots/25000.pth" ["4"]="exps/mri_ct_s1_l1/02_DenseCL_Medical_Train_MIDDLE_1/mySSL_train_densecl_5_fold_CHAOST2_Superpix_lbgroup1_scale_MIDDLE_vfold4_CHAOST2_Superpix_sets_1_1shot/1/snapshots/50000.pth")

for EVAL_FOLD in "${ALL_EV[@]}"
do
    echo "====== FINETUNE $EVAL_FOLD START ======"
    PREFIX="ft_train_5_fold_${DATASET}_lbgroup${LABEL_SETS}_MIX_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps/mri_ct_s1_l0/${CPT}_MIX_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    RELOAD_PATH="/data/zhangzeguang/MedicalSegmentation/mmpretrain/output/simmim_swin-base-w7_4xb64_50e_ROCO/epoch_50.pth"

    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 python3 finetuning.py with \
    'modelname=dlfcn_swin_mmcv' \
    'clsname=grid_proto' \
    'usealign=True' \
    'optim_type=sgd' \
    'use_coco_init=False' \
    'use_densecl_init=False' \
    reload_model_path=$RELOAD_PATH \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    print_interval=$LOG_INTERVAL \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID
    echo "====== FINETUNE $EVAL_FOLD END ======"
done
