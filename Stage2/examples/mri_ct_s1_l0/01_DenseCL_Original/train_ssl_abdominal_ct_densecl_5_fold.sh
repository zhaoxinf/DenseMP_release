#!/bin/bash
# train a model to segment abdominal CT
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="01_DenseCL_Original_Train"
DATASET='SABS_Superpix'
NWORKER=0

ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)
ALL_SCALE=("MIDDLE") # config of pseudolabels

### Use L/R kidney as testing classes
LABEL_SETS=0
EXCLU='[]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
# LABEL_SETS=1
# EXCLU='[1,6]'

###### Training configs ######
NSTEP=100100
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=25000 # interval for saving snapshot
LOG_INTERVAL=30 # interval for logging
SEED='1234'

###### Validation configs ######
SUPP_ID='[6]' # using the additionally loaded scan as support

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    echo "====== TRAIN $EVAL_FOLD START ======"
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
    PREFIX="train_densecl_5_fold_${DATASET}_lbgroup${LABEL_SETS}_scale_${SUPERPIX_SCALE}_vfold${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./exps/mri_ct_s1_l0/${CPT}_${SUPERPIX_SCALE}_${LABEL_SETS}"

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 python3 training.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    'use_coco_init=False' \
    'use_densecl_init=True' \
    'pretrained_backbone=pretrained_model/densecl/imagenet/densecl_r101_imagenet_200ep.pth' \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    print_interval=$LOG_INTERVAL \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID
    done
    echo "====== TRAIN $EVAL_FOLD END ======"
done
