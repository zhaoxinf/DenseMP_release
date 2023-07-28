#!/bin/bash
# train a model to segment abdominal CT
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
CPT="02_DenseCL_Medical_Ft_Test_5k"
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

###### Training configs (irrelavent in testing) ######
NSTEP=5100
DECAY=0.95

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=25000 # interval for saving snapshot
SEED='1234'

###### Validation configs ######
SUPP_ID='[6]' # using the additionally loaded scan as support

MODEL_EPOCH_LIST=(1000 2000 3000 4000 5000)

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do
        for MODEL_EPOCH in "${MODEL_EPOCH_LIST[@]}"
        do
        echo "====== EVAL EVAL_FOLD $EVAL_FOLD MODEL_EPOCH $MODEL_EPOCH START ======"
        PREFIX="ft_test_5_fold_${DATASET}_lbgroup${LABEL_SETS}_MIX_vfold${EVAL_FOLD}_epoch${MODEL_EPOCH}"
        echo $PREFIX
        LOGDIR="./exps/mri_ct_s1_l0/${CPT}_MIX_${LABEL_SETS}"

        if [ ! -d $LOGDIR ]
        then
            mkdir $LOGDIR
        fi

        RELOAD_PATH="exps/mri_ct_s1_l0/02_DenseCL_Medical_Ft_Train_5k_MIX_0/mySSL_ft_train_5_fold_SABS_lbgroup0_MIX_vfold${EVAL_FOLD}_SABS_sets_0_1shot/1/snapshots/${MODEL_EPOCH}.pth" # path to the reloaded model
        echo "RELOAD_PATH: $RELOAD_PATH"

        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 python3 validation.py with \
        'modelname=dlfcn_res101' \
        'usealign=True' \
        'optim_type=sgd' \
        'use_coco_init=True' \
        'use_densecl_init=False' \
        reload_model_path=$RELOAD_PATH \
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
        save_snapshot_every=$SNAPSHOT_INTERVAL \
        superpix_scale=$SUPERPIX_SCALE \
        lr_step_gamma=$DECAY \
        path.log_dir=$LOGDIR \
        support_idx=$SUPP_ID
        echo "====== EVAL EVAL_FOLD $EVAL_FOLD MODEL_EPOCH $MODEL_EPOCH END ======"
        done
    done
done