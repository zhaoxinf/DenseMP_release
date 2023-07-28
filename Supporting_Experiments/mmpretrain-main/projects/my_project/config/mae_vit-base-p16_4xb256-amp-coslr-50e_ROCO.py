model = dict(
    type='MAE',
    backbone=dict(type='MAEViT', arch='b', patch_size=16, mask_ratio=0.75, img_size=256),
    neck=dict(
        type='MAEPretrainDecoder',
        num_patches=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0),
    head=dict(
        type='MAEPretrainHead',
        norm_pix=False,
        patch_size=16,
        loss=dict(type='PixelReconstructionLoss', criterion='L2')),
    init_cfg=dict(type='Pretrained', checkpoint='./pretrained_ckpts/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth')
    # init_cfg=[
        # dict(type='Xavier', layer='Linear', distribution='uniform'),
        # dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)]
    )
dataset_type = 'CustomDataset'
data_root = '../ROCO'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[89.82843423133949, 89.82843423133949, 89.82843423133949],
    std=[61.211195876257506, 61.211195876257506, 61.211195876257506],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadMedicalImage'),
    dict(type='Resize', scale=(256, 256)), 
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackInputs')
]
train_dataset = dict(type=dataset_type,
                    data_root=data_root,
                    data_prefix='train/radiology/images',
                    with_label=False,
                    pipeline=train_pipeline,
                    extensions=('.jpg'))
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=256,
    num_workers=8,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = True
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=0.0024, betas=(0.9, 0.95), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            ln=dict(decay_mult=0.0),
            bias=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0),
            mask_token=dict(decay_mult=0.0),
            cls_token=dict(decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        by_epoch=True,
        begin=5,
        end=50,
        convert_to_iter_based=True)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)
auto_scale_lr = dict(base_batch_size=4096)
find_unused_parameters=False
launcher = 'pytorch'
