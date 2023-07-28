dataset_type = 'CustomDataset'
data_root = '../ROCO'
data_preprocessor = dict(
    type='SelfSupDataPreprocessor',
    mean=[89.82843423133949, 89.82843423133949, 89.82843423133949],
    std=[61.211195876257506, 61.211195876257506, 61.211195876257506],
    to_rgb=False)

view_pipeline = [
    dict(type='RandomResizedCrop', scale=(256, 256), backend='cv2'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2,
                backend='cv2')
        ],
        prob=0.8),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 2.0),
        magnitude_std='inf',
        prob=0.5),
]

train_pipeline = [
    dict(type='LoadMedicalImage'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
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
    batch_size=64,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='LARS', lr=0.3, weight_decay=1e-06, momentum=0.9),
    paramwise_cfg=dict(
        custom_keys=dict({
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            'downsample.1': dict(decay_mult=0, lars_exclude=True)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=45, by_epoch=True, begin=5, end=50)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50)
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
resume = False
randomness = dict(seed=None, deterministic=False)
model = dict(
    type='SimCLR',
    backbone=dict(
        type='ResNet',
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.1),
    init_cfg=dict(type='Pretrained', checkpoint='./pretrained_ckpts/simclr_resnet50_16xb256-coslr-200e_in1k_20220825-4d9cce50.pth'))
auto_scale_lr = dict(base_batch_size=256)
launcher = 'pytorch'
