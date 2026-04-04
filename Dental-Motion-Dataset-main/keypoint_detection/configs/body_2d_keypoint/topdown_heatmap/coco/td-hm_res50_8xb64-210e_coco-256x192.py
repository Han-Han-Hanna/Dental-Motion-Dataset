_base_ = ['../../../_base_/default_runtime.py']

# ===================== 基础配置 =====================
default_scope = 'mmpose'
log_level = 'INFO'

# ⭐ 相对路径（核心）
data_root = 'data/teeth'
work_dir = 'work_dirs/res50_teeth'

NUM_KEYPOINTS = 35

# ===================== 数据集信息 =====================
metainfo = dict(
    dataset_name='teeth_dataset',
    num_keypoints=NUM_KEYPOINTS,
    keypoint_info=dict({
        i: dict(id=i, name=f'kpt_{i}', color=[255, 0, 0], type='teeth', swap='')
        for i in range(NUM_KEYPOINTS)
    }),
    skeleton_info=dict(),
    joint_weights=[1.0] * NUM_KEYPOINTS,
    sigmas=[0.089] * NUM_KEYPOINTS,
    flip_pairs=[],
    norm_keypoint_idxs=(0, 17)
)

# ===================== 编码器 =====================
codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=3
)

# ===================== 数据增强 =====================
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# ===================== 模型 =====================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        out_indices=(3,),
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=NUM_KEYPOINTS,
        deconv_out_channels=[256, 256, 256],
        deconv_kernel_sizes=[4, 4, 4],
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True)
)

# ===================== 数据加载 =====================
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/person_keypoints_train.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo,
        pipeline=val_pipeline,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# ===================== 评估 =====================
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + '/annotations/person_keypoints_val.json'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=(0, 17))
]

test_evaluator = val_evaluator

# ===================== 训练策略 =====================
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=210, milestones=[170, 200], gamma=0.1, by_epoch=True)
]

# ===================== hooks =====================
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='NME', rule='less'),
    logger=dict(type='LoggerHook', interval=50)
)

# ===================== 可视化 =====================
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')]
)