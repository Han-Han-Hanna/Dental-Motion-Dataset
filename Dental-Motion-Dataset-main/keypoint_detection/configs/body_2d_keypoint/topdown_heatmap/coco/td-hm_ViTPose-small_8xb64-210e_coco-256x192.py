# ===================== 基础 =====================
default_scope = 'mmpose'
log_level = 'INFO'

# ⭐ 相对路径（最关键）
data_root = 'data/teeth'
work_dir = 'work_dirs/vitpose_small_teeth'

NUM_KEYPOINTS = 35

# ===================== 可视化 =====================
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='PoseLocalVisualizer', vis_backends=vis_backends)

# ===================== 训练策略 =====================
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=210, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===================== 优化器 =====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.8
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=210, milestones=[170, 200], gamma=0.1, by_epoch=True)
]

# ===================== 编码器 =====================
codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2
)

# ===================== 模型 =====================
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch=dict(
            embed_dims=384,
            num_layers=12,
            num_heads=12,
            feedforward_channels=384 * 4
        ),
        img_size=(256, 192),
        patch_size=16,
        with_cls_token=False,
        out_type='featmap',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_small_20230913.pth'
        )
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=384,
        out_channels=NUM_KEYPOINTS,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(flip_test=True, flip_mode='heatmap')
)

# ===================== 数据集 =====================
metainfo = dict(
    dataset_name='tooth_dataset',
    num_keypoints=NUM_KEYPOINTS,
    keypoint_info=dict({
        i: dict(name=f'tooth_kpt_{i}', id=i, color=[255, 0, 0], type='upper', swap='')
        for i in range(NUM_KEYPOINTS)
    }),
    flip_pairs=[],
    joint_weights=[1.0] * NUM_KEYPOINTS,
    sigmas=[0.089] * NUM_KEYPOINTS,
    skeleton_info=dict(),
    norm_keypoint_idxs=(0, 17)
)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=8,
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
    batch_size=4,
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

# ===================== hooks =====================
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='coco/AP', rule='greater'),
    logger=dict(type='LoggerHook', interval=50)
)