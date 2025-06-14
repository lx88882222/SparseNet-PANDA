# dataset settings
dataset_type = 'CocoDataset'
# data_root will be defined per dataloader to handle different paths for train/val

file_client_args = dict(backend='disk')

# Define PANDA classes
# IMPORTANT: Verify this based on your actual annotation files.
metainfo = dict(classes=('person',)) # Assuming 'person' is the primary or only class

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='/data1/lixiang/panda/', # Adjusted data_root for training
        metainfo=metainfo,
        ann_file='smoke_train_10.json', # <--- 修改为冒烟测试训练标注
        data_prefix=dict(img='patch_mix_alltrain/'), # Relative to data_root, ensure trailing slash
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='/data1/lixiang/PANDA/PANDA/', # Adjusted data_root for validation
        metainfo=metainfo,
        ann_file='coco_json/smoke_val_5.json', # <--- 修改为冒烟测试验证标注
        data_prefix=dict(img='patches/s4/'), # Relative to data_root, ensure trailing slash
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='/data1/lixiang/PANDA/PANDA/coco_json/smoke_val_5.json', # Corrected to smoke_val_5.json
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

#
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     # dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
#     # dict(type='RandomCrop', crop_size=(256 * 14, 256 * 7)),
#     # dict(type='Resize', img_scale=(512 * 4, 512 * 2), keep_ratio=True),
#     # dict(type='Resize', img_scale=(256 * 14, 256 * 7), keep_ratio=True),
#     dict(type='Resize', img_scale=(256 * 10, 256 * 5), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
#
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         # img_scale=(256 * 14, 256 * 7),
#         img_scale=(256 * 10, 256 * 5),
#         # img_scale=(512 * 4, 512 * 2),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# classes = ('person',)
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file='/home/liwenxi/panda/raw/PANDA/coco_json/train_s4.json',
#         img_prefix='/file_client_argshome/liwenxi/panda/raw/PANDA/patches/s4',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file='/home/liwenxi/panda/raw/PANDA/coco_json/val_s4.json',
#         img_prefix='/home/liwenxi/panda/raw/PANDA/patches/s4',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         classes=classes,
#         ann_file='/home/liwenxi/panda/raw/PANDA/coco_json/val_s4.json',
#         img_prefix='/home/liwenxi/panda/raw/PANDA/patches/s4',
#         pipeline=test_pipeline))
#
#
# evaluation = dict(interval=1, metric='bbox')
