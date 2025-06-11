custom_imports = dict(
    imports=['mmdet.models.backbones.sparsenet_ls'], 
    allow_failed_imports=False
)

_base_ = [
    '../_base_/datasets/panda_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
model = dict(
    type='ATSS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=128),
    backbone=dict(
        type='SparseNet',
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        strides=(4, 2, 2, 2),
        layers=(2, 2, 2, 2),
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        top_k=[0.3, 0.3, 0.3, 0.3],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        init_cfg=None
    ),
    neck=[
        dict(
            type='FPN',
            in_channels=[192, 384, 768],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='DyHead',
            in_channels=256,
            out_channels=256,
            num_blocks=6,
            zero_init_offset=False)
    ],
    bbox_head=dict(
        type='ATSSHead',
        num_classes=1,
        in_channels=256,
        pred_kernel_size=1,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            center_offset=0.5),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# --- Dataset specific overrides (example, verify with your panda_detection.py) ---
# data_root = 'data/PANDA/' # Modify if your _base_ dataset config needs override
# train_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=dict(classes=('person',)), # Ensure this matches num_classes in bbox_head
#         ann_file='annotations/panda_train_coco.json',
#         data_prefix=dict(img='train_images/')
#     )
# )
# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         metainfo=dict(classes=('person',)),
#         ann_file='annotations/panda_val_coco.json',
#         data_prefix=dict(img='val_images/')
#     )
# )
# val_evaluator = dict(
#     ann_file=data_root + 'annotations/panda_val_coco.json'
# )
# test_dataloader = val_dataloader
# test_evaluator = val_evaluator
# --- End Dataset specific ---

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    clip_grad=None)

# # Modify learning rate and schedule if needed
# train_cfg = dict(max_epochs=24) # For a 2x schedule for example
# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=False,begin=0,end=1000),
#     dict(type='MultiStepLR',begin=0,end=24,by_epoch=True,milestones=[16, 22],gamma=0.1)
# ]

# --- Settings for Smoke Test ---
# train_cfg = dict( 
#     type='EpochBasedTrainLoop', # 确保与 schedule_1x.py 中的类型一致
#     max_epochs=1, 
#     val_interval=1 
# )
# 如果你的 schedule_1x.py 中 train_cfg.val_interval 不是1，也在这里覆盖
# 例如，如果 schedule_1x.py 使用 IterBasedTrainLoop:
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=100, val_interval=10) # 迭代100次，每10次验证一次
# 具体看你的 schedule_1x.py 是基于 epoch 还是 iteration

# Configure DDP wrapper to find unused parameters
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)
