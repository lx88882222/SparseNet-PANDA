_base_ = './dino-4scale_sparseformer-t_8xb2-12e_panda.py'

# Override backbone to ensure with_cp=False for FLOPs calculation
model = dict(
    backbone=dict(
        _delete_=True,
        type='SparseFormer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        top_k=[0.7, 0.5, 0.3, 0.2],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,  # Explicitly set to False for FLOPs calculation
        convert_weights=False,  # Don't convert weights for FLOPs
        init_cfg=None  # No pretrained weights needed for FLOPs
    )
) 