import torch
    # 假设 sparsenet_ls.py 和 ls_conv_module.py 都在 mmdet.models.backbones 路径下
    # 如果直接运行此脚本，可能需要调整PYTHONPATH或使用更绝对的导入路径
    # 为了简单起见，如果此测试脚本与SparseFormer目录同级，可以尝试：
    # import sys
    # sys.path.insert(0, './SparseFormer') # 将SparseFormer目录加入PYTHONPATH

try:
        from mmdet.models.backbones.sparsenet_ls import SparseNet # 确保类名正确
        print("成功导入 SparseNet (LS版)")
except ImportError as e:
        print(f"导入 SparseNet (LS版) 失败: {e}")
        print("请确保 sparsenet_ls.py 文件路径正确，并且所有依赖（如ls_conv_module, ska）可以被找到。")
        print("如果直接运行此脚本，可能需要将包含 mmdet 的父目录加入 sys.path。")
        exit()
except Exception as e_other:
        print(f"导入时发生其他错误: {e_other}")
        exit()


def test_instantiation_and_forward():
        # 这些参数需要根据 SparseNet 的 __init__ 方法来设定
        # 您可能需要参考 mmdetection 中使用 SparseNet 的配置文件来获取典型的参数值
        # 或者使用其默认值（如果定义了的话）
        # 这里的 'layers' 参数指的是 BlockSequence 中 Global/LocalBlock 内部 BasicBlock/LSConv 的数量
        # SparseNet.__init__ 中的 'depths' 参数控制每个 stage 有多少个 BlockSequence.block
        # BlockSequence.__init__ 中的 'layers' 参数控制每个 Global/Local block内部有多少个 BasicBlock/LSConv.
        # SparseNet.__init__ 中的 'layers' 实际上是传给 BlockSequence的 'layers' 参数
        
        model_config = dict(
            pretrain_img_size=224,
            in_channels=3,
            embed_dims=96,  # 初始 embed_dims
            patch_size=4,
            layers=(2, 2, 2, 2), # 这是传递给 BlockSequence 的 'layers' 参数 -> 即每个Global/LocalBlock内部的卷积单元数
            window_size=7,
            mlp_ratio=4,
            depths=(2, 2, 6, 2), # 每个 stage 中 Global/Local block 对的数量
            num_heads=(3, 6, 12, 24), # 即使是卷积版，这些参数名可能保留
            strides=(4, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            top_k=(0.7, 0.6, 0.5, 0.5), # ScoreNet 相关参数
            qkv_bias=True,
            qk_scale=None,
            patch_norm=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            use_abs_pos_embed=False,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'), # LayerNorm
            # norm_cfg=dict(type='BN'), # 如果模型内部主要用BN，这里也要对应
            with_cp=False,
            # init_cfg=None # 通常用于加载预训练权重
        )

        # 重要: 检查 SparseNet 的 __init__ 中 norm_cfg 的默认类型。
        # BasicBlock 和 LSConv 内部使用的是 BatchNorm2d。
        # SparseNet 的 PatchEmbed, PatchMerging, 和最后的 norm{i} 可能使用 LayerNorm。
        # 需要确保 norm_cfg 与 LSConv/BasicBlock 内部期望的类型兼容，或分别处理。
        # LSConv 内部的 LKP 使用了 Conv2d_BN，SKA 后有 nn.BatchNorm2d。
        # BasicBlock 内部也使用 norm_layer (默认为 nn.BatchNorm2d)。
        # 因此，传递给 SparseNet 的全局 norm_cfg 可能主要影响 PatchEmbed, PatchMerging 和输出的 norm{i}。
        # 如果 norm_cfg=dict(type='LN')，确保 LocalBlock 和 GlobalBlock 内部的 _norm_layer
        # 被正确设置为 nn.BatchNorm2d，或者 LSConv/BasicBlock 能处理 LN。
        # 从代码看，_make_layer 中的 norm_layer = self._norm_layer，
        # 而 LocalBlock/GlobalBlock 的 __init__ 中 self._norm_layer = nn.BatchNorm2d。
        # 所以这里的 norm_cfg=dict(type='LN') 应该是用于 SwinTransformer 风格的模块。

        print("尝试实例化模型...")
        try:
            model = SparseNet(**model_config)
            model.eval() # 设置为评估模式
            print("模型实例化成功!")
        except Exception as e:
            print(f"模型实例化失败: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n尝试前向传播...")
        try:
            # 使用一个典型的输入尺寸，例如 224x224 或 PANDA 数据集切片后的大小
            # 如果 patch_size=4, strides[0]=4, 224/4 = 56.
            # 如果 pretrain_img_size 和实际输入尺寸不一致，且使用了绝对位置编码，可能会有问题。
            # 但这里 use_abs_pos_embed=False
            dummy_input = torch.randn(1, 3, 224, 224) # (B, C, H, W)
            
            # 如果您的环境有GPU并且PyTorch是GPU版本
            if torch.cuda.is_available():
                model = model.cuda()
                dummy_input = dummy_input.cuda()
                print("模型和输入已移至CUDA")

            with torch.no_grad(): # 在评估模式下，不需要计算梯度
                outputs = model(dummy_input)
            
            print("模型前向传播成功!")
            print("输出特征数量:", len(outputs))
            for i, out_feat in enumerate(outputs):
                print(f"  输出特征 {i} 形状: {out_feat.shape}")

        except Exception as e:
            print(f"模型前向传播失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
        test_instantiation_and_forward()
