import time
import warnings
from collections import OrderedDict
from copy import deepcopy

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.bricks.transformer import build_dropout as build_dropout_mmcv

from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import CheckpointLoader

from mmengine.utils import to_2tuple

from mmengine.logging import MMLogger
from mmdet.registry import MODELS

from ..layers import PatchEmbed, PatchMerging
# Removed LSConv import from here, as it will be used via LSNetOriginalBlock if needed

# Import Block from the new lsnet.py
from .lsnet import Block as LSNetOriginalBlock # Use an alias


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module): # This BasicBlock is for GlobalBlock, not LocalBlock
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# MultiheadAttention, SparseFFN, LocalWindowMSA, WindowMSA definitions remain unchanged from before.
# Ensure they are present in your actual file if they were part of the original sparsenet_ls_local.py you intend to modify.
# For brevity, I am omitting them here, but they should be kept if they were part of the original sparsenet_ls_local.py you intend to modify.
# Assuming they are present, the changes continue below:

class MultiheadAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)
        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = build_dropout_mmcv(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))
        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

class LocalBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads, # Not directly used by LSNetOriginalBlock for its core logic
                 blocks,    # This is num_internal_blocks for LSNetOriginalBlock sequence
                 stage_id,  # Not directly used by LSNetOriginalBlock if stage is fixed
                 feedforward_channels, # LSNetOriginalBlock has its own FFN sizing
                 window_size=7,
                 # shift=False, # Shift logic is for Swin MSA, not relevant here
                 qkv_bias=True, # For Swin MSA, not relevant for LSNetOriginalBlock
                 qk_scale=None, # For Swin MSA, not relevant for LSNetOriginalBlock
                 drop_rate=0., # LSNetOriginalBlock might have its own dropouts if any
                 attn_drop_rate=0., # For Swin MSA
                 drop_path_rate=0., # LSNetOriginalBlock might have its own dropouts if any
                 act_cfg=dict(type='GELU'), # LSNetOriginalBlock uses nn.ReLU by default in FFN
                 norm_cfg=dict(type='LN'),   # LSNetOriginalBlock uses BatchNorm
                 with_cp=False,
                 init_cfg=None):

        super(LocalBlock, self).__init__()
        # self._norm_layer = nn.BatchNorm2d # LSNetOriginalBlock uses its own Conv2d_BN
        self.init_cfg = init_cfg
        self.with_cp = with_cp # Checkpoint for the entire sequence if used
        self.window_size = window_size # Still needed for window partitioning

        # 'blocks' parameter from init (number of internal blocks for LSNet style)
        # 'embed_dims' is the channel dimension for these blocks
        self.layer = self._make_layer(embed_dims=embed_dims, num_internal_blocks=blocks)

    def forward(self, x, hw_shape, keep_token_indices):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad = query.shape[1]
        W_pad = query.shape[2]

        query_windows = self.window_partition(query) # Output: (num_windows*B, window_size, window_size, C)
        # Reshape to (num_windows*B, C, window_size, window_size) for conv layers if needed by window_partition output
        # Current window_partition output (based on Swin): (num_windows*B, window_size, window_size, C)
        # Let's assume query_windows is (num_windows*B, self.window_size**2, C)
        # The original SparseNet code for LocalBlock reshapes it to (B, -1, win, win, C)
        # then gathers, then permutes to (B*K, C, win, win)

        # Re-adapting the sparse gathering logic:
        query_windows = query_windows.view(B, -1, self.window_size, self.window_size, C) # (B, nW, win, win, C)
        K = keep_token_indices.shape[1]

        query_windows_sparse = torch.gather(query_windows, sparse_grad=True, dim=1,
                                            index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1,
                                                                                                self.window_size,
                                                                                                self.window_size,
                                                                                                C))
        # Permute and reshape for convolutional layers: (B*K, C, H_win, W_win)
        query_windows_sparse = query_windows_sparse.permute(0, 1, 4, 2, 3).reshape(B*K, C, self.window_size, self.window_size)

        if self.with_cp and query_windows_sparse.requires_grad:
            query_ffn = cp.checkpoint(self.layer, query_windows_sparse)
        else:
            query_ffn = self.layer(query_windows_sparse)

        # Reshape back before scatter
        query_ffn = query_ffn.reshape(B, K, C, self.window_size, self.window_size).permute(0, 1, 3, 4, 2) # (B, K, win, win, C)
        
        # Scatter operation
        x = query_windows.scatter(src=query_ffn, dim=1,
                                             index=keep_token_indices.view(B, K, 1, 1, 1).repeat(1, 1,
                                                                                                 self.window_size,
                                                                                                 self.window_size,
                                                                                                 C))

        x = x.view(-1, self.window_size, self.window_size, C) # (B*nW, win, win, C)
        x = self.window_reverse(x, H_pad, W_pad) # (B, H_pad, W_pad, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C) # (num_windows*B, win, win, C)
        return windows

    def window_reverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def _make_layer(
        self,
        embed_dims: int,
        num_internal_blocks: int,
    ) -> nn.Sequential:
        layers = []
        for i in range(num_internal_blocks):
            # Parameters for LSNetOriginalBlock:
            # ed, kd, nh=8, ar=4, resolution=14, stage=-1, depth=-1
            # We need to pass `ed` and `depth` (which is `i` here).
            # `stage` is set to -1 to ensure conv blocks are used, not MSA.
            # `kd` is a required positional argument.
            # `nh`, `ar`, `resolution` have defaults in LSNetOriginalBlock, but good to be explicit if kd is passed.
            layers.append(
                LSNetOriginalBlock(
                    ed=embed_dims,
                    kd=16,  # Provide a default value for key_dim (e.g., common value from LSNet config)
                    nh=4,   # Provide a default value for num_heads
                    ar=1.0, # Provide a default value for attn_ratio (can be float)
                    resolution=self.window_size, # Can use window_size as a proxy or a fixed default like 7 or 14
                    depth=i, # This ensures alternation of RepVGGDW and LSConv
                    stage=-1, # Ensures LSNetOriginalBlock doesn't try to use MSA
                )
            )
        return nn.Sequential(*layers)

class WindowMSA(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            attn += mask.unsqueeze(1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class GlobalBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads, # For BasicBlock, num_heads isn't directly used in its constructor
                 blocks, # Number of BasicBlocks in this GlobalBlock sequence
                 stage_id,
                 feedforward_channels, # For BasicBlock, FFN structure is fixed
                 qkv_bias=True, # Not for BasicBlock
                 qk_scale=None, # Not for BasicBlock
                 drop_rate=0., # Not for BasicBlock
                 attn_drop_rate=0., # Not for BasicBlock
                 drop_path_rate=0., # Not for BasicBlock (BasicBlock has no DropPath)
                 act_cfg=dict(type='GELU'), # BasicBlock uses nn.ReLU
                 norm_cfg=dict(type='LN'), # BasicBlock uses nn.BatchNorm2d
                 with_cp=False,
                 init_cfg=None):
        super(GlobalBlock, self).__init__()
        self._norm_layer = nn.BatchNorm2d # BasicBlock uses BatchNorm
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.stage_id = stage_id # Used to calculate inplanes for the first BasicBlock
        self.groups = 1 # BasicBlock default
        self.base_width = 64 # BasicBlock default
        # 'blocks' is the number of BasicBlock units for this GlobalBlock
        # 'embed_dims' is the 'planes' for BasicBlock
        self.layer = self._make_layer(BasicBlock, embed_dims, blocks)
        # print(self.layer) # Keep for debugging if needed

    def forward(self, x): # GlobalBlock in SparseNet takes (B, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.layer, x)
        else:
            x = self.layer(x)
        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1, # Default for GlobalBlock internal sequence
        dilate: bool = False, # Default for GlobalBlock internal sequence
    ) -> nn.Sequential:
        # Original SparseNet GlobalBlock inplanes logic:
        # self.inplanes = 64*(2**self.stage_id)
        # For DINO config with embed_dims=64, it looks like the first stage (stage_id=0) inplanes is also embed_dims (64)
        # and subsequent stages might increase. Let's use a more direct approach based on embed_dims for first block
        # and ensure subsequent blocks use planes * block.expansion
        current_inplanes = planes # Assume first block in GlobalBlock takes `planes` (embed_dims of stage) as input
                                # This simplifies matching with SparseNet config. Stage_id might be for ResNet-like scaling.
                                # Let's ensure self.inplanes is set correctly for the first block in sequence.
        
        # The `planes` argument here is the `embed_dims` of the current stage
        # For the first BasicBlock, `inplanes` should be `embed_dims` if no downsampling from a previous different layer.
        # If it's the very first GlobalBlock (stage_id=0), inplanes = embed_dims.
        # The original logic was `self.inplanes = 64*(2**self.stage_id)`. 
        # If embed_dims=64, then for stage_id=0, inplanes=64. For stage_id=1, inplanes=128 (planes for this block would be 128)
        # This implies `planes` passed to _make_layer is the target output channels for blocks in this stage.
        # The input to the *first* BasicBlock in this sequence needs to be correct.

        # Let's re-evaluate inplanes based on SparseNet structure for GlobalBlock
        # If GlobalBlock is the first operation in a stage after PatchEmbed or PatchMerging,
        # its input channel dimension is `planes` (which is `embed_dims` for that stage).
        self.inplanes = planes # Input to the first BasicBlock within this GlobalBlock

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = 1 # Dilation is not used in BasicBlock
        
        layers = []
        # First block might handle downsampling if stride != 1, or if inplanes changes
        # However, in SparseNet's GlobalBlock, stride is always 1 internally.
        # Downsampling (if any) is typically handled by PatchMerging *before* this BlockSequence.
        # The only case for downsample here is if self.inplanes != planes * block.expansion for the *first* block.
        # For BasicBlock, expansion is 1. So, if self.inplanes != planes.
        if self.inplanes != planes * block.expansion: # This condition is true if self.inplanes != planes for BasicBlock
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=1), # Stride is 1 for internal blocks
                norm_layer(planes * block.expansion),
            )
        else:
            downsample = None # Explicitly setting for clarity

        layers.append(
            block(
                self.inplanes, planes, stride=1, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=1, norm_layer=norm_layer
            )
        )
        self.inplanes = planes * block.expansion # Update inplanes for subsequent blocks
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=1, # BasicBlock doesn't use self.dilation
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

class BlockSequence(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 layers, # Number of internal blocks for GlobalBlock/LocalBlock
                 stage_id,
                 feedforward_channels,
                 depth, # Number of (GlobalBlock, LocalBlock) pairs in this sequence
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 top_k = 1.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.global_blocks = ModuleList()
        for i in range(depth):
            block_instance = GlobalBlock( # Renamed from 'block' to avoid conflict
                embed_dims=embed_dims,
                num_heads=num_heads, # Not directly used by BasicBlock based GlobalBlock
                blocks=layers, # 'layers' is num BasicBlocks for GlobalBlock
                stage_id=stage_id,
                feedforward_channels=feedforward_channels, # CORRECTED: Use the passed feedforward_channels
                # Pass other necessary params if GlobalBlock was changed to use them
                # Defaulting other params for BasicBlock compatibility for now
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg, norm_cfg=norm_cfg, # These might be overridden by BasicBlock defaults
                with_cp=with_cp,
                init_cfg=None)
            self.global_blocks.append(block_instance)

        self.local_blocks = ModuleList()
        for i in range(depth):
            block_instance = LocalBlock( # Renamed from 'block' to avoid conflict
                embed_dims=embed_dims,
                num_heads=num_heads, # Not directly used by LSNetOriginalBlock based LocalBlock
                blocks=layers,       # 'layers' is num LSNetOriginalBlocks for LocalBlock
                stage_id=stage_id,   # Not directly used
                feedforward_channels=feedforward_channels, # CORRECTED: Use the passed feedforward_channels
                window_size=window_size,
                # Defaulting other params as LocalBlock now uses LSNetOriginalBlock
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg, norm_cfg=norm_cfg, # LSNetOriginalBlock has its own defaults
                with_cp=with_cp,
                init_cfg=None)
            self.local_blocks.append(block_instance)
        
        self.pooling = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.downsample = downsample
        self.window_size = window_size
        # ScoreNet input dim needs to be C * window_size * window_size
        self.score_net = nn.Linear(embed_dims * (self.window_size**2), 1) 
        self.score_norm = build_norm_layer(norm_cfg, embed_dims)[1] # norm_cfg is LN for SparseNet
        self.top_k = top_k

    def forward(self, x, hw_shape):
        keep_token_indices, x_score, x_att_global = self.score_generator(x, hw_shape)
        x = x + x_att_global * (1-x_score)

        # Input to LocalBlock needs to be (B, L, C) where L=H*W
        # The current x is already in this shape from score_generator or previous stage
        for block_module in self.local_blocks: # Renamed loop variable
            # LocalBlock.forward expects x in (B, L, C) and hw_shape
            x = block_module(x, hw_shape, keep_token_indices)

        x_score = 1 + x_score - x_score.detach()
        x = x * x_score

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            # The output 'out' for norm and out_indices should be 'x' from *before* downsampling
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def score_generator(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        x_reshaped = x.view(B, H, W, C) # Changed from x to x_reshaped to avoid modifying input x directly here

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        x_padded = F.pad(x_reshaped, (0, 0, 0, pad_r, 0, pad_b)) # Use x_reshaped
        # Pooling expects (B, C, H, W)
        x_pooled = self.pooling(x_padded.permute(0, 3, 1, 2)) 

        H_pad, W_pad = x_padded.shape[1], x_padded.shape[2]

        x_mean = F.interpolate(x_pooled, size=(H_pad, W_pad), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        x_residual = x_padded - x_mean

        x_residual_windows = self.window_partition(x_residual) # Input (B, H_pad, W_pad, C)
        # window_partition output: (num_windows*B, window_size, window_size, C)
        x_residual_windows =  x_residual_windows.reshape(-1, self.window_size**2*C) # num_windows*B, win*win*C

        x_windows_score = self.score_net(x_residual_windows) # (num_windows*B, 1)
        x_windows_score = x_windows_score.view(B, -1) # (B, num_windows)

        x_windows_score = F.softmax(x_windows_score, dim=1) # (B, num_windows)

        # _, keep_token_indices = x_windows_score.topk(dim=1, k=int(self.top_k*(H_pad // self.window_size) * (W_pad // self.window_size)))
        # Ensure k is at least 1 and not greater than num_windows
        num_windows_total = (H_pad // self.window_size) * (W_pad // self.window_size)
        k_val = max(1, min(num_windows_total, int(self.top_k * num_windows_total)))
        _, keep_token_indices = x_windows_score.topk(dim=1, k=k_val) # (B, K)

        # Expand x_windows_score to match token dimensions (B, H*W, C)
        # Current x_windows_score is (B, nW). We need to map this back to (B, H_pad, W_pad, 1) then unpad.
        score_map_windows = x_windows_score.view(B, H_pad // self.window_size, W_pad // self.window_size, 1)
        # Upsample to pixel/token level score for each window
        x_score_expanded = score_map_windows.repeat_interleave(self.window_size, dim=1)\
                                        .repeat_interleave(self.window_size, dim=2) # (B, H_pad, W_pad, 1)
        x_score_expanded = x_score_expanded.repeat(1, 1, 1, C) # (B, H_pad, W_pad, C)

        if pad_r > 0 or pad_b > 0:
            x_score_final = x_score_expanded[:, :H, :W, :].contiguous().view(B, -1, C)
        else:
            x_score_final = x_score_expanded.contiguous().view(B, -1, C)

        # Global branch
        # query_global for GlobalBlock should be (B,C,H_pooled,W_pooled)
        query_global = x_pooled # Already (B, C, H_pooled, W_pooled)
        att_global = query_global
        for block_module in self.global_blocks: # Renamed loop variable
            att_global = block_module(att_global) # GlobalBlock takes (B,C,H,W)
        
        x_att_global_pooled = att_global # (B, C, H_pooled, W_pooled)
        x_att_global_expanded = F.interpolate(x_att_global_pooled, size=(H_pad, W_pad), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        
        if pad_r > 0 or pad_b > 0:
            x_att_global_final = x_att_global_expanded[:, :H, :W, :].contiguous().view(B, -1, C)
        else:
            x_att_global_final = x_att_global_expanded.contiguous().view(B, -1, C)

        return keep_token_indices, x_score_final, x_att_global_final

    def window_partition(self, x): # From BlockSequence, input (B,H,W,C)
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C) # (num_windows*B, win, win, C)
        return windows

    # window_reverse removed from BlockSequence as it's specific to LocalBlock's Swin-like handling if it were doing MSA

@MODELS.register_module()
class SparseNet(BaseModule):
    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 layers=(3, 4, 6, 3),
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 top_k=(0.7, 0.6, 0.5, 0.5),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'), # Default norm for SparseNet outputs
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SparseNet, self).__init__(init_cfg=init_cfg)

        num_stages = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth_sum = sum(depths) # Renamed from total_depth to avoid conflict with BlockSequence depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth_sum)
        ]

        self.stages = ModuleList()
        current_in_channels = embed_dims # Renamed from in_channels to avoid confusion
        for i in range(num_stages):
            if i < num_stages - 1:
                downsample_module = PatchMerging( # Renamed from downsample
                    in_channels=current_in_channels,
                    out_channels=2 * current_in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample_module = None

            stage_module = BlockSequence( # Renamed from stage
                embed_dims=current_in_channels,
                num_heads=num_heads[i],
                layers=layers[i],
                stage_id=i,
                feedforward_channels=mlp_ratio * current_in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                top_k = top_k[i],
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample_module,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage_module)
            if downsample_module:
                current_in_channels = downsample_module.out_channels

        self.num_features_stages = [int(embed_dims * 2**i) for i in range(num_stages)] # Renamed from num_features
        for i_idx in out_indices: # Renamed loop var
            layer = build_norm_layer(norm_cfg, self.num_features_stages[i_idx])[1]
            layer_name = f'norm{i_idx}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        super(SparseNet, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m_module in self.modules(): # Renamed from m
                if isinstance(m_module, nn.Linear):
                    trunc_normal_init(m_module, std=.02, bias=0.)
                elif isinstance(m_module, nn.LayerNorm):
                    constant_init(m_module.bias, 0)
                    constant_init(m_module.weight, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            # Adjust for potential backbone prefix
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v # Keep if no prefix

            # Strip further 'module.' prefix if DataParallel was used
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # Reshape absolute position embedding - check if self.use_abs_pos_embed
            if self.use_abs_pos_embed and state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L_load, C1 = absolute_pos_embed.size()
                # Current model's abs_pos_embed is (1, num_patches, embed_dims)
                N2, L_curr, C2 = self.absolute_pos_embed.size() 
                if N1 != N2 or C1 != C2 or L_load != L_curr:
                    logger.warning(f'Error in loading absolute_pos_embed due to shape mismatch. Loaded: {absolute_pos_embed.shape}, Current: {self.absolute_pos_embed.shape}. Skipping.')
                    del state_dict['absolute_pos_embed'] # Don't try to load it
                # else: # Shapes match, no reshape needed if it's already (1, L, C)
                    # state_dict['absolute_pos_embed'] = absolute_pos_embed # Not needed if no reshape
            elif not self.use_abs_pos_embed and state_dict.get('absolute_pos_embed') is not None:
                 del state_dict['absolute_pos_embed'] # Current model doesn't use it

            # Interpolate position bias table if needed (for Swin-like MSA, less relevant for pure conv LocalBlock)
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                # Check if the key exists in the current model's state_dict
                # This is important because our LocalBlock no longer uses Swin MSA
                if table_key not in self.state_dict():
                    logger.warning(f'{table_key} not found in current model, skipping interpolation for this key.')
                    # Remove from loaded state_dict as well if we are not using it, to avoid strict loading issues later if strict=True somewhere
                    if table_key in state_dict:
                        del state_dict[table_key]
                    continue 
                
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1_load, nH1 = table_pretrained.size()
                L2_curr, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key} (num_heads mismatch), pass')
                    del state_dict[table_key]
                elif L1_load != L2_curr:
                    S1 = int(L1_load**0.5)
                    S2 = int(L2_curr**0.5)
                    if S1*S1 != L1_load or S2*S2 != L2_curr: # Check if it's a perfect square
                        logger.warning(f'Error in loading {table_key} (length not a perfect square), pass')
                        del state_dict[table_key]
                        continue
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic', align_corners=False)
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2_curr).permute(1, 0).contiguous()

            # Load state_dict
            load_status = self.load_state_dict(state_dict, strict=False) # Use strict=False to ignore missing/unexpected keys
            logger.info(f'Weight loading status: {load_status}')

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        # The x from BlockSequence.forward is already (B, L, C)
        # The out_tensor for norm is the one *before* potential downsampling for next stage
        for i, stage_module in enumerate(self.stages): # Renamed from stage
            x, hw_shape, out_tensor_for_norm, out_hw_shape = stage_module(x, hw_shape) 
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                # Apply norm to out_tensor_for_norm, which is x *before* downsample
                out_tensor_for_norm = norm_layer(out_tensor_for_norm) 
                out_tensor_for_norm = out_tensor_for_norm.view(-1, *out_hw_shape,
                               self.num_features_stages[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out_tensor_for_norm)
        return outs


