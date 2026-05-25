"""
WTConvNeXt - 基于小波变换的ConvNeXt改进版本

基于timm v1.0.7:
https://github.com/huggingface/pytorch-image-models/blob/v1.0.7/timm/models/convnext.py
"""

from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# 导入timm库的相关模块
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import trunc_normal_, AvgPool2dSame, DropPath, Mlp, GlobalResponseNormMlp, \
    LayerNorm2d, LayerNorm, create_conv2d, get_act_layer, make_divisible, to_ntuple
from timm.layers import NormMlpClassifierHead, ClassifierHead
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

# 导入自定义的小波卷积模块
from wtconv import WTConv2d

__all__ = ['WTConvNeXt']  # 模型注册表将把每个入口函数添加到这个列表


class Downsample(nn.Module):
    """下采样模块

    通过池化+1x1卷积实现空间下采样和通道数调整
    """

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1

        # 1. 池化层：空间下采样
        if stride > 1 or dilation > 1:
            # 根据dilation选择池化类型
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()  # 无下采样时使用恒等映射

        # 2. 1x1卷积：通道数调整
        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)  # 1x1卷积调整通道
        else:
            self.conv = nn.Identity()  # 通道数不变时使用恒等映射

    def forward(self, x):
        x = self.pool(x)  # 空间下采样
        x = self.conv(x)  # 通道调整
        return x


class WTConvNeXtBlock(nn.Module):
    """WTConvNeXt基本块

    将传统ConvNeXt中的深度可分离卷积替换为WTConv2d（小波卷积）
    支持两种实现方式：
      (1) 通道优先格式：WTConv -> LayerNorm2d -> 1x1 Conv -> GELU -> 1x1 Conv
      (2) 通道最后格式：WTConv -> Permute -> LayerNorm -> Linear -> GELU -> Linear -> Permute

    Args:
        in_chs: 输入通道数
        out_chs: 输出通道数（None时等于in_chs）
        kernel_size: 深度卷积核大小，默认5
        stride: 步长，默认1
        dilation: 空洞率，支持int或tuple
        mlp_ratio: MLP扩展比率，默认4
        conv_mlp: 是否使用卷积MLP（通道优先格式）
        conv_bias: 卷积层是否使用偏置
        use_grn: 是否使用全局响应归一化（ConvNeXt-V2特性）
        ls_init_value: 层缩放初始化值，None时禁用层缩放
        act_layer: 激活函数类型
        norm_layer: 归一化层类型
        drop_path: 随机深度丢弃率
        wt_levels: WTConv的小波变换级数，控制感受野大小
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 5,
            stride: int = 1,
            dilation: Union[int, Tuple[int, int]] = (1, 1),
            mlp_ratio: float = 4,
            conv_mlp: bool = False,
            conv_bias: bool = True,
            use_grn: bool = False,
            ls_init_value: Optional[float] = 1e-6,
            act_layer: Union[str, Callable] = 'gelu',
            norm_layer: Optional[Callable] = None,
            drop_path: float = 0.,
            wt_levels: int = 0,  # 关键参数：小波变换级数
    ):
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)  # 确保dilation为2元组

        # 激活函数和归一化层配置
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            # 默认根据conv_mlp选择归一化层类型
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp)

        self.use_conv_mlp = conv_mlp

        # 1. 核心改进：用WTConv2d替换传统深度可分离卷积
        self.conv_dw = WTConv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            bias=conv_bias,
            wt_levels=wt_levels,  # 小波变换级数
        )

        # 2. 归一化层
        self.norm = norm_layer(out_chs)

        # 3. MLP层（两个1x1卷积或线性层）
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)

        # 4. 层缩放（Layer Scale）技术
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value is not None else None

        # 5. 快捷连接（残差连接）
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            # 需要下采样或调整通道数时使用Downsample
            self.shortcut = Downsample(in_chs, out_chs, stride=stride, dilation=dilation[0])
        else:
            self.shortcut = nn.Identity()  # 恒等映射

        # 6. 随机深度（Stochastic Depth）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 残差连接起点
        shortcut = x

        # 1. 小波卷积（核心操作）
        x = self.conv_dw(x)

        # 2. 归一化 + MLP
        if self.use_conv_mlp:
            # 通道优先格式（NCHW）：适合卷积操作
            x = self.norm(x)
            x = self.mlp(x)
        else:
            # 通道最后格式（NHWC）：适合线性层操作
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # 3. 层缩放
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))  # 逐通道缩放

        # 4. 残差连接 + 随机深度
        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class WTConvNeXtStage(nn.Module):
    """WTConvNeXt阶段模块

    包含多个WTConvNeXtBlock，可能包含下采样操作

    Args:
        in_chs: 输入通道数
        out_chs: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        depth: 该阶段的块数
        dilation: 空洞率
        drop_path_rates: 每个块的随机深度率
        ls_init_value: 层缩放初始化值
        conv_mlp: 是否使用卷积MLP
        conv_bias: 是否使用卷积偏置
        use_grn: 是否使用全局响应归一化
        act_layer: 激活函数
        norm_layer: 归一化层（通道优先）
        norm_layer_cl: 归一化层（通道最后）
        wt_levels: 小波变换级数
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=5,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            conv_bias=True,
            use_grn=False,
            act_layer='gelu',
            norm_layer=None,
            norm_layer_cl=None,
            wt_levels=0,  # 该阶段的小波变换级数
    ):
        super().__init__()
        self.grad_checkpointing = False  # 梯度检查点开关

        # 1. 下采样模块（如果需要）
        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0
            self.downsample = nn.Sequential(
                norm_layer(in_chs),  # 先归一化
                create_conv2d(  # 再卷积下采样
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()  # 无需下采样

        # 2. 构建该阶段的所有块
        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(WTConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                wt_levels=wt_levels,  # 传递小波级数参数
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        # 1. 下采样
        x = self.downsample(x)

        # 2. 通过该阶段的所有块（可能使用梯度检查点）
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)  # 梯度检查点节省显存
        else:
            x = self.blocks(x)
        return x


class WTConvNeXt(nn.Module):
    """WTConvNeXt主模型

    基于ConvNeXt架构，用WTConv2d替换传统深度可分离卷积
    论文: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

    主要改进：通过小波变换增强多尺度特征提取能力
    """

    def __init__(
            self,
            in_chans: int = 3,                    # 输入通道数（RGB图像为3）
            num_classes: int = 1000,               # 分类类别数
            global_pool: str = 'avg',             # 全局池化类型
            output_stride: int = 32,              # 输出步长（8,16,32）
            depths: Tuple[int, ...] = (3, 3, 9, 3),  # 4个阶段的块数
            dims: Tuple[int, ...] = (96, 192, 384, 768),  # 4个阶段的通道数
            kernel_sizes: Union[int, Tuple[int, ...]] = 5,  # 卷积核大小
            ls_init_value: Optional[float] = 1e-6,  # 层缩放初始化值
            stem_type: str = 'patch',             # 词干类型：'patch'或'overlap'
            patch_size: int = 4,                  # 词干patch大小
            head_init_scale: float = 1.,          # 分类头初始化缩放
            head_norm_first: bool = False,        # 是否先归一化再池化
            head_hidden_size: Optional[int] = None,  # 分类头隐藏层大小
            conv_mlp: bool = False,               # 是否使用卷积MLP
            conv_bias: bool = True,               # 是否使用卷积偏置
            use_grn: bool = False,                # 是否使用全局响应归一化
            act_layer: Union[str, Callable] = 'gelu',  # 激活函数
            norm_layer: Optional[Union[str, Callable]] = None,  # 归一化层
            norm_eps: Optional[float] = None,     # 归一化层epsilon
            drop_rate: float = 0.,                # 分类头dropout率
            drop_path_rate: float = 0.,           # 随机深度率
            wt_levels: Tuple[int, ...] = (5, 4, 3, 2),  # 关键：4个阶段的小波级数
    ):
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)  # 扩展为4个阶段的核大小

        # 1. 归一化层配置
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert conv_mlp, '如果指定norm_layer，必须使用conv_mlp'
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        # 基础参数
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []  # 特征图信息记录

        # 2. 词干（Stem）模块
        assert stem_type in ('patch', 'overlap', 'overlap_tiered')
        if stem_type == 'patch':
            # Patch词干：类似ViT的Patch Embedding
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size  # 词干下采样倍数
        else:
            # Overlap词干：重叠卷积词干
            mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
                nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
                norm_layer(dims[0]),
            )
            stem_stride = 4  # 两次2倍下采样

        # 3. 构建4个阶段
        self.stages = nn.Sequential()
        # 随机深度率线性衰减
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        stages = []
        prev_chs = dims[0]  # 上一阶段通道数
        curr_stride = stem_stride  # 当前总下采样倍数
        dilation = 1  # 当前空洞率

        # 构建4个特征提取阶段
        for i in range(4):
            # 计算当前阶段的下采样策略
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride  # 达到目标输出步长后使用空洞卷积
                stride = 1
            curr_stride *= stride  # 更新总下采样倍数

            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]  # 当前阶段输出通道数

            # 创建阶段模块
            stages.append(WTConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes[i],
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                conv_bias=conv_bias,
                use_grn=use_grn,
                act_layer=act_layer,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                wt_levels=wt_levels[i],  # 关键：每个阶段使用不同的小波级数
            ))
            prev_chs = out_chs
            # 记录特征图信息（用于特征金字塔等任务）
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)
        self.num_features = self.head_hidden_size = prev_chs

        # 4. 分类头配置
        if head_norm_first:
            # 先归一化再池化（大多数网络的标准顺序）
            assert not head_hidden_size
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            # 先池化再归一化（ConvNeXt原始顺序）
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
                act_layer='gelu',
            )
            self.head_hidden_size = self.head.num_features

        # 5. 权重初始化
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        """不进行权重衰减的参数（小波缩放参数）"""
        return {k for k,_ in self.named_parameters() if 'wavelet_scale' in k or 'base_scale' in k}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        """参数分组匹配器（用于优化器分组）"""
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # 下采样层
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),  # 块层
                (r'^norm_pre', (99999,))  # 最终归一化
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """设置梯度检查点（节省显存）"""
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        """获取分类器"""
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        """重置分类器（用于迁移学习）"""
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(self, x: torch.Tensor, indices: Optional[Union[int, List[int], Tuple[int]]] = None,
                            norm: bool = False, stop_early: bool = False, output_fmt: str = 'NCHW',
                            intermediates_only: bool = False):
        """前向传播并返回中间特征（用于特征提取）"""
        assert output_fmt in ('NCHW',), '输出格式必须是NCHW'
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

        # 前向传播并收集中间特征
        feat_idx = 0  # 词干是索引0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        # 选择要处理的范围
        if torch.jit.is_scripting() or not stop_early:
            stages = self.stages
        else:
            stages = self.stages[:max_index]

        # 通过各阶段
        for stage in stages:
            feat_idx += 1
            x = stage(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates  # 只返回中间特征

        x = self.norm_pre(x)
        return x, intermediates  # 返回最终特征和中间特征

    def prune_intermediate_layers(self, indices: Union[int, List[int], Tuple[int]] = 1,
                                prune_norm: bool = False, prune_head: bool = True):
        """剪枝中间层（用于模型压缩）"""
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)
        self.stages = self.stages[:max_index]  # 截断阶段
        if prune_norm:
            self.norm_pre = nn.Identity()
        if prune_head:
            self.reset_classifier(0, '')  # 移除分类头
        return take_indices

    def forward_features(self, x):
        """前向特征提取（不含分类头）"""
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        """前向分类头"""
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        """完整前向传播"""
        x = self.forward_features(x)  # 特征提取
        x = self.forward_head(x)      # 分类
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    """权重初始化函数"""
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)  # 截断正态分布初始化
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # 偏置初始化为0
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        nn.init.zeros_(module.bias)
        if name and 'head.' in name:
            # 分类头权重特殊初始化
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """检查点过滤函数（兼容不同格式的预训练权重）"""
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # 非Facebook格式的检查点

    if 'model' in state_dict:
        state_dict = state_dict['model']  # 提取模型权重

    # Facebook检查点格式转换
    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        # 转换CLIP格式的权重
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    # 正则表达式替换键名
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


def _create_wtconvnext(variant, pretrained=False, **kwargs):
    """创建WTConvNeXt模型的工厂函数"""
    if kwargs.get('pretrained_cfg', '') == 'fcmae':
        # FCMAE预训练权重没有分类头和最终归一化层
        kwargs.setdefault('pretrained_strict', False)

    model = build_model_with_cfg(
        WTConvNeXt, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,  # 使用自定义的检查点过滤
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),  # 特征提取配置
        **kwargs
    )
    return model


def _cfg(url='', **kwargs):
    """模型配置函数"""
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


# ==================== 模型注册部分 ====================

@register_model
def wtconvnext_tiny(pretrained=False, **kwargs) -> WTConvNeXt:
    """WTConvNeXt-Tiny模型（小模型）"""
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
    model = _create_wtconvnext('wtconvnext_tiny', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def wtconvnext_small(pretrained=False, **kwargs) -> WTConvNeXt:
    """WTConvNeXt-Small模型（中小模型）"""
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768])
    model = _create_wtconvnext('wtconvnext_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def wtconvnext_base(pretrained=False, **kwargs) -> WTConvNeXt:
    """WTConvNeXt-Base模型（基础模型）"""
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = _create_wtconvnext('wtconvnext_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model