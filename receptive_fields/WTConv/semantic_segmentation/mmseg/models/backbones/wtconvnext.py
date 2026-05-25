# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
from mmpretrain.models.utils import GRN, build_norm_layer
from mmpretrain.models.backbones.base_backbone import BaseBackbone

import torch.nn.functional as F
import pywt  # 小波变换库


class WTConvNeXtBlock(BaseModule):
    """WTConvNeXt块 - 结合小波变换的改进版ConvNeXt块

    Args:
        in_channels (int): 输入通道数
        dw_conv_cfg (dict): 深度可分离WTConv卷积配置
            Defaults to ``dict(kernel_size=5, stride=1, bias=True, wt_levels=2)``
        norm_cfg (dict): 归一化层配置
        act_cfg (dict): 激活函数配置
        mlp_ratio (float): MLP扩展比率
        linear_pw_conv (bool): 是否使用线性层进行逐点卷积
        drop_path_rate (float): 随机深度比率
        layer_scale_init_value (float): 层缩放初始化值
        use_grn (bool): 是否使用全局响应归一化
        with_cp (bool): 是否使用梯度检查点
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=5, stride=1, bias=True, wt_levels=2),  #小波级数变换级数
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        # 1. 深度可分离小波卷积（核心改进）
        self.depthwise_conv = _WTConv2d(
            in_channels, in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        # 2. 逐点卷积（MLP部分）
        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # 使用线性层进行逐点卷积（通道最后格式）
            pw_conv = nn.Linear
        else:
            # 使用1x1卷积进行逐点卷积
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)  # 扩展
        self.act = MODELS.build(act_cfg)  # GELU激活
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)  # 压缩

        # 3. 全局响应归一化（ConvNeXt V2特性）
        if use_grn:
            self.grn = GRN(mid_channels)#基因调控网络
        else:
            self.grn = None

        # 4. 层缩放（Layer Scale）技术
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        # 5. 随机深度（Stochastic Depth）
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        """前向传播"""

        def _inner_forward(x):
            # 残差连接
            shortcut = x

            # 1. 深度可分离小波卷积
            x = self.depthwise_conv(x)

            # 2. 逐点卷积部分
            if self.linear_pw_conv:
                # (N, C, H, W) -> (N, H, W, C) 通道最后格式
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x, data_format='channel_last')  # 层归一化
                x = self.pointwise_conv1(x)  # 扩展
                x = self.act(x)  # 激活
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')  # GRN
                x = self.pointwise_conv2(x)  # 压缩
                # (N, H, W, C) -> (N, C, H, W) 恢复格式
                x = x.permute(0, 3, 1, 2)
            else:
                # 通道优先格式的版本
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            # 3. 层缩放
            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            # 4. 残差连接 + 随机深度
            x = shortcut + self.drop_path(x)
            return x

        # 梯度检查点（节省显存）
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class WTConvNeXt(BaseBackbone):
    """WTConvNeXt v1&v2主干网络

    结合小波变换的改进版ConvNeXt，支持多级小波分解
    """

    # 模型架构配置
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],  # 4个阶段的块数
            'channels': [96, 192, 384, 768],  # 4个阶段的通道数
            'wt_levels': [5, 4, 3, 2]  # 4个阶段的小波变换级数
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768],
            'wt_levels': [5, 4, 3, 2]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024],
            'wt_levels': [5, 4, 3, 2]
        },
    }

    def __init__(self,
                 arch='tiny',  # 架构名称
                 in_channels=3,  # 输入通道
                 stem_patch_size=4,  # 词干层patch大小
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,  # 线性逐点卷积
                 use_grn=False,  # 全局响应归一化
                 drop_path_rate=0.,  # 随机深度比率
                 layer_scale_init_value=1e-6,  # 层缩放初始化
                 out_indices=-1,  # 输出阶段索引
                 frozen_stages=0,  # 冻结阶段数
                 gap_before_final_norm=True,  # 最终归一化前全局池化
                 with_cp=False,  # 梯度检查点
                 init_cfg=[...]):  # 初始化配置
        super().__init__(init_cfg=init_cfg)

        # 1. 架构配置解析
        if isinstance(arch, str):
            assert arch in self.arch_settings
            arch = self.arch_settings[arch]

        self.depths = arch['depths']  # 各阶段块数
        self.channels = arch['channels']  # 各阶段通道数
        self.wt_levels = arch['wt_levels']  # 各阶段小波级数
        self.num_stages = len(self.depths)  # 阶段数（通常为4）

        # 2. 输出索引处理
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # 3. 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0

        # 4. 下采样层（包含词干层）
        self.downsample_layers = ModuleList()

        # 词干层：将图像分割为patch
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],  # 输出到第1阶段通道数
                kernel_size=stem_patch_size,
                stride=stem_patch_size),  # 4x4卷积，步长4
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 5. 构建4个特征阶段
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]  # 当前阶段块数
            channels = self.channels[i]  # 当前阶段通道数
            wt_levels = self.wt_levels[i]  # 当前阶段小波级数

            # 阶段间的下采样（第2、3、4阶段）
            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,  # 2x2卷积
                        stride=2),  # 下采样2倍
                )
                self.downsample_layers.append(downsample_layer)

            # 构建当前阶段的所有块
            stage = Sequential(*[
                WTConvNeXtBlock(
                    in_channels=channels,
                    dw_conv_cfg=dict(kernel_size=5, stride=1, bias=True, wt_levels=wt_levels),
                    drop_path_rate=dpr[block_idx + j],  # 递增的随机深度
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth
            self.stages.append(stage)

            # 为输出阶段添加归一化层
            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()  # 冻结指定阶段

    def forward(self, x):
        """前向传播"""
        outs = []
        for i, stage in enumerate(self.stages):
            # 1. 下采样
            x = self.downsample_layers[i](x)
            # 2. 通过当前阶段
            x = stage(x)
            # 3. 如果是输出阶段，处理输出
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    # 全局平均池化 + 归一化（分类任务）
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # 直接归一化（分割任务）
                    outs.append(norm_layer(x))

        return tuple(outs)  # 返回多尺度特征

    def _freeze_stages(self):
        """冻结指定阶段参数"""
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), stage.parameters()):
                param.requires_grad = False  # 冻结梯度

    def train(self, mode=True):
        """训练模式设置"""
        super(WTConvNeXt, self).train(mode)
        self._freeze_stages()  # 保持冻结阶段状态


class _WTConv2d(nn.Module):
    """小波变换卷积层 - 核心创新点
    将传统卷积替换为多级小波变换，增强频域特征提取能力
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 bias=True, wt_levels=1, wt_type='db1'):
        super(_WTConv2d, self).__init__()

        assert in_channels == out_channels  # 深度可分离要求

        self.in_channels = in_channels
        self.wt_levels = wt_levels  # 小波变换级数,控制感受野大小
        self.stride = stride

        # 1. 创建小波滤波器（分解和重构）
        self.wt_filter, self.iwt_filter = _create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)  # 固定滤波器
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 2. 小波变换函数
        self.wt_function = partial(_wavelet_transform, filters=self.wt_filter)  # 分解
        self.iwt_function = partial(_inverse_wavelet_transform, filters=self.iwt_filter)  # 重构

        # 3. 基础卷积路径（低频分量）
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding='same', stride=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])  # 可学习缩放

        # 4. 小波卷积路径（高频分量）- 多级处理
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                      padding='same', stride=1, groups=in_channels * 4, bias=False)
            for _ in range(self.wt_levels)  # 每级小波变换一个卷积
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)  # 小权重初始化
            for _ in range(self.wt_levels)
        ])

        # 5. 步长处理
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None,
                                                   stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        """前向传播：多级小波分解 + 卷积 + 重构"""

        # 存储各级的低频和高频分量
        x_ll_in_levels = []  # 低频分量（LL）
        x_h_in_levels = []  # 高频分量（LH, HL, HH）
        shapes_in_levels = []  # 各层形状

        # 1. 多级小波分解（前向）
        curr_x_ll = x  # 当前低频分量
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            # 边界处理（确保偶数尺寸）
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波分解：LL, LH, HL, HH
            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]  # 提取LL分量用于下一级分解

            # 高频分量卷积处理
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 存储处理后的分量
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # 处理后的LL
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # 处理后的高频

        next_x_ll = 0  # 初始重构分量

        # 2. 多级小波重构（反向）
        for i in range(self.wt_levels - 1, -1, -1):
            # 弹出当前级的分量
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # 融合低频分量（来自上一级重构）
            curr_x_ll = curr_x_ll + next_x_ll

            # 重组分量并重构
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)  # 小波逆变换

            # 裁剪到原始尺寸
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll  # 最终的小波路径输出
        assert len(x_ll_in_levels) == 0  # 确保所有级都已处理

        # 3. 基础卷积路径
        x = self.base_scale(self.base_conv(x))

        # 4. 双路径融合：基础卷积 + 小波卷积
        x = x + x_tag

        # 5. 步长下采样（如果需要）
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    """可学习缩放模块"""

    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)  # 元素级缩放


def _create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """创建小波分解和重构滤波器"""
    w = pywt.Wavelet(wave)  # 创建小波（如db1）

    # 分解滤波器（分析滤波器）
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # 高通
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  # 低通
    # 构建2D滤波器组：LL, LH, HL, HH
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),  # LL: 低通×低通
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),  # LH: 低通×高通
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),  # HL: 高通×低通
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  # HH: 高通×高通
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)  # 扩展到输入通道数

    # 重构滤波器（合成滤波器）
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),  # LL
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),  # LH
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),  # HL
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)  # HH
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)  # 扩展到输出通道数

    return dec_filters, rec_filters


def _wavelet_transform(x, filters):
    """小波变换：将图像分解为4个子带"""
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)  # 填充
    # 使用卷积实现小波分解（步长2下采样）
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    # 重排为(b, c, 4, h/2, w/2)格式：4个子带
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def _inverse_wavelet_transform(x, filters):
    """逆小波变换：从4个子带重构图像"""
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # 展平子带维度
    x = x.reshape(b, c * 4, h_half, w_half)
    # 使用转置卷积实现小波重构（步长2上采样）
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x