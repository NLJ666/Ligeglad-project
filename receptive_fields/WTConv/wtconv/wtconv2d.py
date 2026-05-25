import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import wavelet  # 导入小波变换工具模块


class WTConv2d(nn.Module):
    """基于2D小波变换的卷积层

    将传统2D卷积与小波变换结合，在空域和频域同时进行特征提取
    适用于图像处理、计算机视觉任务，特别适合需要多尺度分析的任务

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数（必须等于in_channels，深度可分离卷积）
        kernel_size (int): 卷积核大小，默认5
        stride (int): 步长，默认1
        bias (bool): 是否使用偏置，默认True
        wt_levels (int): 小波变换级数，控制感受野大小，默认1级
        wt_type (str): 小波基类型，如'db1'、'haar'等，默认'db1'
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        # 验证输入输出通道数相等（深度可分离卷积要求）
        assert in_channels == out_channels

        # 基础参数设置
        self.in_channels = in_channels  # 输入通道数
        self.wt_levels = wt_levels  # 小波变换级数（控制感受野大小）
        self.stride = stride  # 卷积步长
        self.dilation = 1  # 空洞率

        # 1. 创建2D小波滤波器（分解和重构）
        # wt_filter: 用于小波分解（空域→频域）
        # iwt_filter: 用于小波重构（频域→空域）
        self.wt_filter, self.iwt_filter = wavelet.create_2d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float)

        # 将滤波器注册为不可训练参数（固定小波基）
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 2. 基础卷积路径（传统2D卷积）
        # 深度可分离卷积：每个通道独立卷积
        self.base_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding='same',  # 保持空间尺寸不变
            stride=1,  # 步长1，不进行下采样
            dilation=1,  # 无空洞卷积
            groups=in_channels,  # 深度可分离：每个通道独立卷积
            bias=bias  # 是否使用偏置
        )
        # 可学习的缩放参数，增强表示能力
        # 维度：[1, in_channels, 1, 1] - 每个通道独立的缩放因子
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 3. 小波卷积路径（核心创新）
        # 为每个小波变换级创建独立的卷积层
        self.wavelet_convs = nn.ModuleList([
            # 每个小波级处理4倍通道数（LL, LH, HL, HH四个子带）
            nn.Conv2d(
                in_channels * 4, in_channels * 4, kernel_size,
                padding='same',  # 保持空间尺寸
                stride=1,  # 步长1
                dilation=1,  # 无空洞
                groups=in_channels * 4,  # 分组卷积
                bias=False  # 无偏置（减少参数）
            ) for _ in range(self.wt_levels)  # 为每级小波创建卷积
        ])

        # 每级小波卷积的可学习缩放参数
        # 初始缩放0.1，让小波路径初始贡献较小
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        # 4. 步长处理（下采样）
        if self.stride > 1:
            # 使用平均池化进行下采样
            # kernel_size=1 表示使用1x1池化，仅通过stride控制下采样
            self.do_stride = nn.AvgPool2d(
                kernel_size=1,  # 1x1池化核
                stride=stride  # 下采样倍数
            )
        else:
            self.do_stride = None  # 步长1时无需下采样

    def forward(self, x):
        """前向传播：双路径特征提取（基础卷积 + 小波卷积）

        Args:
            x (Tensor): 输入张量，形状为 [batch, channels, height, width]

        Returns:
            Tensor: 输出张量，形状为 [batch, channels, height//stride, width//stride]
        """

        # ==================== 小波路径：多级分解-卷积-重构 ====================

        # 存储各级分解结果
        x_ll_in_levels = []  # 存储各级处理后的低频分量（LL子带）
        x_h_in_levels = []  # 存储各级处理后的高频分量（LH, HL, HH子带）
        shapes_in_levels = []  # 存储各级输入形状（用于重构时裁剪）

        curr_x_ll = x  # 当前处理的低频信号（初始为输入x）

        # 前向：多级小波分解 + 频域卷积
        for i in range(self.wt_levels):
            # 记录当前形状（用于后续重构）
            curr_shape = curr_x_ll.shape  # [batch, channels, height, width]
            shapes_in_levels.append(curr_shape)

            # 边界处理：确保高和宽为偶数（小波变换要求）
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                # 在右侧和底部补零使高和宽为偶数
                # 填充格式：(左, 右, 上, 下)
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 2D小波分解：将图像分解为4个子带
            # 输入: [batch, channels, height, width]
            # 输出: [batch, channels, 4, height//2, width//2]
            #   其中：[:,:,0,:,:] = LL子带（低频近似）
            #         [:,:,1,:,:] = LH子带（水平细节）
            #         [:,:,2,:,:] = HL子带（垂直细节）
            #         [:,:,3,:,:] = HH子带（对角细节）
            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)

            # 提取LL低频子带用于下一级分解
            curr_x_ll = curr_x[:, :, 0, :, :]  # LL子带 [batch, channels, height//2, width//2]

            # 对频域信号进行卷积处理
            shape_x = curr_x.shape  # [batch, channels, 4, height//2, width//2]

            # 重塑为卷积可处理格式：[batch, channels*4, height//2, width//2]
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])

            # 频域卷积 + 可学习缩放
            curr_x_tag = self.wavelet_convs[i](curr_x_tag)  # 卷积处理
            curr_x_tag = self.wavelet_scale[i](curr_x_tag)  # 可学习缩放

            # 恢复为频带格式：[batch, channels, 4, height//2, width//2]
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 存储处理后的频带分量
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # 处理后的LL低频
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # 处理后的高频子带（LH, HL, HH）

        # 反向：多级小波重构（从最深级向最浅级）
        next_x_ll = 0  # 初始重构信号（从0开始累加）

        for i in range(self.wt_levels - 1, -1, -1):
            # 弹出当前级的处理结果（后进先出）
            curr_x_ll = x_ll_in_levels.pop()  # 当前级处理后的LL低频
            curr_x_h = x_h_in_levels.pop()  # 当前级处理后的高频子带
            curr_shape = shapes_in_levels.pop()  # 当前级原始形状

            # 融合：当前级LL低频 + 上一级重构结果
            curr_x_ll = curr_x_ll + next_x_ll

            # 重组频带：将LL低频和高频子带拼接
            # [batch, channels, 1, height//2, width//2] + [batch, channels, 3, height//2, width//2]
            # = [batch, channels, 4, height//2, width//2]
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            # 小波逆变换：频域→空域重构
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            # 裁剪到原始尺寸（去除填充）
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        # 小波路径的最终输出
        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0  # 验证所有级都已处理

        # ==================== 基础卷积路径 ====================
        x_base = self.base_conv(x)  # 传统2D卷积
        x_base = self.base_scale(x_base)  # 可学习缩放

        # ==================== 双路径融合 ====================
        # 残差连接：基础卷积路径 + 小波卷积路径
        x = x_base + x_tag

        # ==================== 下采样处理 ====================
        if self.do_stride is not None:
            x = self.do_stride(x)  # 平均池化下采样

        return x


class _ScaleModule(nn.Module):
    """可学习缩放模块

    对输入张量进行逐通道的可学习缩放，增强表示能力

    Args:
        dims (list): 权重张量的维度，如 [1, channels, 1, 1]
        init_scale (float): 初始缩放值，默认1.0
        init_bias (float): 初始偏置（本实现未使用）
    """

    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        # 可学习的缩放权重，初始化为指定值
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None  # 本实现未使用偏置

    def forward(self, x):
        """前向传播：元素级缩放

        Args:
            x (Tensor): 输入张量

        Returns:
            Tensor: 缩放后的张量，x * weight
        """
        return torch.mul(self.weight, x)  # 逐元素乘法