#MMSegmentation（OpenMMLab语义分割框架）的配置文件，用于定义基于WTConvNeXt的语义分割模型

# 1. 基础配置
norm_cfg = dict(type='SyncBN', requires_grad=True)  # 同步BatchNorm，支持分布式训练
custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
checkpoint_file = './model_base_mmseg.pth'  # 预训练权重文件
# 2. 数据预处理
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],  # ImageNet均值
    std=[58.395, 57.12, 57.375],     # ImageNet标准差
    bgr_to_rgb=True,                  # BGR转RGB（OpenCV格式）
    pad_val=0,                        # 图像填充值
    seg_pad_val=255)                   # 分割标签填充值
# 3. 模型架构（Encoder-Decoder）
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
# 4. 主干网络（Backbone）- WTConvNeXt
    backbone=dict(
        type='WTConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],   #多尺度特征提取，输出4个阶段的特征图
        drop_path_rate=0.4,   # DropPath比率（正则化）
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
# 5. 解码头（UPerHead）- 主流语义分割头
    decode_head=dict(
        type='UPerHead',        # Unified Perceptual Parsing Head用于目标检测和语义分割任务的解耦式头部结构
        in_channels=[128, 256, 512, 1024],  # 输入通道数（对应4个阶段）
        in_index=[0, 1, 2, 3],  # 使用所有4个阶段的特征
        pool_scales=(1, 2, 3, 6),  # 金字塔池化尺度
        channels=512,           # 解码头通道数
        dropout_ratio=0.1,      # Dropout比率
        num_classes=19,         # 分割类别数（Cityscapes数据集）
        norm_cfg=norm_cfg,      # 使用SyncBN同步批归一化
        align_corners=False,    # 上采样对齐方式
        loss_decode=dict(       # 交叉熵损失函数
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
# 6. 辅助头（Auxiliary Head）- 辅助训练
    auxiliary_head=dict(
        type='FCNHead',         # 全卷积网络头
        in_channels=384,        # 输入通道（第3阶段特征）
        in_index=2,             # 使用第3阶段特征
        channels=256,           # 中间通道数
        num_convs=1,            # 卷积层数
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,         # 同样19个类别
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),   # 辅助损失权重0.4
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
