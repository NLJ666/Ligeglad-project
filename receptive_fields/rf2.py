import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns

# 设置中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10


# 定义感受野CNN模型类
class ReceptiveFieldCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ReceptiveFieldCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 感受野计算函数
def calculate_receptive_field(layers):
    """
    计算卷积神经网络的感受野
    layers: 每层的配置列表，例如 [{'kernel_size': 3, 'stride': 1, 'padding': 1}, ...]
    """
    rf = 1
    stride_product = 1
    for layer in reversed(layers):
        kernel_size = layer.get('kernel_size', 1)
        stride = layer.get('stride', 1)
        rf = rf + (kernel_size - 1) * stride_product
        stride_product *= stride
    return rf


# 计算逐层感受野变化
def calculate_layer_rf():
    """
    计算网络每一层的感受野
    返回：(layer_names, rf_sizes, feature_sizes, channels)
    """
    # 定义各层的感受野计算
    layers = [
        {'name': '输入', 'rf': 1, 'channels': 3, 'size': 32, 'color': '#FF6B6B'},
        {'name': 'Conv1+Pool', 'rf': 4, 'channels': 32, 'size': 16, 'color': '#4ECDC4'},
        {'name': 'Conv2+Pool', 'rf': 10, 'channels': 64, 'size': 8, 'color': '#45B7D1'},
        {'name': 'Conv3+Pool', 'rf': 22, 'channels': 128, 'size': 4, 'color': '#96CEB4'},
    ]

    return layers


# 可视化感受野增长过程
def visualize_receptive_field_growth():
    """
    可视化感受野增长过程
    """
    layers = calculate_layer_rf()

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('感受野CNN可视化分析', fontsize=16, fontweight='bold')

    # 1. 感受野增长趋势
    ax1 = axes[0, 0]
    rf_sizes = [layer['rf'] for layer in layers]
    layer_names = [layer['name'] for layer in layers]

    ax1.plot(range(len(rf_sizes)), rf_sizes, 'o-', linewidth=3, markersize=10,
             markerfacecolor='white', markeredgewidth=2, markeredgecolor='#2C3E50')
    ax1.set_xlabel('网络层', fontweight='bold')
    ax1.set_ylabel('感受野大小', fontweight='bold')
    ax1.set_title('感受野增长趋势', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names, rotation=45)

    # 添加数值标签
    for i, rf in enumerate(rf_sizes):
        ax1.text(i, rf + 0.5, f'{rf}x{rf}', ha='center', fontweight='bold')

    # 2. 特征图尺寸变化
    ax2 = axes[0, 1]
    feature_sizes = [layer['size'] for layer in layers]

    bars = ax2.bar(range(len(feature_sizes)), feature_sizes,
                   color=[layer['color'] for layer in layers],
                   edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('网络层', fontweight='bold')
    ax2.set_ylabel('特征图尺寸', fontweight='bold')
    ax2.set_title('特征图尺寸变化', fontweight='bold')
    ax2.set_xticks(range(len(layer_names)))
    ax2.set_xticklabels(layer_names, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, size in zip(bars, feature_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{size}x{size}', ha='center', fontweight='bold')

    # 3. 通道数变化
    ax3 = axes[0, 2]
    channels = [layer['channels'] for layer in layers]

    ax3.plot(range(len(channels)), channels, 's-', linewidth=3, markersize=10,
             markerfacecolor='white', markeredgewidth=2, markeredgecolor='#E74C3C',
             color='#E74C3C')
    ax3.set_xlabel('网络层', fontweight='bold')
    ax3.set_ylabel('通道数', fontweight='bold')
    ax3.set_title('特征通道数变化', fontweight='bold')
    ax3.set_xticks(range(len(layer_names)))
    ax3.set_xticklabels(layer_names, rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 添加数值标签
    for i, ch in enumerate(channels):
        ax3.text(i, ch * 1.2, f'{ch}', ha='center', fontweight='bold')

    # 4. 感受野可视化示意图
    ax4 = axes[1, 0]
    ax4.set_xlim(-5, 5)
    ax4.set_ylim(-5, 5)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('感受野增长示意图', fontweight='bold', pad=20)

    # 绘制感受野增长
    rf_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    rf_sizes = [layer['rf'] for layer in layers]

    for i, (rf, color) in enumerate(zip(rf_sizes, rf_colors)):
        # 绘制正方形表示感受野
        side = min(rf * 0.2, 4)  # 限制最大大小
        rect = Rectangle((-side / 2, -side / 2), side, side,
                         fill=True, alpha=0.7, color=color,
                         edgecolor='black', linewidth=2)
        ax4.add_patch(rect)

        # 添加文本
        ax4.text(0, side / 2 + 0.5, f'RF={rf}x{rf}',
                 ha='center', fontweight='bold', fontsize=9)

        # 绘制连接线
        if i < len(rf_sizes) - 1:
            ax4.annotate('', xy=(0, side / 2 + 0.7), xytext=(0, -side / 2 - 0.7),
                         arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # 5. 网络结构示意图
    ax5 = axes[1, 1]
    ax5.axis('off')
    ax5.set_title('网络结构示意图', fontweight='bold', pad=20)

    # 创建简单的网络图
    G = nx.DiGraph()

    # 添加节点
    positions = {
        'Input': (0, 3),
        'Conv1+BN': (0, 2.5),
        'Pool1': (0, 2),
        'Conv2+BN': (0, 1.5),
        'Pool2': (0, 1),
        'Conv3+BN': (0, 0.5),
        'Pool3': (0, 0),
        'Flatten': (0, -0.5),
        'FC1': (0, -1),
        'FC2': (0, -1.5),
    }

    # 添加边
    edges = [
        ('Input', 'Conv1+BN'),
        ('Conv1+BN', 'Pool1'),
        ('Pool1', 'Conv2+BN'),
        ('Conv2+BN', 'Pool2'),
        ('Pool2', 'Conv3+BN'),
        ('Conv3+BN', 'Pool3'),
        ('Pool3', 'Flatten'),
        ('Flatten', 'FC1'),
        ('FC1', 'FC2'),
    ]

    for edge in edges:
        G.add_edge(*edge)

    # 节点颜色
    node_colors = ['#FF6B6B', '#4ECDC4', '#4ECDC4', '#45B7D1',
                   '#45B7D1', '#96CEB4', '#96CEB4', '#FECA57',
                   '#FF9FF3', '#54A0FF']

    # 绘制网络
    nx.draw(G, positions, ax=ax5, with_labels=True, node_size=2000,
            node_color=node_colors, font_size=8, font_weight='bold',
            edge_color='gray', width=2, arrowsize=20,
            connectionstyle="arc3,rad=0.1")

    # 6. 参数量分析
    ax6 = axes[1, 2]

    # 计算各层参数量
    model = ReceptiveFieldCNN()
    param_counts = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_type = name.split('.')[0]
            if layer_type not in param_counts:
                param_counts[layer_type] = 0
            param_counts[layer_type] += param.numel()

    # 准备数据
    layer_names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'bn1', 'bn2', 'bn3']
    params = [param_counts.get(layer, 0) for layer in layer_names]

    # 只显示有参数量的层
    valid_indices = [i for i, p in enumerate(params) if p > 0]
    valid_layers = [layer_names[i] for i in valid_indices]
    valid_params = [params[i] for i in valid_indices]

    # 创建水平条形图
    y_pos = range(len(valid_layers))
    bars = ax6.barh(y_pos, valid_params, color='#5DADE2', edgecolor='black')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(valid_layers, fontweight='bold')
    ax6.set_xlabel('参数量', fontweight='bold')
    ax6.set_title('各层参数量分布', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for bar, param in zip(bars, valid_params):
        width = bar.get_width()
        ax6.text(width + max(valid_params) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{param:,}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


# 可视化单个神经元的感受野（修正后）
def visualize_single_neuron_rf():
    """
    可视化单个神经元感受野的构成
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('单个神经元感受野构成', fontsize=16, fontweight='bold')

    # 输入层 (32x32)
    ax1 = axes[0]
    input_grid = np.zeros((32, 32))

    # 随机生成一些激活
    np.random.seed(42)
    for i in range(32):
        for j in range(32):
            input_grid[i, j] = np.random.rand()

    im1 = ax1.imshow(input_grid, cmap='viridis', interpolation='nearest')
    ax1.set_title('输入层 (32x32x3)', fontweight='bold')
    ax1.set_xlabel('宽度')
    ax1.set_ylabel('高度')

    # 标注感受野
    rf_size = 4  # 修正为4
    center_x, center_y = 16, 16
    rect1 = Rectangle((center_x - rf_size // 2 - 0.5, center_y - rf_size // 2 - 0.5),
                      rf_size, rf_size, fill=False, edgecolor='red', linewidth=3)
    ax1.add_patch(rect1)
    ax1.text(center_x, center_y - rf_size // 2 - 3, f'RF={rf_size}x{rf_size}',
             ha='center', color='red', fontweight='bold')

    # Conv1+Pool 后 (16x16)
    ax2 = axes[1]
    conv1_grid = np.zeros((16, 16))

    # 模拟卷积和池化后的激活
    for i in range(16):
        for j in range(16):
            conv1_grid[i, j] = np.random.rand() * 0.8

    im2 = ax2.imshow(conv1_grid, cmap='plasma', interpolation='nearest')
    ax2.set_title('Conv1+Pool后 (16x16x32)', fontweight='bold')
    ax2.set_xlabel('宽度')
    ax2.set_ylabel('高度')

    # 标注感受野
    rf_size = 10  # 修正为10
    center_x, center_y = 8, 8
    rect2 = Rectangle((center_x - rf_size // 4 - 0.5, center_y - rf_size // 4 - 0.5),
                      rf_size / 2, rf_size / 2, fill=False, edgecolor='red', linewidth=3)
    ax2.add_patch(rect2)
    ax2.text(center_x, center_y - rf_size // 4 - 1.5, f'RF={rf_size}x{rf_size}',
             ha='center', color='red', fontweight='bold')

    # Conv2+Pool 后 (8x8)
    ax3 = axes[2]
    conv2_grid = np.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            conv2_grid[i, j] = np.random.rand() * 0.6

    im3 = ax3.imshow(conv2_grid, cmap='summer', interpolation='nearest')
    ax3.set_title('Conv2+Pool后 (8x8x64)', fontweight='bold')
    ax3.set_xlabel('宽度')
    ax3.set_ylabel('高度')

    # 标注感受野
    rf_size = 10  # 修正为10
    center_x, center_y = 4, 4
    rect3 = Rectangle((center_x - rf_size // 8 - 0.5, center_y - rf_size // 8 - 0.5),
                      rf_size / 4, rf_size / 4, fill=False, edgecolor='red', linewidth=3)
    ax3.add_patch(rect3)
    ax3.text(center_x, center_y - rf_size // 8 - 0.8, f'RF={rf_size}x{rf_size}',
             ha='center', color='red', fontweight='bold')

    # Conv3+Pool 后 (4x4)
    ax4 = axes[3]
    conv3_grid = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            conv3_grid[i, j] = np.random.rand() * 0.4

    im4 = ax4.imshow(conv3_grid, cmap='winter', interpolation='nearest')
    ax4.set_title('Conv3+Pool后 (4x4x128)', fontweight='bold')
    ax4.set_xlabel('宽度')
    ax4.set_ylabel('高度')

    # 标注感受野
    rf_size = 22  # 修正为22
    center_x, center_y = 2, 2
    rect4 = Rectangle((center_x - rf_size // 16 - 0.5, center_y - rf_size // 16 - 0.5),
                      rf_size / 8, rf_size / 8, fill=False, edgecolor='red', linewidth=3)
    ax4.add_patch(rect4)
    ax4.text(center_x, center_y - rf_size // 16 - 0.4, f'RF={rf_size}x{rf_size}',
             ha='center', color='red', fontweight='bold')

    plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
    plt.colorbar(im4, ax=axes[3], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# 创建综合报告
def create_comprehensive_report():
    """
    创建综合可视化报告
    """
    print("=" * 60)
    print("感受野CNN网络结构可视化报告（修正后）")
    print("=" * 60)

    # 创建模型
    model = ReceptiveFieldCNN()

    # 打印模型结构
    print("\n1. 模型结构:")
    print("-" * 40)
    print(model)

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. 参数量统计:")
    print("-" * 40)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 计算感受野
    layers_config = [
        {'kernel_size': 3, 'stride': 1, 'padding': 1},  # conv1
        {'kernel_size': 2, 'stride': 2, 'padding': 0},  # pool1
        {'kernel_size': 3, 'stride': 1, 'padding': 1},  # conv2
        {'kernel_size': 2, 'stride': 2, 'padding': 0},  # pool2
        {'kernel_size': 3, 'stride': 1, 'padding': 1},  # conv3
        {'kernel_size': 2, 'stride': 2, 'padding': 0},  # pool3
    ]

    receptive_field = calculate_receptive_field(layers_config)
    print(f"\n3. 感受野分析:")
    print("-" * 40)
    print(f"最终感受野大小: {receptive_field}x{receptive_field}")
    print(f"说明：最后一个池化层的每个神经元能看到输入图像的 {receptive_field}x{receptive_field} 区域")

    # 计算各层感受野
    print("\n4. 逐层感受野变化:")
    print("-" * 40)

    # 详细计算过程
    rf = 1
    stride_product = 1

    layer_configs = [
        {'name': '输入', 'k': 1, 's': 1},
        {'name': 'Conv1', 'k': 3, 's': 1},
        {'name': 'Pool1', 'k': 2, 's': 2},
        {'name': 'Conv2', 'k': 3, 's': 1},
        {'name': 'Pool2', 'k': 2, 's': 2},
        {'name': 'Conv3', 'k': 3, 's': 1},
        {'name': 'Pool3', 'k': 2, 's': 2},
    ]

    for i, layer in enumerate(layer_configs):
        if i > 0:  # 从第一层开始计算
            rf = rf + (layer['k'] - 1) * stride_product

        stride_product *= layer['s']

        if i == 1:  # Conv1后
            print(f"Conv1后      -> 感受野: {rf}x{rf}")
        elif i == 2:  # Pool1后
            print(f"Conv1+Pool后 -> 感受野: {rf}x{rf}")
        elif i == 4:  # Pool2后
            print(f"Conv2+Pool后 -> 感受野: {rf}x{rf}")
        elif i == 6:  # Pool3后
            print(f"Conv3+Pool后 -> 感受野: {rf}x{rf}")

    print("\n5. 特征图尺寸变化:")
    print("-" * 40)
    input_size = 32
    sizes = [input_size]
    for layer in layers_config:
        if layer['stride'] > 1:  # 有下采样
            sizes.append(sizes[-1] // layer['stride'])
        else:
            sizes.append(sizes[-1])

    layer_names = ['输入', 'Conv1', 'Pool1', 'Conv2', 'Pool2', 'Conv3', 'Pool3']
    for i, (name, size) in enumerate(zip(layer_names, sizes)):
        print(f"{name:10s} -> 尺寸: {size}x{size}")

    return model


# 可视化卷积核权重分布
def visualize_kernel_distribution():
    """
    可视化卷积核权重分布
    """
    model = ReceptiveFieldCNN()

    # 获取卷积核权重
    conv1_weights = model.conv1.weight.detach().cpu().numpy().flatten()
    conv2_weights = model.conv2.weight.detach().cpu().numpy().flatten()
    conv3_weights = model.conv3.weight.detach().cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('卷积核权重分布', fontsize=16, fontweight='bold')

    # Conv1 权重分布
    ax1 = axes[0]
    ax1.hist(conv1_weights, bins=50, alpha=0.7, color='#3498DB', edgecolor='black')
    ax1.set_xlabel('权重值', fontweight='bold')
    ax1.set_ylabel('频数', fontweight='bold')
    ax1.set_title('Conv1 权重分布', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=conv1_weights.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {conv1_weights.mean():.4f}')
    ax1.legend()

    # Conv2 权重分布
    ax2 = axes[1]
    ax2.hist(conv2_weights, bins=50, alpha=0.7, color='#2ECC71', edgecolor='black')
    ax2.set_xlabel('权重值', fontweight='bold')
    ax2.set_ylabel('频数', fontweight='bold')
    ax2.set_title('Conv2 权重分布', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=conv2_weights.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {conv2_weights.mean():.4f}')
    ax2.legend()

    # Conv3 权重分布
    ax3 = axes[2]
    ax3.hist(conv3_weights, bins=50, alpha=0.7, color='#9B59B6', edgecolor='black')
    ax3.set_xlabel('权重值', fontweight='bold')
    ax3.set_ylabel('频数', fontweight='bold')
    ax3.set_title('Conv3 权重分布', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=conv3_weights.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {conv3_weights.mean():.4f}')
    ax3.legend()

    plt.tight_layout()
    plt.show()


# 可视化感受野重叠
def visualize_receptive_field_overlap():
    """
    可视化感受野重叠情况
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('感受野重叠与覆盖', fontsize=16, fontweight='bold')

    # 输入图像
    img_size = 32
    input_img = np.random.randn(img_size, img_size)

    # 1. Conv1 层的感受野覆盖
    ax1 = axes[0]
    ax1.imshow(input_img, cmap='gray', interpolation='nearest')
    ax1.set_title('Conv1+Pool层感受野覆盖', fontweight='bold')
    ax1.set_xlabel('宽度')
    ax1.set_ylabel('高度')

    # 绘制多个4x4感受野
    rf_size = 4
    stride = 2
    for i in range(0, img_size - rf_size + 1, stride * 2):
        for j in range(0, img_size - rf_size + 1, stride * 2):
            rect = Rectangle((j - 0.5, i - 0.5), rf_size, rf_size,
                             fill=False, edgecolor='red', alpha=0.5, linewidth=1)
            ax1.add_patch(rect)

    # 2. Conv2 层的感受野覆盖
    ax2 = axes[1]
    ax2.imshow(input_img, cmap='gray', interpolation='nearest')
    ax2.set_title('Conv2+Pool层感受野覆盖', fontweight='bold')
    ax2.set_xlabel('宽度')
    ax2.set_ylabel('高度')

    # 绘制多个10x10感受野
    rf_size = 10
    stride = 4
    for i in range(0, img_size - rf_size + 1, stride * 2):
        for j in range(0, img_size - rf_size + 1, stride * 2):
            rect = Rectangle((j - 0.5, i - 0.5), rf_size, rf_size,
                             fill=False, edgecolor='blue', alpha=0.5, linewidth=1)
            ax2.add_patch(rect)

    # 3. Conv3 层的感受野覆盖
    ax3 = axes[2]
    ax3.imshow(input_img, cmap='gray', interpolation='nearest')
    ax3.set_title('Conv3+Pool层感受野覆盖', fontweight='bold')
    ax3.set_xlabel('宽度')
    ax3.set_ylabel('高度')

    # 绘制多个22x22感受野
    rf_size = 22
    stride = 8
    for i in range(0, img_size - rf_size + 1, stride * 2):
        for j in range(0, img_size - rf_size + 1, stride * 2):
            rect = Rectangle((j - 0.5, i - 0.5), rf_size, rf_size,
                             fill=False, edgecolor='green', alpha=0.5, linewidth=1)
            ax3.add_patch(rect)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Conv1+Pool: 4x4 RF'),
        Patch(facecolor='none', edgecolor='blue', label='Conv2+Pool: 10x10 RF'),
        Patch(facecolor='none', edgecolor='green', label='Conv3+Pool: 22x22 RF'),
    ]

    ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1))

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    """主函数：运行所有可视化"""
    print("开始生成感受野CNN可视化...")

    # 1. 创建综合报告
    model = create_comprehensive_report()

    # 2. 可视化感受野增长
    print("\n生成感受野增长可视化...")
    visualize_receptive_field_growth()

    # 3. 可视化单个神经元感受野
    print("生成单个神经元感受野可视化...")
    visualize_single_neuron_rf()

    # 4. 可视化卷积核分布
    print("生成卷积核权重分布可视化...")
    visualize_kernel_distribution()

    # 5. 可视化感受野重叠
    print("生成感受野重叠可视化...")
    visualize_receptive_field_overlap()

    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)
    print("  - Conv1+Pool: 4×4")
    print("  - Conv2+Pool: 10×10")
    print("  - Conv3+Pool: 22×22")


# 运行主函数
if __name__ == "__main__":
    main()