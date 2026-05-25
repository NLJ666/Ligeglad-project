import numpy as np
#用于测试比对计算结果导入库验证
from scipy import ndimage


def chamfer_distance_transform_manual(binary_image, metric='D4'):
    """
    参数:
    binary_image: 二值图像(0表示背景，1表示前景)
    metric: 距离度量('D4'或'D8')

    返回:
    距离变换结果
    """
    # 确保输入是二值图像
    img = np.array(binary_image, dtype=np.float32)
    h, w = img.shape

    # 步骤1: 初始化
    N_max = max(h, w) + 10  # 超过图像维度的最大值

    # 前景像素设为0，背景像素设为N_max
    F = np.where(img > 0, 0, N_max)

    # 定义距离权重
    if metric == 'D4':
        # 城市街区距离权重
        weights = [(0, -1, 1), (-1, 0, 1)]  # (dy, dx, weight)
    else:  # D8
        # 棋盘距离权重
        weights = [(0, -1, 1), (-1, 0, 1), (-1, -1, np.sqrt(2))]

    # 步骤2和3的迭代
    max_iterations = 10
    for iteration in range(max_iterations):
        F_old = F.copy()

        # 步骤2: 前向传递 (左上到右下)
        for i in range(h):
            for j in range(w):
                if F[i, j] > 0:  # 只处理背景像素
                    min_val = F[i, j]
                    for dy, dx, weight in weights:
                        ni, nj = i + dy, j + dx
                        if 0 <= ni < h and 0 <= nj < w:
                            min_val = min(min_val, F[ni, nj] + weight)
                    F[i, j] = min_val

        # 步骤3: 反向传递 (右下到左上)
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                if F[i, j] > 0:  # 只处理背景像素
                    min_val = F[i, j]
                    # 反向传递时考虑不同的邻域
                    reverse_weights = [(0, 1, 1), (1, 0, 1)]
                    if metric == 'D8':
                        reverse_weights.append((1, 1, np.sqrt(2)))

                    for dy, dx, weight in reverse_weights:
                        ni, nj = i + dy, j + dx
                        if 0 <= ni < h and 0 <= nj < w:
                            min_val = min(min_val, F[ni, nj] + weight)
                    F[i, j] = min_val

        # 步骤4: 检查收敛
        if np.allclose(F, F_old):
            break

    return F


# 测试函数
def test_manual_implementation():
    # 创建测试图像
    test_image = np.zeros((10, 10), dtype=np.float32)
    test_image[3:7, 3:7] = 1  # 中心区域为前景

    print("测试图像:")
    print(test_image.astype(int))

    # 计算距离变换
    result_D4 = chamfer_distance_transform_manual(test_image, 'D4')
    result_D8 = chamfer_distance_transform_manual(test_image, 'D8')

    print("\nD4距离变换结果:")
    print(np.round(result_D4, 2))

    print("\nD8距离变换结果:")
    print(np.round(result_D8, 2))

    return result_D4, result_D8


def chamfer_distance_transform_library(binary_image, metric='D4'):
    """
    使用库函数实现距离变换

    参数:
    binary_image: 二值图像(0表示背景，1表示前景)
    metric: 距离度量('D4'或'D8')

    返回:
    距离变换结果
    """
    # 确保输入是二值图像
    img = np.array(binary_image, dtype=bool)

    if metric == 'D4':
        # 使用城市街区距离
        return ndimage.distance_transform_cdt(img)
    else:
        # 使用欧几里得距离（最接近D8）
        return ndimage.distance_transform_edt(img)


def compare_implementations():
    """
    比较手动实现和库函数实现的结果
    """
    # 创建测试图像
    np.random.seed(42)
    test_image = np.zeros((8, 8), dtype=np.float32)
    test_image[2:6, 2:6] = 1

    print("=" * 50)
    print("距离变换算法比较")
    print("=" * 50)

    print("测试图像:")
    print(test_image.astype(int))

    # 手动实现
    manual_D4 = chamfer_distance_transform_manual(test_image, 'D4')
    manual_D8 = chamfer_distance_transform_manual(test_image, 'D8')

    # 库函数实现
    library_D4 = chamfer_distance_transform_library(test_image, 'D4')
    library_D8 = chamfer_distance_transform_library(1 - test_image, 'D8')  # 注意库函数的前景定义可能不同

    print("\n手动实现 - D4距离:")
    print(np.round(manual_D4, 2))

    print("\n库函数实现 - D4距离:")
    print(np.round(library_D4, 2))

    print("\n手动实现 - D8距离:")
    print(np.round(manual_D8, 2))

    print("\n库函数实现 - D8距离:")
    print(np.round(library_D8, 2))

    # 计算差异
    diff_D4 = np.abs(manual_D4 - library_D4)
    diff_D8 = np.abs(manual_D8 - library_D8)

    print(f"\nD4距离变换平均差异: {np.mean(diff_D4):.4f}")
    print(f"D8距离变换平均差异: {np.mean(diff_D8):.4f}")


# 运行测试
if __name__ == "__main__":
    # 测试手动实现
    manual_D4, manual_D8 = test_manual_implementation()

    # 比较两种方法
    compare_implementations()