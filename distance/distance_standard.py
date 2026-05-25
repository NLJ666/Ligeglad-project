import distance.py
from distance.py import chamfer_distance_transform_manual
from scipy import ndimage
import numpy as np


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