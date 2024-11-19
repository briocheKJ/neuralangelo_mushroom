import json
import numpy as np
import math
import os
import sys
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

from third_party.colmap.scripts.python.read_write_model import read_points3D_binary


def calculate_from_points(points3D):
    """
    基于 3D 点计算中心 (center)、半径 (radius)、以及边界盒 (bounding_box)
    """
    # 提取所有点的坐标
    xyzs = np.stack([point.xyz for point in points3D.values()])

    # 计算中心：点云的平均位置
    center = xyzs.mean(axis=0)

    # 计算标准差，用于估计半径
    std = xyzs.std(axis=0)

    # 使用 2 倍标准差定义球体半径
    radius = float(std.max() * 2)

    # 使用 3 倍标准差定义边界盒
    bounding_box = [
        [center[0] - std[0] * 3, center[0] + std[0] * 3],
        [center[1] - std[1] * 3, center[1] + std[1] * 3],
        [center[2] - std[2] * 3, center[2] + std[2] * 3],
    ]

    return center.tolist(), radius, bounding_box


def calculate_aabb_scale(radius):
    """
    计算 AABB 缩放值，使用 2 的幂次，适配 INGP 的分辨率要求。
    """
    return int(2 ** np.rint(np.log2(radius)))


def convert_to_transforms_colmap(input_dir, output_filename):
    """
    从 COLMAP 的 points3D.bin 和 transformations_colmap.json 生成 transforms.json。
    """
    # 读取 transformations_colmap.json 文件
    colmap_json_path = os.path.join(input_dir, "transformations_colmap.json")
    points3D_path = os.path.join(input_dir, "../../colmap/points3D.bin")
    output_file_path = os.path.join(input_dir, output_filename)

    with open(colmap_json_path, "r") as f:
        colmap_data = json.load(f)

    # 提取基础相机参数
    w = colmap_data["w"]
    h = colmap_data["h"]
    fl_x = colmap_data["fl_x"]
    fl_y = colmap_data["fl_y"]
    cx = colmap_data["cx"]
    cy = colmap_data["cy"]

    # 计算视角 (angle_x 和 angle_y)
    angle_x = 2 * math.atan(w / (2 * fl_x))
    angle_y = 2 * math.atan(h / (2 * fl_y))

    # 读取 COLMAP 的 points3D.bin
    points3D = read_points3D_binary(points3D_path)

    # 基于点云计算 bounding box 和 sphere
    center, radius, bounding_box = calculate_from_points(points3D)

    # 计算 AABB 缩放值
    aabb_scale = calculate_aabb_scale(radius)

    # 创建 transforms.json 的基本结构
    transforms_data = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "sk_x": 0.0,  # 默认值
        "sk_y": 0.0,  # 默认值
        "k1": 0.0,    # 默认值
        "k2": 0.0,    # 默认值
        "k3": 0.0,    # 默认值
        "k4": 0.0,    # 默认值
        "p1": 0.0,    # 默认值
        "p2": 0.0,    # 默认值
        "is_fisheye": False,  # 默认值
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": aabb_scale,
        "aabb_range": bounding_box,
        "sphere_center": center,
        "sphere_radius": radius,
        "frames": [],
    }

    # 遍历 frames 并转换
    for frame in colmap_data["frames"]:
        transform_matrix = frame["transform_matrix"]
        file_path = frame["file_path"]

        # 创建 transforms.json 中的 frame 结构
        new_frame = {
            "file_path": file_path,
            "transform_matrix": transform_matrix,
        }

        # 如果有 depth 文件路径，也可以附加
        if "depth_file_path" in frame:
            new_frame["depth_file_path"] = frame["depth_file_path"]

        # 添加到 frames 列表
        transforms_data["frames"].append(new_frame)

    # 将结果保存到 transforms.json 文件
    with open(output_file_path, "w") as f:
        json.dump(transforms_data, f, indent=2)

    print(f"Transformed data saved to {output_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="COLMAP 数据的目录，包含 transformations_colmap.json 和 points3D.bin")
    parser.add_argument("--output_filename", type=str, default="transforms.json", help="输出 transforms.json 文件路径")
    args = parser.parse_args()

    convert_to_transforms_colmap(args.input_dir, args.output_filename)
