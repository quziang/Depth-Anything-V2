"""
Born out of Depth Anything V1 Issue 36
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a set of images to generate depth maps and corresponding point clouds.
The resulting point clouds are saved in the specified output directory.

该脚本处理一组图像以生成深度图和对应的点云。
生成的点云保存在指定的输出目录中。

Usage:
    python script.py --encoder vitl --load-from path_to_model --max-depth 20 --img-path path_to_images --outdir output_directory --focal-length-x 470.4 --focal-length-y 470.4

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --img-path: Path to the input image or directory containing images.
    --outdir: Directory to save the output point clouds.
    --focal-length-x: Focal length along the x-axis.
    --focal-length-y: Focal length along the y-axis.
    
参数说明:
    --encoder: 使用的模型编码器，选择范围为 ['vits', 'vitb', 'vitl', 'vitg']
    --load-from: 预训练模型权重文件路径
    --max-depth: 深度图的最大深度值
    --img-path: 输入图像路径或包含图像的目录路径
    --outdir: 保存输出点云的目录
    --focal-length-x: x轴方向的焦距
    --focal-length-y: y轴方向的焦距
"""

import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    # Parse command-line arguments
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Generate depth maps and point clouds from images.')
    parser.add_argument('--encoder', default='vitl', type=str, choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model encoder to use.')  # 使用的模型编码器
    parser.add_argument('--load-from', default='', type=str, required=True,
                        help='Path to the pre-trained model weights.')  # 预训练模型权重路径
    parser.add_argument('--max-depth', default=20, type=float,
                        help='Maximum depth value for the depth map.')  # 深度图最大深度值
    parser.add_argument('--img-path', type=str, required=True,
                        help='Path to the input image or directory containing images.')  # 输入图像路径
    parser.add_argument('--outdir', type=str, default='./vis_pointcloud',
                        help='Directory to save the output point clouds.')  # 点云输出目录
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')  # x轴焦距
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')  # y轴焦距

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    # 确定使用的设备（CUDA、MPS或CPU）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Model configuration based on the chosen encoder
    # 基于所选编码器的模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Initialize the DepthAnythingV2 model with the specified configuration
    # 使用指定配置初始化 DepthAnythingV2 模型
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))  # 加载模型权重
    depth_anything = depth_anything.to(DEVICE).eval()  # 移到指定设备并设为评估模式

    # Get the list of image files to process
    # 获取要处理的图像文件列表
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            # 如果是txt文件，读取文件中的图像路径列表
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            # 单个图像文件
            filenames = [args.img_path]
    else:
        # 递归搜索目录中的所有文件
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    # Create the output directory if it doesn't exist
    # 如果输出目录不存在则创建
    os.makedirs(args.outdir, exist_ok=True)

    # Process each image file
    # 处理每个图像文件
    for k, filename in enumerate(filenames):
        print(f'Processing {k+1}/{len(filenames)}: {filename}')

        # Load the image
        # 加载图像
        color_image = Image.open(filename).convert('RGB')
        width, height = color_image.size

        # Read the image using OpenCV
        # 使用OpenCV读取图像
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, height)  # 推理获取深度图

        # Resize depth prediction to match the original image size
        # 将深度预测结果调整为原图像尺寸
        resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

        # Generate mesh grid and calculate point cloud coordinates
        # 生成网格并计算点云坐标
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x  # 归一化x坐标
        y = (y - height / 2) / args.focal_length_y  # 归一化y坐标
        z = np.array(resized_pred)  # 深度值
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)  # 3D坐标点
        colors = np.array(color_image).reshape(-1, 3) / 255.0  # 颜色信息

        # Create the point cloud and save it to the output directory
        # 创建点云并保存到输出目录
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)  # 设置点坐标
        pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置颜色
        o3d.io.write_point_cloud(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + ".ply"), pcd)  # 保存点云文件


if __name__ == '__main__':
    main()
