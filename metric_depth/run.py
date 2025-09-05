import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    # 参数解析器，用于度量深度估计
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)  # 输入图片路径
    parser.add_argument('--input-size', type=int, default=518)  # 输入图片尺寸
    parser.add_argument('--outdir', type=str, default='./vis_depth')  # 输出目录
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])  # 编码器选择
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')  # 模型权重路径
    parser.add_argument('--max-depth', type=float, default=20)  # 最大深度值
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')  # 保存原始输出
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')  # 仅显示预测
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')  # 使用灰度图
    
    args = parser.parse_args()
    
    # 选择合适的设备（CUDA、MPS或CPU）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 不同编码器的模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 初始化模型
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    # 加载预训练权重
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    # 将模型移到指定设备并设为评估模式
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # 处理输入路径
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            # 如果是txt文件，读取文件列表
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            # 单个文件
            filenames = [args.img_path]
    else:
        # 目录递归搜索所有文件
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    # 获取颜色映射
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    # 处理每个图片文件
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        # 读取原始图片
        raw_image = cv2.imread(filename)
        
        # 推理获取深度图
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        if args.save_numpy:
            # 保存原始深度数据（米为单位）
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)
        
        # 深度值归一化到0-255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            # 灰度模式：重复单通道到三通道
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            # 彩色模式：应用颜色映射
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            # 仅保存深度预测结果
            cv2.imwrite(output_path, depth)
        else:
            # 拼接原图和深度图
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path, combined_result)