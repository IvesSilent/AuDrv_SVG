# -*coding=utf-8*-
import torch
import torch.nn as nn
import os
import re
import datetime
from models import SVGenerator
from dataset import AudioVideoDataset
import argparse
from torchvision import transforms
from tqdm import tqdm
import cv2

def test(generator, test_loader, device, save_dir):
    generator.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing'):
            audio, video = data['audio'].to(device), data['video'].to(device)
            batch_size = video.shape[0]
            sample_length = video.shape[1]

            # 生成器预测
            generated_video = torch.zeros(batch_size, sample_length, video.shape[2], video.shape[3], video.shape[4]).to(device)
            generated_video[:, 0, :, :, :] = video[:, 0, :, :, :]  # 第一帧同样本第一帧
            for j in range(sample_length - 1):
                audio_frame = audio[:, j, :]  # 获取当前帧音频切片
                face_frame = generated_video[:, j, :]  # 获取当前帧视频切片
                gen_frame = generator(audio_frame, face_frame)  # 生成下一帧
                generated_video[:, j + 1, :, :, :] = gen_frame  # 保存生成的帧

            # 保存生成的视频
            for k in range(batch_size):
                video_path = os.path.join(save_dir, f'result_{i * batch_size + k}.mp4')
                video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))
                for frame in generated_video[k]:
                    frame = frame.permute(1, 2, 0).cpu().numpy() * 255
                    frame = frame.astype(np.uint8)
                    video_writer.write(frame)
                video_writer.release()

if __name__ == "__main__":
    # ##############################################################################
    # Phase_0 - 参数设置与传递
    ################################################################################

    print("开始执行：Phase_0<参数设置与传递>")

    parser = argparse.ArgumentParser(description='Testing: AuDrv_SVG Model for Audio Driven Speaker Video Generation')
    parser.add_argument('--test_csv', type=str, default='data/test.csv',
                        help='directory of test csv')
    parser.add_argument('--root_dir', type=str, default='data',
                        help='root directory of dataset')
    parser.add_argument('--save_dir', type=str, default='data/result',
                        help='save directory of test results')
    parser.add_argument('--modelG', type=str, required=True,
                        help='path to trained generator model')

    args = parser.parse_args()

    test_csv = args.test_csv
    root_dir = args.root_dir
    save_dir = args.save_dir
    modelG = args.modelG

    # 如可用则启用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据增强
    audio_transforms = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        # 这里可以添加更多的音频转换
    ])

    video_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # 这里可以添加更多的视频转换
    ])

    print("\t传参完成\n")

    # ##############################################################################
    # Phase_1 - 数据加载
    ################################################################################
    print("开始执行：Phase_1<数据加载>")

    test_dataset = AudioVideoDataset(csv_file=test_csv, root_dir=root_dir, audio_transforms=audio_transforms,
                                     video_transforms=video_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\t数据加载完成\n")

    # ##############################################################################
    # Phase_2 - 模型加载
    ################################################################################
    print("开始执行：Phase_2<模型加载>")

    # 模型初始化
    generator = SVGenerator().to(device)

    # 加载预训练模型
    if os.path.exists(modelG):
        generator.load_state_dict(torch.load(modelG, map_location=device), strict=False)
        print(f"\t已从{modelG}加载预训练模型权重。\n\t已应用保存的模型权重到当前模型。")
    else:
        print(f"Warning: Pretrained model file not found at {modelG}. Please check the path.")

    print("\t模型加载完成\n")

    # ##############################################################################
    # Phase_3 - 模型测试
    ################################################################################

    print("开始执行：Phase_3<模型测试>")

    test(generator, test_loader, device, save_dir)

    print("\t模型测试完成\n")