# -*- coding=utf-8 -*-
import os
import torch
import torchvision
from torchvision import transforms
from models import SVGenerator
from dataset import AudioVideoDataset
import librosa
import cv2
import numpy as np
from datetime import datetime


def predict(generator, audio_path, face_image_path, output_dir):
    # 加载音频文件并转换为梅尔谱
    audio, sr = librosa.load(audio_path, sr=None)
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = torch.tensor(mel_spect, dtype=torch.float32).unsqueeze(0)  # 添加批次维度

    # 加载人脸图片并进行预处理
    face_image = cv2.imread(face_image_path)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    face_image = transform(face_image).unsqueeze(0)  # 添加批次维度

    # 生成视频帧序列
    generated_video_frames = []
    for i in range(mel_spect.shape[2]):
        audio_frame = mel_spect[:, :, i]
        generated_frame = generator(audio_frame, face_image)
        generated_video_frames.append(generated_frame.squeeze(0))  # 去掉批次维度

    # 将生成的视频帧序列保存为视频文件
    output_video_path = os.path.join(output_dir,
                                     f"{os.path.basename(audio_path)}_{os.path.basename(face_image_path)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25, (256, 256))
    for frame in generated_video_frames:
        frame = frame.permute(1, 2, 0).numpy()  # 转换为(H, W, C)格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换为BGR格式
        video_writer.write(frame)
    video_writer.release()
    print(f"生成的视频文件已保存至 {output_video_path}")


if __name__ == "__main__":
    # 加载预训练的生成器模型
    generator = SVGenerator()
    generator.load_state_dict(torch.load("path/to/pretrained_generator.pth", map_location=torch.device("cpu")))
    generator.eval()

    # 指定音频文件路径和人脸图片路径
    audio_path = "path/to/audio_file.wav"
    face_image_path = "path/to/face_image.jpg"

    # 指定输出目录
    output_dir = "singleprd"
    os.makedirs(output_dir, exist_ok=True)

    # 进行预测
    predict(generator, audio_path, face_image_path, output_dir)