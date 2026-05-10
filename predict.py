# -*- coding: utf-8 -*-
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


def predict(generator, audio_path, face_image_path, output_dir, device='cpu'):
    generator.eval()
    # 加载音频文件并转换为梅尔谱
    audio, sr = librosa.load(audio_path, sr=None)
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = torch.tensor(mel_spect, dtype=torch.float32).unsqueeze(0).to(device)  # (1, n_mels, T)

    # 加载人脸图片并进行预处理
    face_image = cv2.imread(face_image_path)
    if face_image is None:
        raise FileNotFoundError(f"无法加载图片: {face_image_path}")
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    face_image = transform(face_image).unsqueeze(0).to(device)  # (1, 3, 256, 256)

    # 自回归生成视频帧序列
    generated_video_frames = []
    current_face = face_image
    with torch.no_grad():
        for t in range(mel_spect.shape[2]):
            audio_frame = mel_spect[:, :, t]  # (1, n_mels)
            gen_frame = generator(audio_frame, current_face)  # (1, 3, 256, 256)
            generated_video_frames.append(gen_frame.squeeze(0).cpu())  # 去掉批次维度
            current_face = gen_frame  # 自回归：用生成的帧作为下一帧的输入

    # 保存为视频文件
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_video_path = os.path.join(output_dir,
                                     f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 25, (256, 256))
    for frame in generated_video_frames:
        frame = frame.permute(1, 2, 0).numpy()  # (H, W, C)
        frame = (frame.clip(0, 1) * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()
    print(f"生成的视频文件已保存至 {output_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Predict: Audio-Driven Speaker Video Generation')
    parser.add_argument('--audio_path', type=str, required=True, help='path to audio file (.wav)')
    parser.add_argument('--face_image', type=str, required=True, help='path to face image (.jpg/.png)')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained generator .pth')
    parser.add_argument('--output_dir', type=str, default='singleprd', help='output directory')
    parser.add_argument('--cpu', action='store_true', help='force CPU')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")

    # 加载预训练的生成器模型
    generator = SVGenerator().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    generator.load_state_dict(state_dict, strict=False)
    generator.eval()

    print(f"已加载模型: {args.model_path}")
    predict(generator, args.audio_path, args.face_image, args.output_dir, device)