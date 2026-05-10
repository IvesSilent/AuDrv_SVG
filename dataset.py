# -*coding=utf-8*-
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
import librosa.display
import numpy as np
import cv2


class AudioVideoDataset(Dataset):
    def __init__(self, csv_file, root_dir, audio_transforms=None, video_transforms=None):
        """
        初始化数据集类。
        :param csv_file: 包含音频和视频文件子路径的CSV文件路径。
        :param root_dir: 包含音频和视频文件的根目录路径。
        :param audio_transforms: 音频转换的函数或转换序列。
        :param video_transforms: 视频帧转换的函数或转换序列。
        """
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.audio_transforms = audio_transforms
        self.video_transforms = video_transforms

        # self.max_audio_length = 0  # 最大音频长度
        # self.max_video_length = 0  # 最大视频长度
        self.data = pd.read_csv(csv_file)
        # self.infer_lengths()  # 调用一个方法来推断最大长度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从CSV文件中读取音频和视频文件的路径
        record = self.data.iloc[idx]

        if isinstance(record, pd.Series) or isinstance(record, pd.Index):
            record = record.values[0]  # 如果record是Series或Index，取其第一个值
        record = str(record).split('\t')[0]  # 去掉可能的制表符和额外的列

        speaker, date_file_name = record.split('/')
        date = date_file_name.split('_')[0]
        real_file_name_parts = date_file_name.split('_')[1:]
        real_file_name = '_'.join(real_file_name_parts)

        audio_path = os.path.join(self.root_dir, 'audio', speaker, date, real_file_name + '.wav')
        video_path = os.path.join(self.root_dir, 'video', speaker, date, real_file_name + '.mp4')

        # 加载音频文件并转换为梅尔谱
        audio, sr = librosa.load(audio_path, sr=None)
        mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        # 数据增强
        if self.audio_transforms:
            mel_spect = self.audio_transforms(mel_spect)

        # 确保mel_spect是张量
        if not isinstance(mel_spect, torch.Tensor):
            mel_spect = torch.tensor(mel_spect, dtype=torch.float32)

        # 加载视频文件并提取帧
        video_frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.video_transforms:
                frame = self.video_transforms(frame)
            # frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # 转换为Tensor并调整通道顺序
            video_frames.append(frame)
        cap.release()

        # 视频帧数与音频帧数匹配的填充
        # 获取音频帧数和视频帧数
        audio_length = mel_spect.shape[1]  # 假设mel_spect的形状是[80, audio_length]
        video_length = len(video_frames)

        # 如果音频帧数大于视频帧数，则均匀保留音频关键帧使音频帧数等于视频帧数
        if audio_length > video_length:
            # 计算音频帧的抽帧间隔
            audio_interval = audio_length // video_length
            # 计算剩余的音频帧数
            remaining_audio_frames = audio_length % video_length

            # 均匀保留音频关键帧
            indices = []
            for i in range(video_length):
                if i < remaining_audio_frames:
                    indices.append(i * (audio_interval + 1))
                else:
                    indices.append(i * audio_interval + remaining_audio_frames)
            mel_spect = mel_spect[:, indices]
        else:
            # 音频帧数 <= 视频帧数：保持 (n_mels, time_steps)
            pass
        mel_spect = mel_spect.permute(1, 0)  # → (time_steps, n_mels)
        audio_length = mel_spect.shape[0]
        video_frames = torch.stack(video_frames)

        # mel_spect.shape = (audio_length,n_mels)
        # video_frames.shape = (video_length,3,256,256)
        # audio_length = video_length

        # 随机截取50帧或补满50帧
        target_length = 50
        if audio_length > target_length:
            # 随机选择起始帧
            start_frame = np.random.randint(0, audio_length - target_length)
            mel_spect = mel_spect[start_frame:start_frame + target_length, :]
            video_frames = video_frames[start_frame:start_frame + target_length, :, :, :]
            #
            # print(f"随机截取50帧")
            # print(f"mel_spect.shape = {mel_spect.shape}")
            # print(f"video_frames.shape = {video_frames.shape}")

        else:
            # 补满50帧
            padding_frames = target_length - audio_length
            mel_padding = torch.zeros(padding_frames, mel_spect.shape[1])  # 创建补零帧
            mel_spect = torch.cat((mel_spect, mel_padding), dim=0)  # 补零
            video_padding = video_frames[-1].unsqueeze(0).repeat(padding_frames, 1, 1, 1)  # 复制最后一帧
            video_frames = torch.cat((video_frames, video_padding), dim=0)  # 补充视频帧

            # print(f"补满50帧")
            # print(f"mel_spect.shape = {mel_spect.shape}")
            # print(f"video_frames.shape = {video_frames.shape}")

        return {
            'audio': mel_spect,
            'video': video_frames
        }


if __name__ == "__main__":
    train_csv = 'data/train.csv'
    root_dir = 'data'

    # 数据增强示例
    audio_transforms = transforms.Compose([
        transforms.Lambda(lambda x: torch.tensor(x)),
        # 这里可以添加更多的音频转换
    ])

    video_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=30),
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # 这里可以添加更多的视频转换
    ])

    # 使用示例
    dataset = AudioVideoDataset(csv_file=train_csv, root_dir=root_dir, audio_transforms=audio_transforms,
                                video_transforms=video_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # sample = dataset[0]
    # sample_audio_1 = sample['audio']
    # sample_vidio_1 = sample['video']
    #
    # # print(f"testing dataset class.\nsample_audio_1: \n{sample_audio_1}\nsample_vidio_1: \n{sample_vidio_1}\n")
    # print(
    #     f"testing dataset class.\nsample_audio_1.shape = {sample_audio_1.shape}\nsample_vidio_1.shape = {sample_vidio_1.shape}\n")
    # breakpoint()

    # audio.shape = (audio_length, n_mel)
    # vidio.shape = (audio_length,3,256,256)

    for i in range(50):
        print(f"\nsample_{i + 1} :")

        sample = dataset[i]
        sample_audio = sample['audio']
        sample_vidio = sample['video']
        # print(f"audio_length = {sample_audio.shape[0]}")
        # print(f"vidio_length = {sample_vidio.shape[0]}")

