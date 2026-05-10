# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import re
import datetime
from models import SVGenerator, SVDiscriminator
from dataset import AudioVideoDataset
import argparse
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import torchmetrics
import numpy as np


# 训练函数
def train(generator, discriminator, train_loader, val_loader, criterion_L1, optimizer_G,
          optimizer_D, scheduler_G, scheduler_D, scaler, lambda_l1, num_epochs, pretrain,
          num_old_epoch, save_dir, min_loss, train_losses, val_losses, early_stop_count,
          ssim_metric, psnr_metric, val_ssimes, val_psnres):
    if pretrain:
        last_epoch = num_old_epoch
    else:
        last_epoch = 0

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_train_loss = 0.0
        tqdm_bar_train = tqdm(enumerate(train_loader), total=len(train_loader),
                              desc=f'Epoch {epoch + 1}/{num_epochs} Training')
        for batch_idx, data in tqdm_bar_train:
            audio, video = data['audio'].to(device), data['video'].to(device)
            batch_size = video.shape[0]
            sample_length = video.shape[1]

            # ── 先训练判别器 ──
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            with autocast():
                # 随机采样一个帧位置
                frame_idx = np.random.randint(0, sample_length - 1) if sample_length > 1 else 0
                audio_frame = audio[:, frame_idx, :]
                face_frame = video[:, frame_idx, :]
                real_frame = video[:, frame_idx + 1, :]
                gen_frame = generator(audio_frame, face_frame).detach()  # 分离梯度，不更新G

                # D判断真帧
                d_real = discriminator(audio_frame, face_frame, real_frame, real_frame)
                # D判断假帧
                d_fake = discriminator(audio_frame, face_frame, gen_frame, real_frame)

                # hinge loss
                loss_D = torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake))

            scaler_D = scaler if False else GradScaler()  # 复用已有的scaler
            # 用同一个scaler
            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

            # ── 训练生成器（自回归生成完整序列） ──
            optimizer_G.zero_grad()
            with autocast():
                generated_video = torch.zeros(batch_size, sample_length, video.shape[2], video.shape[3],
                                              video.shape[4]).to(device)
                generated_video[:, 0, :, :, :] = video[:, 0, :, :, :]
                y_disc = torch.zeros(batch_size, sample_length - 1).to(device)
                for t in range(sample_length - 1):
                    audio_frame_t = audio[:, t, :]
                    face_frame_t = video[:, t, :]
                    gen_frame_t = generator(audio_frame_t, face_frame_t)
                    generated_video[:, t + 1, :, :, :] = gen_frame_t

                    real_frame_t = video[:, t + 1, :]
                    disc_out = discriminator(audio_frame_t, face_frame_t, gen_frame_t, real_frame_t)
                    y_disc[:, t] = disc_out.squeeze()

                loss_adv = -torch.log(y_disc + 1e-8).mean()
                loss_l1 = criterion_L1(generated_video, video)
                loss_G = loss_adv + lambda_l1 * loss_l1

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            epoch_train_loss += loss_G.item()
            tqdm_bar_train.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        # ── 验证 ──
        epoch_val_loss = 0.0
        epoch_val_ssim = 0.0
        epoch_val_psnr = 0.0
        val_batch_count = 0

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            tqdm_bar_val = tqdm(enumerate(val_loader), total=len(val_loader),
                                desc=f'Epoch {epoch + 1}/{num_epochs} Validation')
            for batch_idx, data in tqdm_bar_val:
                audio, video = data['audio'].to(device), data['video'].to(device)
                batch_size = video.shape[0]
                sample_length = video.shape[1]

                generated_video = torch.zeros(batch_size, sample_length, video.shape[2], video.shape[3],
                                              video.shape[4]).to(device)
                generated_video[:, 0, :, :, :] = video[:, 0, :, :, :]

                batch_ssim = 0.0
                batch_psnr = 0.0
                y_disc = torch.zeros(batch_size, sample_length - 1).to(device)

                for t in range(sample_length - 1):
                    audio_frame = audio[:, t, :]
                    face_frame = video[:, t, :]
                    gen_frame = generator(audio_frame, face_frame)
                    generated_video[:, t + 1, :, :, :] = gen_frame

                    real_frame = video[:, t + 1, :]
                    disc_output = discriminator(audio_frame, face_frame, gen_frame, real_frame)
                    y_disc[:, t] = disc_output.squeeze()

                    ssim_value = ssim_metric(gen_frame, video[:, t + 1, :])
                    psnr_value = psnr_metric(gen_frame, video[:, t + 1, :])
                    batch_ssim += ssim_value.item()
                    batch_psnr += psnr_value.item()

                loss_adv = -torch.log(y_disc + 1e-8).mean()
                loss_l1 = criterion_L1(generated_video, video)
                loss_G = loss_adv + lambda_l1 * loss_l1

                epoch_val_loss += loss_G.item()
                epoch_val_ssim += batch_ssim / (sample_length - 1)
                epoch_val_psnr += batch_psnr / (sample_length - 1)
                val_batch_count += 1

                tqdm_bar_val.set_postfix(loss=loss_G.item())

        # ── epoch 汇总 ──
        scheduler_G.step()
        if scheduler_D:
            scheduler_D.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_ssim = epoch_val_ssim / max(val_batch_count, 1)
        avg_val_psnr = epoch_val_psnr / max(val_batch_count, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_ssimes.append(avg_val_ssim)
        val_psnres.append(avg_val_psnr)

        # 早停机制
        if val_loss < min_loss:
            min_loss = val_loss
            early_stop_count = 0
            # 保存模型
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if pretrain:
                best_generator_path = os.path.join(save_dir,
                                                   f'model_generator_epoch_{num_old_epoch + epoch + 1}_{current_time}.pth')
                best_discriminator_path = os.path.join(save_dir,
                                                       f'model_discriminator_epoch_{num_old_epoch + epoch + 1}_{current_time}.pth')
            else:
                best_generator_path = os.path.join(save_dir, f'model_generator_epoch_{epoch + 1}_{current_time}.pth')
                best_discriminator_path = os.path.join(save_dir,
                                                       f'model_discriminator_epoch_{epoch + 1}_{current_time}.pth')

            torch.save(generator.state_dict(), best_generator_path)
            torch.save(discriminator.state_dict(), best_discriminator_path)

        else:

            early_stop_count += 1
            if early_stop_count >= args.early_stop_limit:
                print(
                    f'Early stopping triggered.已触发早停机制。\nValidation loss did not decrease for {args.early_stop_limit} consecutive epochs.')
                break

        last_epoch += 1

    generator_save_path = os.path.join(save_dir,
                                       f'model_generator_epoch_{last_epoch + 1}_{current_time}.pth')
    discriminator_save_path = os.path.join(save_dir,
                                           f'model_discriminator_epoch_{last_epoch + 1}_{current_time}.pth')
    os.makedirs(os.path.dirname(generator_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(discriminator_save_path), exist_ok=True)
    torch.save(generator.state_dict(), generator_save_path)
    torch.save(discriminator.state_dict(), discriminator_save_path)
    print(f"\t{last_epoch}ep 训练完成")
    print(f"\t生成器模块已保存至 {generator_save_path}\n")
    print(f"\t判别器模块已保存至 {discriminator_save_path}\n")


if __name__ == "__main__":
    # ##############################################################################
    # Phase_0 - 参数设置与传递
    ################################################################################

    print("开始执行：Phase_0<参数设置与传递>")

    parser = argparse.ArgumentParser(description='Training: AuDrv_SVG Model for Audio Driven Speaker Video Generation')
    parser.add_argument('--train_csv', type=str, default='data/train.csv',
                        help='directory of train csv')
    parser.add_argument('--val_csv', type=str, default='data/val.csv',
                        help='directory of validation csv')
    parser.add_argument('--root_dir', type=str, default='data',
                        help='root directory of dataset')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='save directory of trained models')
    parser.add_argument('--preModelG', type=str, default='',
                        help='path to trained generator model')
    parser.add_argument('--preModelD', type=str, default='',
                        help='path to trained discriminator models')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lambda_l1', type=float, default=1,
                        help='weight of l1 loss')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='amount of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--audio_encoder_layers', type=int, default=4,
                        help='layer num of audio encoder')
    parser.add_argument('--mel_channels', type=int, default=80,
                        help='dimension amount of Mel spectrum attributes')
    # parser.add_argument('--face_encoder_layers', type=int, default=7,
    #                     help='layer num of video encoder')
    # parser.add_argument('--face_decoder_layers', type=int, default=7,
    #                     help='layer num of video decoder')

    parser.add_argument('--early_stop_count', type=int, default=0,
                        help='early stop count')
    parser.add_argument('--early_stop_limit', type=int, default=5,
                        help='early stop limit')

    parser.add_argument('--pretrain', action='store_true',
                        help='load trained models')

    args = parser.parse_args()

    save_dir = args.save_dir
    train_csv = args.train_csv
    val_csv = args.val_csv
    root_dir = args.root_dir
    preModelG = args.preModelG
    preModelD = args.preModelD

    # 模型超参
    audio_encoder_layers = args.audio_encoder_layers
    # face_encoder_layers = args.face_encoder_layers
    # face_decoder_layers = args.face_decoder_layers

    # 训练超参
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    lambda_l1 = args.lambda_l1

    # 早停参数
    early_stop_count = args.early_stop_count
    early_stop_limit = args.early_stop_limit
    min_loss = float('inf')
    pretrain = args.pretrain
    num_old_epoch = 0

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
        transforms.RandomRotation(degrees=30),
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # 这里可以添加更多的视频转换
    ])

    # 在训练循环外部定义两个列表来存储训练和验证损失
    # 在每个epoch后更新它们
    train_losses = []
    val_losses = []
    # 在训练循环外部定义两个列表来存储评估指标
    val_ssimes = []
    val_psnres = []

    print("\t传参完成\n")

    # ##############################################################################
    # Phase_1 - 数据加载
    ################################################################################
    print("开始执行：Phase_1<数据加载>")

    train_dataset = AudioVideoDataset(csv_file=train_csv, root_dir=root_dir, audio_transforms=audio_transforms,
                                      video_transforms=video_transforms)
    val_dataset = AudioVideoDataset(csv_file=val_csv, root_dir=root_dir, audio_transforms=audio_transforms,
                                    video_transforms=video_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\t数据加载完成\n")

    # ##############################################################################
    # Phase_2 - 初始化
    ################################################################################
    print("开始执行：Phase_2<模型初始化>")

    # 模型初始化
    generator = SVGenerator(audio_encoder_layers=audio_encoder_layers, n_mel=80).to(device)
    discriminator = SVDiscriminator(audio_encoder_layers=audio_encoder_layers, n_mel=80).to(
        device)

    # 如需要，加载预训练模型
    # pretrain = True
    if pretrain:
        generator_filepath = preModelG
        discriminator_filepath = preModelD
        # 从文件名中提取 epoch 数，假设文件名格式为 "model_epoch_XX_YYYY-MM-DD_HH-MM-SS.pth"
        match = re.search(r'model_generator_epoch_(\d+)_', preModelG)
        if match:
            num_old_epoch = int(match.group(1))
        else:
            num_old_epoch = 0  # 如果没有找到，设置为0或其他默认值

        if os.path.exists(generator_filepath):
            generator.load_state_dict(torch.load(generator_filepath, map_location=device), strict=False)
            discriminator.load_state_dict(torch.load(discriminator_filepath, map_location=device), strict=False)
            print(f"\t已从{generator_filepath}加载预训练模型权重。\n\t已应用保存的模型权重到当前模型。")

        else:
            print(
                f"Warning: Pretrained model file not found at {generator_filepath}. Starting training with a fresh model.")

    # 损失函数
    criterion_L1 = nn.L1Loss()

    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)

    # 学习率调度器
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=100, eta_min=0.00001)

    # 创建GradScaler
    scaler = GradScaler()

    # 初始化评估指标
    ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = torchmetrics.PeakSignalNoiseRatio().to(device)

    print("\t初始化完成\n")

    # ##############################################################################
    # Phase_3 - 模型训练
    ################################################################################

    print("开始执行：Phase_3<模型训练>")

    # 创建日志文件
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 训练
    train(generator, discriminator, train_loader, val_loader, criterion_L1, optimizer_G,
          scheduler_G, scaler, lambda_l1, num_epochs, pretrain, num_old_epoch, save_dir,
          min_loss, train_losses, val_losses, early_stop_count, ssim_metric, psnr_metric, val_ssimes, val_psnres)

    # ##############################################################################
    # Phase_4 - 训练结果可视化
    ################################################################################

    print("开始执行：Phase_4<训练结果可视化>")

    # 绘制损失图
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.figure(figsize=(10, 5))

    plt.plot(train_losses, label='Training Loss')
    # plt.plot(train_contrast_losses, label='Training Contrast Loss')
    plt.plot(val_losses, label='Validation Loss')
    # plt.plot(val_contrast_losses, label='Validation Contrast Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt_dir = f'Result_Fig/loss_plot_epoch_{num_epochs}_{current_time}.png'
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)
    plt.savefig(plt_dir)

    print(f"训练数据可视化保存至 {plt_dir} 目录")

    # 绘制评估指标图
    plt.figure(figsize=(10, 5))
    plt.plot(val_ssimes, label='SSIM Score')
    plt.plot(val_psnres, label='PSNR Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validate Scores Across Epochs')
    plt.legend()

    plt_dir_2 = f'Result_Fig/SSIM-PSNR_plot_epoch_{num_epochs}_{current_time}.png'
    if not os.path.exists(plt_dir_2):
        os.makedirs(plt_dir_2)
    plt.savefig(plt_dir_2)
    plt.show()

    print(f"训练评估指标保存至 {plt_dir_2} 目录")



