# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.2)


# 定义语音编码器
class AudioEncoder(nn.Module):
    def __init__(self, num_layers=4, n_mel=80):
        super(AudioEncoder, self).__init__()
        layers = []
        n_mel = n_mel  # 梅尔谱特征维度
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv1d(n_mel, 128, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1))
            # layers.append(nn.BatchNorm1d(128))# 暂时禁用批归一化
            # layers.append(nn.LayerNorm(128))  # 暂时禁用层归一化
            layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)

    def forward(self, audio_frame):
        # print("\n开始检测 语音编码器")
        # print(f"audio_frame.shape = {audio_frame.shape}")

        # audio_frame.shape = (batch_size, n_mel)

        x = audio_frame.unsqueeze(2)  # 添加一个维度，使其成为 (batch_size, n_mel, 1)
        # print("\nx = audio_frame.unsqueeze(2)")
        # print(f"x.shape = {x.shape}")

        # breakpoint()

        audio_feature = self.layers(x).squeeze(dim=-1)  # 应用层后去除最后一个维度，返回 (batch_size, 128)
        # print("\naudio_feature = self.layers(x).squeeze(dim=-1)")
        # print(f"audio_feature.shape = {audio_feature.shape}")

        return audio_feature


# 定义人脸编码器
class FaceEncoder(nn.Module):
    def __init__(self):
        super(FaceEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.residual_block1 = ResidualBlock(16)
        self.residual_block2 = ResidualBlock(32)
        self.residual_block3 = ResidualBlock(64)

    def forward(self, face_frame):
        # print("\n开始检测 人脸编码器")
        # print(f"face_frame.shape = {face_frame.shape}")

        x = F.leaky_relu(self.conv1(face_frame), 0.2)
        x = self.residual_block1(x)
        # print("\nx = F.leaky_relu(self.conv1(face_frame), 0.2)")
        # print("x = self.residual_block1(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.residual_block2(x)
        # print("\nx = F.leaky_relu(self.conv2(x), 0.2)\nx = self.residual_block2(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.residual_block3(x)
        # print("\nx = F.leaky_relu(self.conv3(x), 0.2)\nx = self.residual_block3(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.conv4(x), 0.2)
        # print("\nx = F.leaky_relu(self.conv4(x), 0.2)")
        # print(f"x.shape = {x.shape}")

        x = self.global_avg_pool(x)
        # print("\nx = self.global_avg_pool(x)")
        # print(f"x.shape = {x.shape}")

        face_feature = x.view(x.size(0), -1)  # 展平为 (batch_size, 256)
        # print("\nface_feature = x.view(x.size(0), -1)")
        # print(f"face_feature.shape = {face_feature.shape}")

        return face_feature


# 定义人脸解码器
class FaceDecoder(nn.Module):
    def __init__(self, num_layers=7):
        super(FaceDecoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, padding=0)
        self.upconv6 = nn.ConvTranspose2d(4, 3, kernel_size=2, stride=2, padding=0)
        self.upconv7 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
        self.upconv8 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
        self.residual_block1 = ResidualBlock(128)
        self.residual_block2 = ResidualBlock(64)
        self.residual_block3 = ResidualBlock(32)
        self.residual_block4 = ResidualBlock(8)
        self.residual_block5 = ResidualBlock(4)
        self.residual_block6 = ResidualBlock(3)
        self.residual_block7 = ResidualBlock(3)

    def forward(self, fusion_feature):
        # print("\n开始检测 人脸解码器")
        # print(f"fusion_feature.shape = {fusion_feature.shape}")

        x = F.leaky_relu(self.upconv1(fusion_feature), 0.2)  # (batch_size, 128, 2, 2)
        x = self.residual_block1(x)
        # print("\nx = F.leaky_relu(self.upconv1(fusion_feature), 0.2)")
        # print("x = self.residual_block1(x)")
        # print(f"x.shape = {x.shape}")
        # breakpoint()

        x = F.leaky_relu(self.upconv2(x), 0.2)  #
        x = self.residual_block2(x)
        # print("\nx = F.leaky_relu(self.upconv2(x), 0.2)")
        # print("x = self.residual_block2(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.upconv3(x), 0.2)  #
        x = self.residual_block3(x)
        # print("\nx = F.leaky_relu(self.upconv3(x), 0.2)")
        # print("x = self.residual_block3(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.upconv4(x), 0.2)  #
        x = self.residual_block4(x)
        # print("\nx = F.leaky_relu(self.upconv4(x), 0.2)")
        # print("x = self.residual_block4(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.upconv5(x), 0.2)  #
        x = self.residual_block5(x)
        # print("\nx = F.leaky_relu(self.upconv5(x), 0.2)")
        # print("x = self.residual_block5(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.upconv6(x), 0.2)  #
        x = self.residual_block6(x)
        # print("\nx = F.leaky_relu(self.upconv6(x), 0.2)")
        # print("x = self.residual_block6(x)")
        # print(f"x.shape = {x.shape}")

        x = F.leaky_relu(self.upconv7(x), 0.2)  #
        x = self.residual_block7(x)
        # print("\nx = F.leaky_relu(self.upconv7(x), 0.2)")
        # print("x = self.residual_block7(x)")
        # print(f"x.shape = {x.shape}")

        output_frame = F.leaky_relu(self.upconv8(x), 0.2)  # (batch_size, 3, 256, 256)
        # print("\noutput_frame = F.leaky_relu(self.upconv8(x), 0.2)")
        # print(f"output_frame.shape = {output_frame.shape}")

        return output_frame


# 定义生成器
class SVGenerator(nn.Module):
    def __init__(self, audio_encoder_layers=4, n_mel=80):
        super(SVGenerator, self).__init__()
        self.audio_encoder = AudioEncoder(audio_encoder_layers, n_mel)
        self.face_encoder = FaceEncoder()
        self.face_decoder = FaceDecoder()

    def forward(self, audio_frame, face_frame):
        # print("\n开始检测生成器")

        audio_feature = self.audio_encoder(audio_frame)
        # print("\naudio_encoder运行成功")
        # breakpoint()

        face_feature = self.face_encoder(face_frame)
        # print("\nface_encoder运行成功")
        # breakpoint()

        # 调整音频特征的维度
        # print("\n开始组合低维特征")

        fusion_feature = torch.cat((audio_feature, face_feature), dim=1)  # (batch_size, 256)
        # print("\nfusion_feature = torch.cat((audio_feature, face_feature), dim=1)")
        # print(f"fusion_feature.shape = {fusion_feature.shape}")

        fusion_feature = fusion_feature.unsqueeze(2).unsqueeze(3)  # (batch_size, 256, 1, 1)
        # print("\nfusion_feature = fusion_feature.unsqueeze(2).unsqueeze(3)")
        # print(f"fusion_feature.shape = {fusion_feature.shape}")

        # print("\n低维特征组合成功")
        # breakpoint()

        output_frame = self.face_decoder(fusion_feature)

        # print("face_decoder运行成功")
        # print(f"output.shape = {output_frame.shape}")
        # breakpoint()

        return output_frame


# 定义判别器
class SVDiscriminator(nn.Module):
    def __init__(self, audio_encoder_layers=4, n_mel=80):
        super(SVDiscriminator, self).__init__()
        self.audio_encoder = AudioEncoder(audio_encoder_layers, n_mel)
        self.face_encoder = FaceEncoder()

        # 特征融合层
        # self.fc_fusion = nn.Linear(128 * 4, 256)  # 假设每个特征的维度为128
        self.fc_fusion = nn.Linear(128 * 2, 256)  # 假设每个特征的维度为128

        # 判别网络
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, audio_frame, face_frame, gen_frame, real_frame):

        true_delta_frame = real_frame - face_frame
        gen_delta_frame = gen_frame - face_frame


        audio_feature = self.audio_encoder(audio_frame)  # 提取当前音频帧低维特征

        gen_delta_feature = self.face_encoder(gen_delta_frame)
        true_delta_feature = self.face_encoder(true_delta_frame)

        delta_feature = true_delta_feature - gen_delta_feature

        # 特征融合
        # 将音频特征与当前帧和下一帧的面部特征进行拼接
        combined_feature = torch.cat((audio_feature, delta_feature), dim=1)
        fused_feature = F.leaky_relu(self.fc_fusion(combined_feature), 0.2)  # (batch_size, 256)

        # 判别网络
        x = F.leaky_relu(self.fc1(fused_feature), 0.2)  # (batch_size, 128)
        output = torch.sigmoid(self.fc2(x))  # (batch_size, 1)

        return output


if __name__ == "__main__":
    # 初始化模型
    generator = SVGenerator()
    discriminator = SVDiscriminator()

    # 如果有GPU，将模型移动到GPU上
    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator = discriminator.cuda()
