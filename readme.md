# AuDrv_SVG — 音频驱动说话人视频生成项目

本项目通过深度学习从音频和人物肖像生成说话人视频。参考 LipGAN 架构，采用 CMLR 数据集，基于 GAN + 2D CNN 实现视频帧生成，包含生成器 (SVGenerator) 和判别器 (SVDiscriminator)。

## 项目结构

```
AuDrv_SVG/
├── data/                       # 数据集目录
│   ├── train.csv               # 训练集文件路径
│   ├── val.csv                 # 验证集文件路径
│   └── test.csv                # 测试集文件路径
├── Result_Fig/                 # 训练过程可视化图片
├── models/                     # 训练好的模型保存目录
├── singleprd/                  # predict.py 生成的视频文件
├── dataset.py                  # 数据集类定义
├── models.py                   # 模型架构定义 (Generator + Discriminator)
├── train.py                    # 训练与验证脚本
├── test.py                     # 测试脚本
├── predict.py                  # 单样本预测脚本
├── requirements.txt            # 项目依赖
└── README.md                   # 本文档
```

## 环境要求

- Python 3.8+
- NVIDIA GPU (推荐 4GB+ VRAM，如 GTX 1050 Ti)
- CUDA 11.8 / 12.1 / 12.4

## 安装指南

### 1. 克隆项目

```bash
git clone https://github.com/IvesSilent/AuDrv_SVG.git
cd AuDrv_SVG
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# 或 .venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# CPU 版本
pip install -r requirements.txt

# CUDA 12.1 版本（推荐，兼容性好）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 数据集

本项目使用 [Chinese Mandarin Lip Reading (CMLR) Dataset](https://www.vipazoo.cn/CMLR.html)，包含 11 位说话人共 102,076 个片段，每个片段对应 29 个字母。

数据集下载地址（东南大学云盘，需校园网访问）：
<https://pan.seu.edu.cn:443/link/215E44A851DA77CA52FF410F26F15498>

下载后解压至项目 `data/` 目录，确保目录结构如下：

```
data/
├── train.csv
├── val.csv
├── test.csv
├── audio/
│   └── s6/
│       └── 20100814/
│           └── 20100814_section_2_000.03_001.53.wav
└── video/
    └── s6/
        └── 20100814/
            └── 20100814_section_2_000.03_001.53.mp4
```

## 模型架构

### 生成器 SVGenerator

| 组件 | 说明 |
|------|------|
| 音频编码器 | 4 层 1D CNN，将梅尔谱编码为 128 维特征 |
| 人脸编码器 | 4 层 2D CNN + 残差块 + 全局池化，输出 128 维特征 |
| 人脸解码器 | 8 层转置 2D CNN + 残差块，从 256 维融合特征生成 256×256 帧 |

### 判别器 SVDiscriminator

- 计算真实帧与生成帧的人脸变化差异（delta）
- 将音频特征与 delta 特征融合后分类真假

## 训练说明

### 基本用法

```bash
python train.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --root_dir data \
  --save_dir models \
  --lr 0.001 \
  --lambda_l1 1 \
  --num_epochs 100 \
  --batch_size 8 \
  --early_stop_limit 10
```

### 断点续训

```bash
python train.py \
  --pretrain \
  --preModelG models/model_generator_epoch_XX_YYYY-MM-DD_HH-MM-SS.pth \
  --preModelD models/model_discriminator_epoch_XX_YYYY-MM-DD_HH-MM-SS.pth
```

### 训练策略

| 策略 | 说明 |
|------|------|
| 损失函数 | L1 损失 + 对抗损失（BCE） |
| 优化器 | Adam（G: lr=0.001 / D: lr=0.0005，weight_decay=1e-5） |
| 学习率 | CosineAnnealingLR（T_max=100, eta_min=1e-5） |
| 混合精度 | 自动混合精度训练 (AMP) |
| 早停 | 验证损失连续 N 个 epoch 未下降时停止 |
| 评估指标 | SSIM、PSNR |
| 保存策略 | 每次验证损失降低时自动保存最佳模型 |

### 常见问题

**Q: 训练时 loss 不下降？**
- 检查判别器是否被正确训练（本项目的 Bug 修复之一）
- 确保数据集路径正确

**Q: 显存不足？**
- 降低 `--batch_size`（4 或 2）
- 确保使用混合精度训练（AMP 默认开启）

## 测试

```bash
python test.py --test_csv data/test.csv --root_dir data --modelG models/best_generator.pth
```

## 单样本预测

```bash
python predict.py \
  --audio_path data/audio/s6/20100814/sample.wav \
  --face_image data/video/s6/20100814/sample_0001.jpg \
  --model_path models/best_generator.pth \
  --output_dir singleprd
```

生成结果保存在 `singleprd/` 目录。

## 项目更新记录

### v2.0 (2026-05-10) — Bug 大修复

**严重 Bug 修复：**
- **判别器训练修复**：添加了 `optimizer_D`，判别器现在真正参与对抗训练（之前只创建了生成器的优化器）
- **判别器输入修复**：修复了训练时传给判别器的参数错误（之前是 `gen_frame` vs `gen_frame`，现在正确传入 `gen_frame` vs `real_frame`）
- **损失函数修复**：改用 BCE Loss（与 sigmoid 输出匹配），之前误用了 Hinge Loss
- **训练损失修复**：修复了 `train_loss` 从未累加的严重 bug（一直为 0）
- **SSIM/PSNR 计算修复**：修复了跨 batch 累加未重置、除法分母错误的问题
- **数据集双转置 Bug**：修复了 `mel_spect.permute(1, 0)` 在特定路径下被执行两次导致维度错乱

**优化改进：**
- 使用 `torch.no_grad()` 分离判别器训练时生成器的梯度计算
- predict.py 改为自回归模式（匹配训练时的行为）
- predict.py 添加了完整的 argparse 命令行支持
- 修复了 final_save_time 在早停触发前未定义的问题
- 修复了验证损失 `val_loss` 变量名不一致的问题
- 完善了 requirements.txt
- 添加了 `.gitignore`

## 许可

本项目基于 [MIT License](https://opensource.org/licenses/MIT) 开源。
