# AuDrv_SVG 音频驱动的说话人视频生成项目

该项目旨在通过深度学习技术，基于音频和人物图片生成说话人视频。 我参考LipGAN架构，采用CMLR数据集，利用基于GAN的2D方法实现视频生成任务，包含一个生成器（Generator）和一个判别器（Discriminator）

## 项目结构

```
audio-driven-video-generation/
│
├── data/                       # 数据集目录
│   ├── train.csv               # 训练集文件子路径
│   ├── val.csv                 # 验证集文件子路径
│   └── test.csv                # 测试集文件子路径
│
├── /Result_Fig/                # 训练过程可视化的图片文档
├── /models/                    # 训练后模型保存目录
├── /singleprd/                 # 存放predict.py生成的视频文件
│
├── dataset.py                  # 数据集处理类定义文件
├── models.py                   # 模型架构定义文件
├── train.py                    # 训练及验证过程脚本
├── test.py                     # 测试过程脚本
├── predict.py                  # 单独预测过程脚本
├── requirements.txt            # 项目依赖环境文件
└── README.md                   # 项目说明文档
```

## 环境配置

在开始之前，请确保您的环境中安装了以下依赖：

```plaintext
librosa==0.8.1
torch==1.9.0
torchvision==0.10.0
torchmetrics==0.7.0
opencv-python==4.5.3.56
pandas==1.2.3
tqdm==4.59.0
argparse
```

您可以通过运行以下命令来安装这些依赖：

```bash
pip install -r requirements.txt
```

### 关于PyTorch

`torch`和`torchvision`的安装可能需要额外的步骤，因为它们通常需要与你的CUDA版本兼容。如果你的系统安装了CUDA，你需要确保安装的`torch`和`torchvision`版本与CUDA版本相匹配。

你可以通过访问PyTorch官方网站的安装指南来获取正确的安装命令：[PyTorch Get Started](https://pytorch.org/get-started/previous-versions/).

## 数据集
我选用了中文新闻联播视频数据集，包含由11位主持人所表述的共102076条句子，每个句子最多包含29个汉字。这个数据集大约有2100条视频和对应的2100条文本。

[Chinese Mandarin Lip Reading (CMLR) Dataset](https://www.vipazoo.cn/CMLR.html)

无需翻墙，百度云下载到本地。

## 数据处理

数据集是CMLR，训练集、验证集和测试集分别位于根目录下的`train.csv`、`val.csv`、`test.csv`文件中。

训练用音频文件和视频文件的路径格式如下：

- 音频文件路径：`data/audio/(说话人)/(日期)/(文件名).wav`
- 视频文件路径：`data/video/(说话人)/(日期)/(文件名).mp4`

例如，`train.csv`中的一行记录为`s6/20100814_section_2_000.03_001.53`，对应的音频文件路径为`data/audio/s6/20100814/20100814_section_2_000.03_001.53.wav`，视频文件路径为`data/video/s6/20100814/20100814_section_2_000.03_001.53.mp4`。

我在构建项目时所用数据集来自老师发的文件：[东南大学云盘](https://pan.seu.edu.cn:443/link/215E44A851DA77CA52FF410F26F15498)。下载需要校园网。本仓库内不含数据文件，需要单独下载或将数据集自行处理后放入指定位置。

我使用Librosa进行音频数据加载和预处理，并在加载音频数据时将其转换为梅尔谱。同时，使用OpenCV进行视频帧的提取和预处理，并在加载视频帧时采用了数据增强技术，如旋转、缩放、裁剪、颜色抖动等，以增强模型的泛化能力。

## 模型架构

我选用了基于GAN的2D方法实现视频生成任务，模型架构定义在`models.py`文件中。

### 生成器SVGenerator

- **语音编码器**：一个4层的标准CNN网络，将输入的音频特征进行编码，得到低维的特征。
- **人脸编码器**：一个具有一系列残差块的7层CNN网络，对输入的需要进行修改的视频帧进行编码，得到低维的特征。
- **人脸解码器**：一个7层的CNN网络，使用音频和面部编码器的输出作为输入，生成修改后视频帧。

### 判别器SVDiscriminator

- **语音编码器**：一个4层的标准CNN网络，结构同生成器的语音编码器。
- **人脸编码器**：一个7层的CNN网络，输入修改前视频帧、修改后视频帧和真实视频帧，输出结果y判别语音编码器和人脸编码器的输出是否同步，以判别唇部同步性。

## 训练及验证

训练和验证过程由`train.py`脚本控制。我使用Xavier均匀初始化权重，采用L1损失和对抗损失进行优化，并使用余弦退火学习率调度机制。训练过程中还会计算SSIM和PSNR等评估指标。

| 技术/优化手段 | 描述                                    | 参数细节                    |
|---------|---------------------------------------|-------------------------|
| 损失函数    | 使用L1损失和对抗损失来优化模型                      |                         |
| 优化器     | 使用Adam优化器                             | 学习率：0.001，权重衰减：1e-5    |
| 学习率调度   | 余弦退火的学习率调度机制                          | T_max：100，η_min：0.00001 |
| 混合精度计算  | 使用混合精度计算的方式训练                         |                         |
| 进度显示    | 以进度条显示训练的进度                           |                         |
| 模型存储    | 训练完成后存储模型方便调整学习率多次训练，模型存储于/models目录   |                         |
| 早停机制    | 加入早停机制来避免过拟合，并在验证集上性能不再提升时停止训练        | 早停阈值：5个epoch无提升         |
| 性能评估    | 训练同时进行验证，计算SSIM和PSNR等指标并可视化    |                         |
| 可视化存储   | 将可视化的图片存入/Result_Fig，将训练后的模型存入/models |                         |
| 进度显示    | 以进度条显示验证的进度                           |                         |

在终端启动`train.py`，配置对应的参数，或直接采用默认参数。

```bash
python train.py \
  --train_csv data/train.csv \
  --val_csv data/val.csv \
  --root_dir data \
  --save_dir models \
  --preModelG models/your_pretrained_generator.pth \
  --preModelD models/your_pretrained_discriminator.pth \
  --lr 0.001 \
  --lambda_l1 1 \
  --num_epochs 10 \
  --batch_size 8 \
  --audio_encoder_layers 4 \
  --mel_channels 80 \
  --early_stop_limit 5 \
  --pretrain
```

## 测试

测试过程由`test.py`脚本控制，该脚本读取`test.csv`文件中的音频文件和对应视频文件的第一帧，并在`data/result`目录内生成对应的视频预测文件。

## 单独预测

单独预测过程由`predict.py`脚本控制，它读取指定路径的音频文件和人物图片，并在`/singleprd`目录内生成一段人物说话的视频文件，唇部动作能和音频保持一致。

## 使用说明

- 训练模型：运行`python train.py`。
- 测试模型：运行`python test.py`。
- 单独预测：运行`python predict.py --audio_path <your_audio_path> --face_image_path <your_face_image_path>`。

## 贡献与反馈

欢迎对本项目提出宝贵的意见和建议。如有任何问题，请通过GitHub Issues进行反馈。

## 许可证

本项目采用[MIT License](https://opensource.org/licenses/MIT).
