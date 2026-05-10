# CMLR 数据集下载指南

## 下载方式

### 方式一：官方百度云（推荐）

1. 打开链接：http://t.cn/A6waiog1
2. 提取码：emqx
3. 下载 `CMLR.zip`（约 5-8 GB）
4. 解压到项目 `data/` 目录

### 方式二：东南大学云盘（需校园网）

1. 打开链接：https://pan.seu.edu.cn:443/link/215E44A851DA77CA52FF410F26F15498
2. 需要东南大学校园网访问
3. 下载后解压到项目 `data/` 目录

## 期望的目录结构

```
data/
├── train.csv              # 训练集列表（71448 条）
├── val.csv                # 验证集列表（10206 条）
├── test.csv               # 测试集列表（20418 条）
├── audio/
│   └── s1/ ~ s11/         # 11 位说话人
│       └── YYYYMMDD/      # 录制日期
│           └── *.wav      # 音频文件（16kHz mono）
└── video/
    └── s1/ ~ s11/
        └── YYYYMMDD/
            └── *.mp4      # 视频文件（25fps, 256x256）
```

## 验证数据

```bash
python prepare_data.py --check-only --root data
```

## 使用模拟数据快速测试

```bash
# 生成 10 个模拟样本用于测试训练流程
python prepare_data.py --mock --mock-dir data_mock --mock-train 4 --mock-val 3 --mock-test 3

# 用模拟数据训练
python train.py --train_csv data_mock/train.csv --val_csv data_mock/val.csv \
    --root data_mock --save_dir models_mock --num_epochs 2 --batch_size 2
```
