#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AuDrv_SVG 数据准备与预处理工具
==============================
功能：
1. 检测/下载 CMLR 数据集
2. 验证数据完整性
3. 生成 train/val/test CSV
4. 预处理音频（梅尔谱）和视频帧
5. 生成小样本测试集
"""

import os
import sys
import csv
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


def parse_csv_line(line):
    """解析 CSV 中的记录行"""
    record = line.strip()
    if not record or record.startswith('#'):
        return None
    parts = record.split('\t')
    record_path = parts[0].strip()
    if not record_path:
        return None
    try:
        speaker, date_file = record_path.split('/')
        date_part = date_file.split('_')[0]
        rest_parts = date_file.split('_')[1:]
        filename = '_'.join(rest_parts)
        return {
            'speaker': speaker,
            'date': date_part,
            'filename': filename,
            'record': record_path
        }
    except (ValueError, IndexError):
        return None


def check_dataset_structure(root_dir):
    """
    检查数据集目录结构是否完整
    返回 (exists: bool, missing: list)
    """
    root = Path(root_dir)
    required = ['train.csv', 'val.csv', 'test.csv']
    audio_dir = root / 'audio'
    video_dir = root / 'video'

    missing = []
    for f in required:
        if not (root / f).exists():
            missing.append(f)

    if not audio_dir.exists():
        missing.append('audio/')
    if not video_dir.exists():
        missing.append('video/')

    return len(missing) == 0, missing


def validate_csv_entries(csv_path, root_dir, sample_limit=None):
    """
    验证 CSV 中的每一行是否对应实际存在的文件
    返回 (valid: int, invalid: int, errors: list)
    """
    root = Path(root_dir)
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.error(f"CSV not found: {csv_path}")
        return 0, 0, ["CSV not found"]

    valid = 0
    invalid = 0
    errors = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if sample_limit and i >= sample_limit:
                break
            entry = parse_csv_line(line)
            if not entry:
                continue

            audio_file = root / 'audio' / entry['speaker'] / entry['date'] / f"{entry['filename']}.wav"
            video_file = root / 'video' / entry['speaker'] / entry['date'] / f"{entry['filename']}.mp4"

            if not audio_file.exists():
                invalid += 1
                errors.append(f"Line {i+1}: missing audio {audio_file}")
                continue
            if not video_file.exists():
                invalid += 1
                errors.append(f"Line {i+1}: missing video {video_file}")
                continue
            valid += 1

    return valid, invalid, errors


def scan_existing_files(root_dir):
    """
    扫描现有文件，尝试重建 CSV
    用于手动下载数据集后自动生成 CSV
    """
    root = Path(root_dir)
    audio_dir = root / 'audio'
    video_dir = root / 'video'

    if not audio_dir.exists() or not video_dir.exists():
        log.error("audio/ 或 video/ 目录不存在")
        return []

    records = []
    # 遍历 audio 目录
    for speaker_dir in sorted(audio_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for date_dir in sorted(speaker_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            for wav_file in sorted(date_dir.glob('*.wav')):
                stem = wav_file.stem
                # 检查对应的视频文件是否存在
                mp4_path = video_dir / speaker / date / f"{stem}.mp4"
                if mp4_path.exists():
                    records.append(f"{speaker}/{date}_{stem}")
                else:
                    log.warning(f"跳过（无对应视频）: {wav_file}")

    log.info(f"扫描完成，找到 {len(records)} 个有效样本")
    return records


def create_splits(records, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    将扫描到的文件列表划分为 train/val/test 并输出 CSV
    """
    import random
    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        'train.csv': records[:n_train],
        'val.csv': records[n_train:n_train + n_val],
        'test.csv': records[n_train + n_val:]
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        path = output_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(record + '\n')
        log.info(f"生成 {path} ({len(data)} 条)")

    return splits


def create_mini_dataset(source_dir, target_dir, num_samples=20):
    """
    从数据集提取小样本用于快速测试
    """
    import shutil
    source = Path(source_dir)
    target = Path(target_dir)

    # 读取 val.csv（优先使用验证集）
    val_csv = source / 'val.csv'
    if val_csv.exists():
        with open(val_csv, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    else:
        log.error("val.csv 不存在，无法创建小样本集")
        return False

    # 确保目标目录结构
    for sub in ['audio', 'video']:
        (target / sub).mkdir(parents=True, exist_ok=True)

    # 复制文件
    copied = 0
    samples = []
    for line in lines[:num_samples]:
        entry = parse_csv_line(line)
        if not entry:
            continue

        src_audio = source / 'audio' / entry['speaker'] / entry['date'] / f"{entry['filename']}.wav"
        src_video = source / 'video' / entry['speaker'] / entry['date'] / f"{entry['filename']}.mp4"
        dst_audio = target / 'audio' / entry['speaker'] / entry['date']
        dst_video = target / 'video' / entry['speaker'] / entry['date']

        # 检查源文件是否存在
        if not src_audio.exists() or not src_video.exists():
            log.warning(f"跳过（文件缺失）: {entry['record']}")
            continue

        # 创建子目录
        dst_audio.mkdir(parents=True, exist_ok=True)
        dst_video.mkdir(parents=True, exist_ok=True)

        # 复制
        shutil.copy2(src_audio, dst_audio / f"{entry['filename']}.wav")
        shutil.copy2(src_video, dst_video / f"{entry['filename']}.mp4")
        samples.append(entry['record'])
        copied += 1
        log.info(f"[{copied}/{num_samples}] 已复制 {entry['record']}")

    # 生成 CSV
    csv_path = target / 'val.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(s + '\n')

    log.info(f"小样本数据集创建完成: {target} ({copied} 个样本)")
    return True


def generate_mock_dataset(target_dir, num_train=4, num_val=2, num_test=2,
                          frame_size=(256, 256), n_mels=80, fps=25, duration=2.0):
    """
    生成模拟数据集用于测试训练流程
    （仅用于验证代码运行，不包含真实数据）
    """
    import numpy as np
    import cv2
    import soundfile as sf

    target = Path(target_dir)
    samples = []

    # 模拟说话人
    speakers = ['mock_speaker']

    for split_name, num in [('train', num_train), ('val', num_val), ('test', num_test)]:
        records = []
        for i in range(num):
            speaker = np.random.choice(speakers)
            date = '20260101'
            filename = f"{split_name}_sample_{i:04d}"

            # 生成模拟音频（随机噪声）
            sr = 16000
            t = np.linspace(0, duration, int(sr * duration))
            audio = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.random.randn(len(t))
            audio = audio.astype(np.float32)

            # 生成模拟视频帧（渐变颜色）
            num_frames = int(fps * duration)
            h, w = frame_size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            speaker_dir = target / 'video' / speaker / date
            speaker_dir.mkdir(parents=True, exist_ok=True)
            video_path = speaker_dir / f"{filename}.mp4"

            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            for f_idx in range(num_frames):
                # 模拟人脸：中心渐变圆 + 随机噪声
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                center = (w // 2, h // 2)
                color = (
                    int(128 + 64 * np.sin(2 * np.pi * f_idx / num_frames)),
                    int(100 + 50 * np.cos(2 * np.pi * f_idx / num_frames)),
                    int(140 + 40 * np.sin(2 * np.pi * f_idx / num_frames + 1))
                )
                cv2.circle(frame, center, 80, color, -1)
                # 添加微动（嘴唇区域）
                lip_center = (w // 2, h // 2 + 40)
                lip_w = int(30 + 10 * np.sin(2 * np.pi * f_idx / (num_frames / 2)))
                cv2.ellipse(frame, lip_center, (lip_w, 10), 0, 0, 360,
                            (60, 30, 120), -1)
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                out.write(frame)
            out.release()

            # 保存音频
            audio_dir = target / 'audio' / speaker / date
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{filename}.wav"
            sf.write(str(audio_path), audio, sr)

            records.append(f"{speaker}/{date}_{filename}")
            log.info(f"[{split_name}] 生成 {records[-1]}")

        # 写入 CSV
        csv_path = target / f"{split_name}.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(r + '\n')
        samples.extend(records)

    log.info(f"模拟数据集已生成至 {target}（共 {len(samples)} 个样本）")
    return samples


def print_dataset_summary(root_dir):
    """打印数据集概况"""
    root = Path(root_dir)
    if not root.exists():
        log.error(f"目录不存在: {root}")
        return

    # 统计音频
    wav_files = list(root.rglob('*.wav'))
    mp4_files = list(root.rglob('*.mp4'))
    csv_files = [root / f for f in ['train.csv', 'val.csv', 'test.csv']]

    log.info(f"数据集: {root}")
    log.info(f"  WAV 音频: {len(wav_files)}")
    log.info(f"  MP4 视频: {len(mp4_files)}")

    # 统计说话人
    speakers = set()
    for f in wav_files:
        parts = f.relative_to(root).parts
        if len(parts) >= 3:
            speakers.add(parts[1])
    log.info(f"  说话人: {len(speakers)} ({', '.join(sorted(speakers))})")

    for cf in csv_files:
        if cf.exists():
            count = sum(1 for _ in open(cf) if _.strip())
            log.info(f"  {cf.name}: {count} 条")
        else:
            log.info(f"  {cf.name}: 不存在")

    # 估算总大小
    total_size = sum(f.stat().st_size for f in wav_files + mp4_files)
    log.info(f"  总大小: {total_size / 1024**3:.2f} GB")


def create_download_guide(output_path='DOWNLOAD_GUIDE.md'):
    """创建数据集下载指南"""
    guide = """# CMLR 数据集下载指南

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
python train.py --train_csv data_mock/train.csv --val_csv data_mock/val.csv \\
    --root data_mock --save_dir models_mock --num_epochs 2 --batch_size 2
```
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    log.info(f"下载指南已生成: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AuDrv_SVG 数据准备工具')
    parser.add_argument('--root', type=str, default='data',
                        help='数据集根目录')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查数据集状态')
    parser.add_argument('--validate', action='store_true',
                        help='验证数据集完整性')
    parser.add_argument('--scan', action='store_true',
                        help='扫描现有文件生成 CSV')
    parser.add_argument('--mock', action='store_true',
                        help='生成模拟数据集用于测试')
    parser.add_argument('--mock-dir', type=str, default='data_mock',
                        help='模拟数据集输出目录')
    parser.add_argument('--mock-train', type=int, default=4)
    parser.add_argument('--mock-val', type=int, default=2)
    parser.add_argument('--mock-test', type=int, default=2)
    parser.add_argument('--mini', type=str,
                        help='从数据集创建小样本集，参数为目标目录')
    parser.add_argument('--mini-count', type=int, default=20,
                        help='小样本数量')
    parser.add_argument('--guide', action='store_true',
                        help='生成下载指南')

    args = parser.parse_args()

    if args.guide:
        create_download_guide()
        sys.exit(0)

    if args.check_only:
        exists, missing = check_dataset_structure(args.root)
        print_dataset_summary(args.root)
        sys.exit(0 if exists else 1)

    if args.validate:
        for split in ['train.csv', 'val.csv', 'test.csv']:
            csv_path = Path(args.root) / split
            if csv_path.exists():
                valid, invalid, errors = validate_csv_entries(csv_path, args.root, sample_limit=1000)
                log.info(f"{split}: {valid} 有效, {invalid} 无效")
            else:
                log.warning(f"{split} 不存在，跳过验证")
        sys.exit(0)

    if args.scan:
        records = scan_existing_files(args.root)
        if records:
            create_splits(records, args.root)
        sys.exit(0)

    if args.mock:
        log.info("生成模拟数据集...")
        generate_mock_dataset(
            args.mock_dir,
            num_train=args.mock_train,
            num_val=args.mock_val,
            num_test=args.mock_test
        )
        sys.exit(0)

    if args.mini:
        create_mini_dataset(args.root, args.mini, args.mini_count)
        sys.exit(0)

    # 默认行为
    print_dataset_summary(args.root)
    exists, missing = check_dataset_structure(args.root)
    if not exists:
        log.warning(f"缺失文件/目录: {missing}")
        log.info("可用 --mock 生成模拟数据，或 --guide 查看下载指南")
