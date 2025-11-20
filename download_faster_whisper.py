#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from faster_whisper import WhisperModel
from tqdm import tqdm
try:
    from huggingface_hub import snapshot_download
    HAS_HF = True
except Exception:
    HAS_HF = False

# ========== 配置 ==========
MODEL_SIZE = "medium"   # 可选: tiny, base, small, medium, large, large-v3
DEVICE = "cuda"         # 或 "cpu"
COMPUTE_TYPE = "float16"  # float16 / int8 / float32
LOCAL_MODEL_DIR = Path("/home/clearpyh/models/faster-whisper")  # 本地存放目录
# =========================

LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = LOCAL_MODEL_DIR / MODEL_SIZE

print(f"模型将下载到: {model_path}")

def repo_id_for(size: str):
    if size in ("tiny", "base", "small", "medium"):
        return f"Systran/faster-whisper-{size}"
    if size in ("large-v2", "large-v3"):
        return f"Systran/faster-whisper-{size}"
    if size == "large":
        return "guillaumekln/faster-whisper-large-v2"
    return None

print("开始下载/加载模型（可能联网第一次下载）...")
need_download = not (model_path.exists() and (model_path / "model.bin").exists())
class TqdmProxy(tqdm):
    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        if name and "desc" not in kwargs:
            kwargs["desc"] = name
        super().__init__(*args, **kwargs)

if need_download and HAS_HF:
    rid = repo_id_for(MODEL_SIZE)
    if rid:
        snapshot_download(repo_id=rid, local_dir=str(model_path), local_dir_use_symlinks=False, tqdm_class=TqdmProxy)
    else:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=str(LOCAL_MODEL_DIR))
        del model

if model_path.exists():
    model = WhisperModel(str(model_path), device=DEVICE, compute_type=COMPUTE_TYPE)
else:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, download_root=str(LOCAL_MODEL_DIR))

# 简单测试离线加载
test_audio = Path("test.wav")  # 可以放一个短音频文件测试
if test_audio.exists():
    print("开始测试模型识别...")
    segments, info = model.transcribe(str(test_audio), language="zh")
    for seg in segments:
        print(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}")
else:
    print("没有 test.wav 测试文件，仅验证模型已下载成功。")

print("模型下载/加载完成，可以离线使用 faster-whisper 了！")
