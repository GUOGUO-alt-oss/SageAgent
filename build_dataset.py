#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # pip install tqdm

def extract_audio_segment(video_path: Path, out_dir: Path, segment_idx: int, sr=16000, start_time=None, duration=None, threads=2):
    """
    提取视频或音频的一段，返回生成的 wav 文件路径
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_name = f"{video_path.stem}_seg{segment_idx:03d}.wav"
    wav_path = out_dir / wav_name

    cmd = ["ffmpeg", "-y", "-threads", str(threads), "-i", str(video_path)]
    if start_time is not None and duration is not None:
        cmd += ["-ss", str(start_time), "-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(wav_path)]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def load_engine(engine: str, model_size: str, device: str, compute_type: str):
    if engine == "mslite":
        try:
            from mslite_whisper import LiteWhisperASR
            return ("mslite", LiteWhisperASR(model_size, device, compute_type))
        except Exception:
            if engine == "mslite":
                print("mslite engine not available", file=sys.stderr)
    if engine in ("faster-whisper", "auto"):
        try:
            from faster_whisper import WhisperModel
            return ("faster-whisper", WhisperModel(model_size, device=device, compute_type=compute_type))
        except Exception:
            if engine == "faster-whisper":
                print("faster-whisper not available", file=sys.stderr)
    try:
        import whisper
        return ("whisper", whisper.load_model(model_size, device=device))
    except Exception:
        raise RuntimeError("No ASR engine available. Install faster-whisper or openai-whisper.")

def transcribe_faster(model, audio_path, language):
    segments, info = model.transcribe(str(audio_path), language=language)
    for i, seg in enumerate(segments):
        yield {
            "segment_id": f"{i:05d}",
            "start_ms": int(seg.start * 1000),
            "end_ms": int(seg.end * 1000),
            "text": seg.text.strip()
        }

def transcribe_whisper(model, audio_path, language):
    result = model.transcribe(str(audio_path), language=language, verbose=False)
    for i, seg in enumerate(result.get("segments", [])):
        yield {
            "segment_id": f"{i:05d}",
            "start_ms": int(seg.get("start", 0) * 1000),
            "end_ms": int(seg.get("end", 0) * 1000),
            "text": str(seg.get("text", "")).strip()
        }

def transcribe_mslite(model, audio_path, language):
    for i, seg in enumerate(model.transcribe(str(audio_path), language=language)):
        yield {
            "segment_id": f"{i:05d}",
            "start_ms": int(seg.get("start_ms", 0)),
            "end_ms": int(seg.get("end_ms", 0)),
            "text": str(seg.get("text", "")).strip()
        }

def get_video_duration(video_path: Path):
    """获取视频时长，单位秒"""
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration", "-of",
           "default=noprint_wrappers=1:nokey=1", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def build_dataset(videos, out_jsonl, tmp_audio_dir, engine, model_size, device, compute_type, language, segment_time, max_workers=4, ffmpeg_threads=2):
    eng_name, eng_obj = load_engine(engine, model_size, device, compute_type)

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for v in tqdm(videos, desc="Videos", unit="video"):
            vp = Path(v)
            if not vp.exists():
                continue
            video_id = vp.stem
            duration = get_video_duration(vp)
            num_segments = int(duration // segment_time) + 1

            # 并行处理每个片段
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx in range(num_segments):
                    start = idx * segment_time
                    futures[executor.submit(extract_audio_segment, vp, Path(tmp_audio_dir), idx, 16000, start, segment_time, ffmpeg_threads)] = idx

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Transcribing {video_id}", unit="seg", leave=False):
                    idx = futures[future]
                    audio_path = future.result()
                    if eng_name == "faster-whisper":
                        gen = transcribe_faster(eng_obj, audio_path, language)
                    elif eng_name == "mslite":
                        gen = transcribe_mslite(eng_obj, audio_path, language)
                    else:
                        gen = transcribe_whisper(eng_obj, audio_path, language)
                    for seg in gen:
                        rec = {
                            "id": str(uuid4()),
                            "video_id": video_id,
                            "segment_id": f"{idx}_{seg['segment_id']}",
                            "start_ms": seg["start_ms"] + idx * segment_time * 1000,
                            "end_ms": seg["end_ms"] + idx * segment_time * 1000,
                            "src": seg["text"]
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def collect_videos(path):
    p = Path(path)
    if p.is_dir():
        exts = [".mp4", ".mkv", ".mov", ".avi", ".flv"]
        return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in exts])
    if p.is_file():
        return [str(p)]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out/train.jsonl")
    ap.add_argument("--tmp_audio_dir", default="tmp_audio")
    ap.add_argument("--engine", default="auto", choices=["auto", "faster-whisper", "whisper", "mslite"])
    ap.add_argument("--model_size", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--segment_time", default=120, type=int, help="每段切片时长（秒）")
    args = ap.parse_args()

    videos = collect_videos(args.input)
    if not videos:
        print("No videos found in:", args.input, file=sys.stderr)
        sys.exit(1)

    build_dataset(videos, args.out, args.tmp_audio_dir, args.engine, args.model_size, args.device,
                  args.compute_type, args.language, args.segment_time)

if __name__ == "__main__":
    main()
