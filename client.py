#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import json

import build_dataset as bd
import clean_text as ct
import chapter_segmenter as cs
import summarizer as sz
import summarize_chapters as sc
import llm_analyzer as la

def ensure_dirs(outdir, cleanout, finalout):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(cleanout).mkdir(parents=True, exist_ok=True)
    Path(finalout).mkdir(parents=True, exist_ok=True)

def run_asr(videos, out_jsonl, tmp_audio_dir, engine, model_size, device, compute_type, language, segment_time):
    vids = bd.collect_videos(videos)
    if not vids:
        print("no videos found", file=sys.stderr)
        sys.exit(1)
    try:
        fast = (engine == "faster-whisper" and model_size in ("small", "base"))
        bd.build_dataset(vids, out_jsonl, tmp_audio_dir, engine, model_size, device, compute_type, language, segment_time, max_workers=8 if fast else 4, ffmpeg_threads=4 if fast else 2)
    except Exception:
        print("gpu unavailable, fallback to cpu", file=sys.stderr)
        bd.build_dataset(vids, out_jsonl, tmp_audio_dir, engine, model_size, "cpu", "float32", language, segment_time, max_workers=4, ffmpeg_threads=2)

def run_clean(train_jsonl, clean_paragraphs, min_chars, max_gap_ms, style):
    ct.process_files([train_jsonl], clean_paragraphs, min_chars, max_gap_ms, style=style)

def run_chapters(clean_paragraphs, chapters_jsonl, min_gap_ms, min_len_chars, threshold, text_format=False):
    paras = cs.load_paragraphs(clean_paragraphs)
    chs = cs.segment_chapters(paras, min_gap_ms=min_gap_ms, min_len_chars=min_len_chars, threshold=threshold)
    cs.write_chapters(chs, chapters_jsonl)
    
    # 可选生成文本格式
    if text_format:
        text_path = str(Path(chapters_jsonl).with_suffix('')) + "_text.txt"
        cs.write_chapters_text(chs, text_path)

def run_summaries(train_jsonl, chapters_jsonl, finalout, window_sec, exam, text_format=False):
    segs = sz.load_segments(train_jsonl)
    chs = sz.load_chapters(chapters_jsonl)
    micro = sz.micro_summaries(segs, window_sec=window_sec, exam=exam)
    chap = sz.chapter_summaries(chs)
    glob = sz.global_summary(chs, exam=exam)
    
    # 始终生成JSON格式
    sz.write_jsonl(micro, str(Path(finalout) / "micro_summary.jsonl"))
    sz.write_jsonl(chap, str(Path(finalout) / "chapter_summary.jsonl"))
    sz.write_jsonl([glob], str(Path(finalout) / "global_summary.jsonl"))
    sc.write_summaries(chs, str(Path(finalout) / "chapters_summary_exam.jsonl"), style="exam")
    sc.write_summaries(chs, str(Path(finalout) / "chapters_summary.jsonl"), style="plain")
    
    # 可选生成文本格式
    if text_format:
        sz.write_text(micro, str(Path(finalout) / "micro_summary.txt"))
        sz.write_text(chap, str(Path(finalout) / "chapter_summary.txt"))
        sz.write_text([glob], str(Path(finalout) / "global_summary.txt"))
        sc.write_summaries_text(chs, str(Path(finalout) / "chapters_summary_text.txt"), style="plain")
        sc.write_summaries_text(chs, str(Path(finalout) / "chapters_summary_exam_text.txt"), style="exam")

def run_llm_analysis(clean_paragraphs, analysis_jsonl, api_key, base_url, model, text_format=False, dry_run=False):
    la.analyze_file(clean_paragraphs, analysis_jsonl, api_key=api_key, base_url=base_url, model=model, dry_run=dry_run)
    
    # 可选生成文本格式
    if text_format:
        text_path = str(Path(analysis_jsonl).with_suffix('')) + "_text.txt"
        la.write_analysis_text(analysis_jsonl, text_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=False)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--cleanout", default="cleanout")
    ap.add_argument("--finalout", default="finalout")
    ap.add_argument("--tmp_audio_dir", default="tmp_audio")
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--model_size", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--segment_time", type=int, default=120)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_gap_ms", type=int, default=1500)
    ap.add_argument("--style", default="student")
    ap.add_argument("--min_gap_chapter_ms", type=int, default=10000)
    ap.add_argument("--min_len_chapter_chars", type=int, default=100)
    ap.add_argument("--chapter_threshold", type=int, default=2)
    ap.add_argument("--window_sec", type=int, default=60)
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("--skip_asr", action="store_true")
    ap.add_argument("--train_jsonl", default="")
    ap.add_argument("--text_format", action="store_true", help="生成文本格式输出")
    ap.add_argument("--llm_api_key", default="", help="LLM API密钥")
    ap.add_argument("--llm_base_url", default="https://api.deepseek.com/v1", help="LLM API地址")
    ap.add_argument("--llm_model", default="deepseek-chat", help="LLM模型名称")
    args = ap.parse_args()

    ensure_dirs(args.outdir, args.cleanout, args.finalout)

    train_jsonl = args.train_jsonl or str(Path(args.outdir) / "train.jsonl")
    if not args.skip_asr:
        if not args.videos:
            print("missing --videos for ASR", file=sys.stderr)
            sys.exit(1)
        run_asr(args.videos, train_jsonl, args.tmp_audio_dir, args.engine, args.model_size, args.device, args.compute_type, args.language, args.segment_time)

    clean_paragraphs = str(Path(args.cleanout) / "clean_paragraphs.jsonl")
    run_clean(train_jsonl, clean_paragraphs, args.min_chars, args.max_gap_ms, args.style)

    chapters_jsonl = str(Path(args.outdir) / "chapters.jsonl")
    run_chapters(clean_paragraphs, chapters_jsonl, args.min_gap_chapter_ms, args.min_len_chapter_chars, args.chapter_threshold, args.text_format)

    run_summaries(train_jsonl, chapters_jsonl, args.finalout, args.window_sec, args.exam, args.text_format)

    # LLM分析（可选）
    if args.llm_api_key:
        analysis_jsonl = str(Path(args.finalout) / "focus_analysis.jsonl")
        run_llm_analysis(clean_paragraphs, analysis_jsonl, args.llm_api_key, args.llm_base_url, args.llm_model, args.text_format)

    manifest = {
        "out": {
            "train_jsonl": train_jsonl,
            "chapters_jsonl": chapters_jsonl
        },
        "cleanout": {
            "clean_paragraphs": clean_paragraphs
        },
        "finalout": {
            "micro_summary": str(Path(args.finalout) / "micro_summary.jsonl"),
            "chapter_summary": str(Path(args.finalout) / "chapter_summary.jsonl"),
            "global_summary": str(Path(args.finalout) / "global_summary.jsonl"),
            "chapters_summary": str(Path(args.finalout) / "chapters_summary.jsonl"),
            "chapters_summary_exam": str(Path(args.finalout) / "chapters_summary_exam.jsonl"),
        }
    }
    
    # 添加文本格式文件到清单
    if args.text_format:
        manifest["out"].update({
            "chapters_text": str(Path(args.outdir) / "chapters_text.txt")
        })
        manifest["finalout"].update({
            "micro_summary_txt": str(Path(args.finalout) / "micro_summary.txt"),
            "chapter_summary_txt": str(Path(args.finalout) / "chapter_summary.txt"),
            "global_summary_txt": str(Path(args.finalout) / "global_summary.txt"),
            "chapters_summary_text": str(Path(args.finalout) / "chapters_summary_text.txt"),
            "chapters_summary_exam_text": str(Path(args.finalout) / "chapters_summary_exam_text.txt"),
        })
    
    if args.llm_api_key:
        manifest["finalout"]["focus_analysis"] = str(Path(args.finalout) / "focus_analysis.jsonl")
        if args.text_format:
            manifest["finalout"]["focus_analysis_text"] = str(Path(args.finalout) / "focus_analysis_text.txt")
    
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__":
    main()
