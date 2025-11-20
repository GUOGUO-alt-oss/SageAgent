#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
import unicodedata

def load_jsonl(paths):
    items = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "src" in obj:
                        items.append(obj)
                except Exception:
                    continue
    return items

FILLERS = ["然后", "就是", "那么", "那个", "这个", "对吧", "你知道", "我觉得", "其实", "好吧", "呃", "嗯", "啊", "嘛", "吧", "呢"]

def normalize_text(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[·••]+", "", s)
    s = re.sub(r"\[(?:音乐|掌声|笑声|旁白|杂音)\]", "", s)
    s = re.sub(r"\((?:音乐|掌声|笑声|旁白|杂音)\)", "", s)
    s = re.sub(r"[呃嗯啊]{2,}", "", s)
    for w in FILLERS:
        s = re.sub(rf"(?:^|\s){re.escape(w)}(?:$|\s)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def ends_with_punct(s):
    return bool(re.search(r"[。！？.!?]$", s))

def autopunct(s):
    if not s:
        return s
    if not ends_with_punct(s):
        if re.search(r"[吗呢吧啊嘛]$", s):
            return s + "？"
        return s + "。"
    return s

def split_sentences(s):
    parts = []
    i = 0
    buf = []
    for ch in s:
        buf.append(ch)
        if ch in "。！？.!?":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
    if buf:
        tail = "".join(buf).strip()
        if tail:
            parts.append(autopunct(tail))
    return parts

def group_by_video(items):
    groups = {}
    for it in items:
        vid = it.get("video_id", "unknown")
        groups.setdefault(vid, []).append(it)
    for vid in groups:
        groups[vid].sort(key=lambda x: (int(x.get("start_ms", 0)), int(x.get("end_ms", 0))))
    return groups

def merge_segments(segments, min_chars, max_gap_ms):
    paragraphs = []
    cur_texts = []
    cur_start = None
    cur_end = None
    last_end = None
    for seg in segments:
        t = normalize_text(str(seg.get("src", "")))
        if not t:
            continue
        gap_ok = True
        if last_end is not None:
            gap_ok = int(seg.get("start_ms", 0)) - int(last_end) <= max_gap_ms
        if cur_texts and (not gap_ok) and (len("".join(cur_texts)) >= min_chars):
            paragraphs.append({
                "video_id": seg.get("video_id"),
                "start_ms": cur_start,
                "end_ms": cur_end,
                "text": finalize_paragraph("".join(cur_texts))
            })
            cur_texts = []
            cur_start = None
            cur_end = None
        if not cur_texts:
            cur_start = int(seg.get("start_ms", 0))
        cur_texts.append(t + " ")
        cur_end = int(seg.get("end_ms", 0))
        last_end = cur_end
        if len("".join(cur_texts)) >= min_chars:
            paragraphs.append({
                "video_id": seg.get("video_id"),
                "start_ms": cur_start,
                "end_ms": cur_end,
                "text": finalize_paragraph("".join(cur_texts))
            })
            cur_texts = []
            cur_start = None
            cur_end = None
            last_end = None
    if cur_texts:
        paragraphs.append({
            "video_id": segments[0].get("video_id") if segments else "unknown",
            "start_ms": cur_start,
            "end_ms": cur_end,
            "text": finalize_paragraph("".join(cur_texts))
        })
    return paragraphs

def to_chinese_punct(s):
    s = s.replace("?", "？").replace("!", "！").replace(",", "，").replace(".", "。")
    s = re.sub(r"[，]{2,}", "，", s)
    s = re.sub(r"[。]{2,}", "。", s)
    return s

KEYWORDS_PAUSE = ["所以", "因此", "但是", "不过", "然后", "接着", "首先", "其次", "最后", "总之", "此外"]

def refine_style_student(s):
    s = to_chinese_punct(s)
    for w in KEYWORDS_PAUSE:
        s = re.sub(rf"(?<![，。！？]){w}", f"，{w}", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"([，。！？])\1+", r"\1", s)
    return s

def finalize_paragraph(s, style="plain"):
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    sents = []
    for part in split_sentences(s):
        part = autopunct(normalize_text(part))
        sents.append(part)
    out = "".join(sents)
    if style == "student":
        out = refine_style_student(out)
    return out

def process_files(inputs, output, min_chars, max_gap_ms, style="plain"):
    items = load_jsonl(inputs)
    groups = group_by_video(items)
    outp = Path(output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for vid, segs in groups.items():
            paras = merge_segments(segs, min_chars=min_chars, max_gap_ms=max_gap_ms)
            for i, p in enumerate(paras):
                rec = {
                    "video_id": vid,
                    "paragraph_id": i,
                    "start_ms": p["start_ms"],
                    "end_ms": p["end_ms"],
                    "text": finalize_paragraph(p["text"], style=style)
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    app = FastAPI()
    class CleanRequest(BaseModel):
        files: list[str]
        min_chars: int = 200
        max_gap_ms: int = 1500
        style: str = "student"
    @app.post("/clean")
    def clean(req: CleanRequest):
        items = load_jsonl(req.files)
        groups = group_by_video(items)
        result = {}
        for vid, segs in groups.items():
            paras = merge_segments(segs, min_chars=req.min_chars, max_gap_ms=req.max_gap_ms)
            result[vid] = [{
                "paragraph_id": i,
                "start_ms": p["start_ms"],
                "end_ms": p["end_ms"],
                "text": finalize_paragraph(p["text"], style=req.style)
            } for i, p in enumerate(paras)]
        return {"data": result}
    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=False)
    ap.add_argument("--output", required=False)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_gap_ms", type=int, default=1500)
    ap.add_argument("--style", type=str, default="student")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    if args.serve:
        try:
            import uvicorn
        except Exception:
            print("uvicorn not installed", file=sys.stderr)
            sys.exit(1)
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return
    if not args.input or not args.output:
        print("missing arguments", file=sys.stderr)
        sys.exit(1)
    process_files(args.input, args.output, args.min_chars, args.max_gap_ms, style=args.style)

if __name__ == "__main__":
    main()
