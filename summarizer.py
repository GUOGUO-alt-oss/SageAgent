#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path

def load_segments(path):
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if all(k in obj for k in ("video_id", "start_ms", "end_ms", "src")):
                    segs.append(obj)
            except Exception:
                continue
    segs.sort(key=lambda x: (x.get("video_id"), int(x.get("start_ms", 0))))
    return segs

def load_chapters(path):
    chs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if all(k in obj for k in ("chapter_id", "title", "items")):
                    chs.append(obj)
            except Exception:
                continue
    return chs

def simple_summarize(text, max_len=120):
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"[。！？]\s*", text)
    sents = [s for s in sents if s]
    if not sents:
        return text[:max_len]
    key = sents[0]
    if len(key) < 20 and len(sents) > 1:
        key = sents[0] + "，" + sents[1]
    return (key[:max_len] + ("…" if len(key) > max_len else ""))

def exam_extract(text):
    points = []
    if re.search(r"导数|单调", text):
        points.append("导数符号与单调性的对应关系")
    if re.search(r"原函数|导函数|奇偶", text):
        points.append("原/导奇偶关系与原函数可加常数")
    if re.search(r"周期", text):
        points.append("周期性质相互推出需同时条件")
    pitfalls = []
    if re.search(r"奇函数加常数", text):
        pitfalls.append("奇函数加常数不再为奇；偶函数加常数仍为偶")
    tips = []
    if re.search(r"导数", text):
        tips.append("先判定导数范围与符号，再下单调结论")
    return points, pitfalls, tips

def micro_summaries(segs, window_sec=60, exam=False):
    out = []
    cur_start = None
    cur_end = None
    buf = []
    for s in segs:
        start = int(s.get("start_ms", 0))
        end = int(s.get("end_ms", 0))
        if cur_start is None:
            cur_start = start
            cur_end = start + window_sec*1000
        if end <= cur_end:
            buf.append(str(s.get("src", "")))
        else:
            text = "。".join(buf)
            rec = {
                "start_ms": cur_start,
                "end_ms": cur_end,
                "summary": simple_summarize(text, 60)
            }
            if exam:
                pts, pits, tips = exam_extract(text)
                rec["exam_points"] = pts
                rec["pitfalls"] = pits
                rec["tips"] = tips
            out.append(rec)
            buf = [str(s.get("src", ""))]
            cur_start = cur_end
            cur_end = cur_start + window_sec*1000
    if buf:
        text = "。".join(buf)
        rec = {
            "start_ms": cur_start,
            "end_ms": cur_end,
            "summary": simple_summarize(text, 60)
        }
        if exam:
            pts, pits, tips = exam_extract(text)
            rec["exam_points"] = pts
            rec["pitfalls"] = pits
            rec["tips"] = tips
        out.append(rec)
    return out

def chapter_summaries(chs):
    out = []
    for c in chs:
        items = c.get("items", [])
        text = "。".join([str(it.get("text", "")) for it in items])
        one_line = simple_summarize(text, 80)
        one_paragraph = simple_summarize(text, 240)
        out.append({
            "chapter_id": c.get("chapter_id"),
            "title": c.get("title"),
            "one_line": one_line,
            "one_paragraph": one_paragraph
        })
    return out

def global_summary(chs, exam=False):
    all_text = []
    for c in chs:
        for it in c.get("items", []):
            t = str(it.get("text", ""))
            if t:
                all_text.append(t)
    text = "。".join(all_text)
    rec = {"one_paragraph": simple_summarize(text, 360), "one_line": simple_summarize(text, 100)}
    if exam:
        pts, pits, tips = exam_extract(text)
        rec["exam_points"] = pts
        rec["pitfalls"] = pits
        rec["tips"] = tips
    return rec

def write_jsonl(recs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="原始段落/片段 jsonl (src)")
    ap.add_argument("--chapters", required=True, help="章节 jsonl")
    ap.add_argument("--outdir", default="finalout")
    ap.add_argument("--window_sec", type=int, default=60)
    ap.add_argument("--exam", action="store_true")
    args = ap.parse_args()
    segs = load_segments(args.segments)
    chs = load_chapters(args.chapters)
    micro = micro_summaries(segs, window_sec=args.window_sec, exam=args.exam)
    chap = chapter_summaries(chs)
    glob = global_summary(chs, exam=args.exam)
    write_jsonl(micro, str(Path(args.outdir)/"micro_summary.jsonl"))
    write_jsonl(chap, str(Path(args.outdir)/"chapter_summary.jsonl"))
    write_jsonl([glob], str(Path(args.outdir)/"global_summary.jsonl"))

if __name__ == "__main__":
    main()
