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

def write_text(recs, path):
    """将JSON数据转换为易读的文本格式"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # 判断记录类型
        if not recs:
            f.write("暂无数据\n")
            return
        
        # 检查是否为全局摘要（只有一个记录，包含one_paragraph字段）
        if len(recs) == 1 and "one_paragraph" in recs[0]:
            rec = recs[0]
            f.write("=" * 60 + "\n")
            f.write("🌍 全局摘要\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("📝 一句话总结：\n")
            f.write(f"   {rec.get('one_line', '无')}\n\n")
            
            f.write("📄 详细总结：\n")
            f.write(f"   {rec.get('one_paragraph', '无')}\n\n")
            
            if rec.get('exam_points'):
                f.write("🎯 考试要点：\n")
                for point in rec['exam_points']:
                    f.write(f"   • {point}\n")
                f.write("\n")
            
            if rec.get('pitfalls'):
                f.write("⚠️ 易错点：\n")
                for pitfall in rec['pitfalls']:
                    f.write(f"   • {pitfall}\n")
                f.write("\n")
            
            if rec.get('tips'):
                f.write("💡 学习建议：\n")
                for tip in rec['tips']:
                    f.write(f"   • {tip}\n")
                f.write("\n")
        
        # 检查是否为微段摘要（包含start_ms字段）
        elif "start_ms" in recs[0]:
            f.write("=" * 60 + "\n")
            f.write("📈 微段摘要\n")
            f.write("=" * 60 + "\n\n")
            
            for i, rec in enumerate(recs, 1):
                start_ms = rec.get('start_ms', 0)
                end_ms = rec.get('end_ms', 0)
                start_min = start_ms // 60000
                start_sec = (start_ms % 60000) // 1000
                end_min = end_ms // 60000
                end_sec = (end_ms % 60000) // 1000
                
                f.write(f"【时间段 {i:02d}】 {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}\n")
                f.write(f"📝 摘要：{rec.get('summary', '无')}\n")
                
                if rec.get('exam_points'):
                    f.write("🎯 考试要点：")
                    f.write("、".join(rec['exam_points']))
                    f.write("\n")
                
                if rec.get('pitfalls'):
                    f.write("⚠️ 易错点：")
                    f.write("、".join(rec['pitfalls']))
                    f.write("\n")
                
                if rec.get('tips'):
                    f.write("💡 学习建议：")
                    f.write("、".join(rec['tips']))
                    f.write("\n")
                
                f.write("-" * 40 + "\n\n")
        
        # 检查是否为章节摘要（包含chapter_id字段）
        elif "chapter_id" in recs[0]:
            f.write("=" * 60 + "\n")
            f.write("📚 章节摘要\n")
            f.write("=" * 60 + "\n\n")
            
            for i, rec in enumerate(recs, 1):
                f.write(f"【章节 {rec.get('chapter_id', i)}】{rec.get('title', '无标题')}\n")
                f.write("-" * 40 + "\n")
                f.write(f"📝 一句话总结：\n   {rec.get('one_line', '无')}\n\n")
                f.write(f"📄 详细总结：\n   {rec.get('one_paragraph', '无')}\n\n")
                f.write("=" * 60 + "\n\n")
        
        else:
            # 未知格式，直接输出JSON
            f.write("数据格式：\n")
            for rec in recs:
                f.write(f"{json.dumps(rec, ensure_ascii=False, indent=2)}\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="原始段落/片段 jsonl (src)")
    ap.add_argument("--chapters", required=True, help="章节 jsonl")
    ap.add_argument("--outdir", default="finalout")
    ap.add_argument("--window_sec", type=int, default=60)
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("--text_format", action="store_true", help="生成文本格式输出")
    args = ap.parse_args()
    segs = load_segments(args.segments)
    chs = load_chapters(args.chapters)
    micro = micro_summaries(segs, window_sec=args.window_sec, exam=args.exam)
    chap = chapter_summaries(chs)
    glob = global_summary(chs, exam=args.exam)
    
    # 始终生成JSON格式
    write_jsonl(micro, str(Path(args.outdir)/"micro_summary.jsonl"))
    write_jsonl(chap, str(Path(args.outdir)/"chapter_summary.jsonl"))
    write_jsonl([glob], str(Path(args.outdir)/"global_summary.jsonl"))
    
    # 可选生成文本格式
    if args.text_format:
        write_text(micro, str(Path(args.outdir)/"micro_summary.txt"))
        write_text(chap, str(Path(args.outdir)/"chapter_summary.txt"))
        write_text([glob], str(Path(args.outdir)/"global_summary.txt"))
        print("✅ 文本格式文件已生成")

if __name__ == "__main__":
    main()
