#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

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

def collapse_text(items, max_len=2000):
    buf = []
    for it in items:
        t = str(it.get("text", "")).strip()
        if t:
            buf.append(t)
        if sum(len(x) for x in buf) >= max_len:
            break
    return "。".join(buf)

def rule_summary(title, text, style="exam"):
    pts = []
    if re.search(r"单调性", title + text):
        if re.search(r"导数大于0|d\s*f\s*>\s*0", text):
            pts.append("导数>0 ⇒ 函数单调增，导数<0 ⇒ 函数单调减")
        if re.search(r"原函数|导函数", text):
            pts.append("原函数与导函数奇偶关系：原奇⇒导偶，原偶⇒导奇")
    if re.search(r"原函数|导函数", title + text):
        if re.search(r"常数|无数个|加上任意常数", text):
            pts.append("已知导函数，原函数可加常数形成无穷多解")
    if re.search(r"周期", title + text):
        pts.append("周期结论相互推出需满足同时条件，不可单边推出")
    if not pts:
        sents = re.split(r"[。！？]\s*", text)
        pts = [s for s in sents if s][:5]
    brief = title if title else "本章内容"
    if style == "exam":
        exam_points = []
        question_patterns = []
        pitfalls = []
        tips = []
        if re.search(r"单调性", title + text):
            exam_points.append("导数符号与单调性的对应关系")
            question_patterns.append("给定导数符号判断单调增减")
            pitfalls.append("忽略导数符号，仅凭\"单调\"字样下结论")
            tips.append("先判定导数范围与符号，再下结论")
        if re.search(r"原函数|导函数", title + text):
            exam_points.append("原与导的奇偶关系与可加常数性质")
            question_patterns.append("已知小f判断大F奇偶；已知大F判断小f奇偶")
            pitfalls.append("奇函数加常数不再为奇；偶函数加常数仍为偶")
            tips.append("记忆：原奇⇒导偶，原偶⇒导奇；已知导函数，原函数可加常数")
        if re.search(r"周期", title + text):
            exam_points.append("周期性质的相互推出条件")
            question_patterns.append("仅给一侧周期试图推出另一侧周期")
            pitfalls.append("缺少同时条件仍做推出")
            tips.append("周期结论需两侧条件同时满足方可推出")
        return {
            "title": brief,
            "bullets": [p for p in pts if p],
            "exam_points": exam_points,
            "question_patterns": question_patterns,
            "pitfalls": pitfalls,
            "tips": tips,
        }
    return {
        "title": brief,
        "bullets": [p for p in pts if p]
    }

def write_summaries(chs, out_path, style="exam"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chs:
            text = collapse_text(c.get("items", []), max_len=3000)
            summ = rule_summary(c.get("title", ""), text, style=style)
            rec = {
                "chapter_id": c.get("chapter_id"),
                "title": c.get("title"),
                "summary_title": summ["title"],
                "bullets": summ["bullets"]
            }
            if style == "exam":
                rec["exam_points"] = summ.get("exam_points", [])
                rec["question_patterns"] = summ.get("question_patterns", [])
                rec["pitfalls"] = summ.get("pitfalls", [])
                rec["tips"] = summ.get("tips", [])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    app = FastAPI()
    class Req(BaseModel):
        file: str
        style: str = "exam"
    @app.post("/summarize")
    def summarize(req: Req):
        chs = load_chapters(req.file)
        out = []
        for c in chs:
            text = collapse_text(c.get("items", []), max_len=3000)
            summ = rule_summary(c.get("title", ""), text, style=req.style)
            out.append({
                "chapter_id": c.get("chapter_id"),
                "title": c.get("title"),
                "summary_title": summ["title"],
                "bullets": summ["bullets"],
                "exam_points": summ.get("exam_points", []),
                "question_patterns": summ.get("question_patterns", []),
                "pitfalls": summ.get("pitfalls", []),
                "tips": summ.get("tips", []),
            })
        return {"summaries": out}
    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="章节jsonl")
    ap.add_argument("--output", default="finalout/chapters_summary.jsonl")
    ap.add_argument("--style", default="exam")
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    if args.serve:
        try:
            import uvicorn
        except Exception:
            print("uvicorn not installed")
            return
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return
    chs = load_chapters(args.input)
    write_summaries(chs, args.output, style=args.style)

if __name__ == "__main__":
    main()
