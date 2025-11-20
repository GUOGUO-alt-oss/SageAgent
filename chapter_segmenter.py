#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

KEY_PHRASES = [
    "接下来我们讲", "接下来讲", "接着我们讲", "下一部分是", "下一节是", "这一节我们讲",
    "本章内容", "首先我们", "其次我们", "然后我们", "最后我们", "总结一下", "小结"
]

def load_paragraphs(path):
    paras = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if all(k in obj for k in ("video_id", "start_ms", "end_ms", "text")):
                    paras.append(obj)
            except Exception:
                continue
    paras.sort(key=lambda x: (x.get("video_id"), int(x.get("start_ms", 0))))
    return paras

def score_break(prev, cur, min_gap_ms, min_len_chars):
    gap = int(cur.get("start_ms", 0)) - int(prev.get("end_ms", 0))
    gap_score = 1 if gap >= min_gap_ms else 0
    len_score = 1 if len(cur.get("text", "")) >= min_len_chars else 0
    phrase = cur.get("text", "")
    phrase_score = 1 if any(kw in phrase for kw in KEY_PHRASES) else 0
    return gap_score + len_score + phrase_score

def segment_chapters(paras, min_gap_ms=10000, min_len_chars=100, threshold=2):
    chapters = []
    cur = {"title": None, "items": [], "start_ms": None, "end_ms": None}
    chapter_id = 1
    for i, p in enumerate(paras):
        if not cur["items"]:
            cur["items"].append(p)
            cur["start_ms"] = int(p.get("start_ms", 0))
            cur["end_ms"] = int(p.get("end_ms", 0))
            continue
        prev = cur["items"][-1]
        sc = score_break(prev, p, min_gap_ms, min_len_chars)
        if sc >= threshold:
            title = infer_title(cur["items"]) or f"Chapter {chapter_id}"
            chapters.append({
                "chapter_id": chapter_id,
                "title": title,
                "start_ms": cur["start_ms"],
                "end_ms": cur["end_ms"],
                "items": cur["items"],
            })
            chapter_id += 1
            cur = {"title": None, "items": [p], "start_ms": int(p.get("start_ms", 0)), "end_ms": int(p.get("end_ms", 0))}
        else:
            cur["items"].append(p)
            cur["end_ms"] = int(p.get("end_ms", 0))
    if cur["items"]:
        title = infer_title(cur["items"]) or f"Chapter {chapter_id}"
        chapters.append({
            "chapter_id": chapter_id,
            "title": title,
            "start_ms": cur["start_ms"],
            "end_ms": cur["end_ms"],
            "items": cur["items"],
        })
    return chapters

def infer_title(items):
    for it in items:
        t = it.get("text", "")
        m = re.search(r"(单调性|周期性|连续|极限|原函数与导函数|原函数|导函数|积分|微分|不等式|函数性质)", t)
        if m:
            return m.group(1)
    for it in items:
        t = it.get("text", "")
        for kw in KEY_PHRASES:
            if kw in t:
                tail = t.split(kw, 1)[-1]
                tail = re.sub(r"[，。！？、]", " ", tail).strip()
                if tail:
                    return tail[:20]
    return None

def write_chapters(chapters, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chapters:
            f.write(json.dumps({
                "chapter_id": ch["chapter_id"],
                "title": ch["title"],
                "start_ms": ch["start_ms"],
                "end_ms": ch["end_ms"],
                "items": [{"start_ms": it["start_ms"], "end_ms": it["end_ms"], "text": it["text"]} for it in ch["items"]]
            }, ensure_ascii=False) + "\n")

def create_app():
    from fastapi import FastAPI
    from pydantic import BaseModel
    app = FastAPI()
    class Req(BaseModel):
        file: str
        min_gap_ms: int = 10000
        min_len_chars: int = 100
        threshold: int = 2
    @app.post("/chapters")
    def chapters(req: Req):
        paras = load_paragraphs(req.file)
        chs = segment_chapters(paras, req.min_gap_ms, req.min_len_chars, req.threshold)
        return {"chapters": chs}
    return app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="清洗后的段落jsonl")
    ap.add_argument("--output", default="out/chapters.jsonl")
    ap.add_argument("--min_gap_ms", type=int, default=10000)
    ap.add_argument("--min_len_chars", type=int, default=100)
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    if args.serve:
        try:
            import uvicorn
        except Exception:
            print("uvicorn not installed", file=sys.stderr)
            return
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return
    paras = load_paragraphs(args.input)
    chs = segment_chapters(paras, args.min_gap_ms, args.min_len_chars, args.threshold)
    write_chapters(chs, args.output)

if __name__ == "__main__":
    main()