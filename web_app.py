#!/usr/bin/env python3
import uuid
from pathlib import Path
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import client as pipeline
from threading import Thread
import time
import llm_analyzer as la

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
PROGRESS = {}
RESULTS = {}

def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

@app.get("/")
def index():
    return FileResponse(Path("static")/"chat.html")

@app.get("/process_ui")
def process_ui():
    return FileResponse(Path("static")/"process.html")

def _set_progress(uid, stage, steps, error=None):
    PROGRESS[uid] = {"uid": uid, "stage": stage, "steps": steps, "error": error}

def _run_job(uid, save_path, params):
    try:
        base = Path("web_out")/uid
        outdir = base/"out"
        cleanout = base/"cleanout"
        finalout = base/"finalout"
        tmp_audio_dir = base/"tmp_audio"
        pipeline.ensure_dirs(str(outdir), str(cleanout), str(finalout))
        steps = {"upload": "done", "asr": "pending", "clean": "pending", "chapters": "pending", "summaries": "pending", "llm": "pending", "done": "pending"}
        _set_progress(uid, "asr", steps)
        train_jsonl = str(outdir/"train.jsonl")
        pipeline.run_asr(str(save_path), train_jsonl, str(tmp_audio_dir), params["engine"], params["model_size"], params["device"], params["compute_type"], params["language"], params["segment_time"])
        steps["asr"] = "done"
        _set_progress(uid, "clean", steps)
        clean_paragraphs = str(cleanout/"clean_paragraphs.jsonl")
        pipeline.run_clean(train_jsonl, clean_paragraphs, params["min_chars"], params["max_gap_ms"], params["style"])
        steps["clean"] = "done"
        _set_progress(uid, "chapters", steps)
        chapters_jsonl = str(outdir/"chapters.jsonl")
        pipeline.run_chapters(clean_paragraphs, chapters_jsonl, params["min_gap_chapter_ms"], params["min_len_chapter_chars"], params["chapter_threshold"], text_format=True)
        steps["chapters"] = "done"
        _set_progress(uid, "summaries", steps)
        pipeline.run_summaries(train_jsonl, chapters_jsonl, str(finalout), params["window_sec"], bool(params["exam"]), text_format=True)
        steps["summaries"] = "done"
        if params["llm_enable"] and params["llm_api_key"]:
            _set_progress(uid, "llm", steps)
            analysis_path = finalout/"focus_analysis.jsonl"
            try:
                pipeline.run_llm_analysis(clean_paragraphs, str(analysis_path), params["llm_api_key"], params["llm_base_url"], params["llm_model"], text_format=True, dry_run=False)
                steps["llm"] = "done"
            except Exception:
                steps["llm"] = "failed"
        _set_progress(uid, "done", steps)
        micro = read_jsonl(finalout/"micro_summary.jsonl")
        chapter = read_jsonl(finalout/"chapter_summary.jsonl")
        global_ = read_jsonl(finalout/"global_summary.jsonl")
        analysis = []
        ap = finalout/"focus_analysis.jsonl"
        if ap.exists():
            recs = read_jsonl(ap)
            flat = []
            for r in recs:
                items = r.get("items", [])
                for it in items:
                    flat.append(it)
            key_first = []
            non_after = []
            for it in flat:
                cat = str(it.get("类别", ""))
                typ = str(it.get("type", ""))
                if cat == "重点" or typ == "key_content":
                    key_first.append(it)
                else:
                    non_after.append(it)
            analysis = key_first + non_after
        RESULTS[uid] = {"uid": uid, "micro_summary": micro, "chapter_summary": chapter, "global_summary": global_, "focus_analysis": analysis}
    except Exception:
        _set_progress(uid, "error", PROGRESS.get(uid, {}).get("steps", {}), error="pipeline_failed")

@app.post("/analyze_clean")
def analyze_clean(
    clean_path: str = Form(...),
    llm_api_key: str = Form(""),
    llm_model: str = Form("deepseek-reasoner"),
    llm_base_url: str = Form("https://api.deepseek.com/v1"),
    dry_run: int = Form(0)
):
    base = Path("web_out")/str(uuid.uuid4())
    finalout = base/"finalout"
    finalout.mkdir(parents=True, exist_ok=True)
    outp = finalout/"focus_analysis.jsonl"
    try:
        pipeline.run_llm_analysis(clean_path, str(outp), llm_api_key, llm_base_url, llm_model, text_format=True, dry_run=bool(dry_run))
    except Exception:
        return JSONResponse(status_code=500, content={"error": "llm_failed"})
    return read_jsonl(outp)

@app.post("/analyze_text")
def analyze_text(
    text: str = Form(...),
    llm_api_key: str = Form(""),
    llm_model: str = Form("deepseek-reasoner"),
    llm_base_url: str = Form("https://api.deepseek.com/v1"),
    dry_run: int = Form(0)
):
    sents = la._split_sentences(text)
    arr = la.analyze_sentences_custom(sents, api_key=llm_api_key, base_url=llm_base_url, model=llm_model, dry_run=bool(dry_run))
    return arr

@app.post("/process")
def process_video(
    video: UploadFile = File(...),
    engine: str = Form("auto"),
    model_size: str = Form("medium"),
    device: str = Form("cuda"),
    compute_type: str = Form("float16"),
    language: str = Form("zh"),
    segment_time: int = Form(120),
    min_chars: int = Form(200),
    max_gap_ms: int = Form(1500),
    style: str = Form("student"),
    min_gap_chapter_ms: int = Form(10000),
    min_len_chapter_chars: int = Form(100),
    chapter_threshold: int = Form(2),
    window_sec: int = Form(60),
    exam: int = Form(0),
    llm_enable: int = Form(1),
    llm_api_key: str = Form("") ,
    llm_model: str = Form("deepseek-reasoner"),
    llm_base_url: str = Form("https://api.deepseek.com/v1")
):
    vid_dir = Path("uploads")
    vid_dir.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid4())
    save_path = vid_dir / f"{uid}_{video.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    base = Path("web_out")/uid
    outdir = base/"out"
    cleanout = base/"cleanout"
    finalout = base/"finalout"
    tmp_audio_dir = base/"tmp_audio"
    pipeline.ensure_dirs(str(outdir), str(cleanout), str(finalout))
    train_jsonl = str(outdir/"train.jsonl")
    try:
        pipeline.run_asr(str(save_path), train_jsonl, str(tmp_audio_dir), engine, model_size, device, compute_type, language, segment_time)
    except Exception:
        return JSONResponse(status_code=500, content={"error": "asr_failed"})
    clean_paragraphs = str(cleanout/"clean_paragraphs.jsonl")
    pipeline.run_clean(train_jsonl, clean_paragraphs, min_chars, max_gap_ms, style)
    chapters_jsonl = str(outdir/"chapters.jsonl")
    pipeline.run_chapters(clean_paragraphs, chapters_jsonl, min_gap_chapter_ms, min_len_chapter_chars, chapter_threshold, text_format=True)
    pipeline.run_summaries(train_jsonl, chapters_jsonl, str(finalout), window_sec, bool(exam), text_format=True)
    analysis = []
    analysis_path = finalout/"focus_analysis.jsonl"
    if llm_enable and llm_api_key:
        try:
            la.analyze_file_custom(clean_paragraphs, str(analysis_path), llm_api_key, llm_base_url, llm_model, dry_run=False)
        except Exception:
            pipeline.run_llm_analysis(clean_paragraphs, str(analysis_path), llm_api_key, llm_base_url, llm_model, text_format=True, dry_run=False)
        recs = read_jsonl(analysis_path)
        flat = []
        for r in recs:
            items = r.get("items", [])
            for it in items:
                flat.append(it)
        key_first = []
        non_after = []
        for it in flat:
            cat = str(it.get("类别", ""))
            typ = str(it.get("type", ""))
            if cat == "重点" or typ == "key_content":
                key_first.append(it)
            else:
                non_after.append(it)
        analysis = key_first + non_after
    micro = read_jsonl(finalout/"micro_summary.jsonl")
    chapter = read_jsonl(finalout/"chapter_summary.jsonl")
    global_ = read_jsonl(finalout/"global_summary.jsonl")
    return {"uid": uid, "micro_summary": micro, "chapter_summary": chapter, "global_summary": global_, "focus_analysis": analysis}

@app.post("/start_process")
def start_process(
    video: UploadFile = File(...),
    engine: str = Form("auto"),
    model_size: str = Form("medium"),
    device: str = Form("cuda"),
    compute_type: str = Form("float16"),
    language: str = Form("zh"),
    segment_time: int = Form(120),
    min_chars: int = Form(200),
    max_gap_ms: int = Form(1500),
    style: str = Form("student"),
    min_gap_chapter_ms: int = Form(10000),
    min_len_chapter_chars: int = Form(100),
    chapter_threshold: int = Form(2),
    window_sec: int = Form(60),
    exam: int = Form(0),
    llm_enable: int = Form(1),
    llm_api_key: str = Form(""),
    llm_model: str = Form("deepseek-reasoner"),
    llm_base_url: str = Form("https://api.deepseek.com/v1")
):
    vid_dir = Path("uploads")
    vid_dir.mkdir(parents=True, exist_ok=True)
    uid = str(uuid.uuid4())
    save_path = vid_dir / f"{uid}_{video.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    params = {
        "engine": engine, "model_size": model_size, "device": device, "compute_type": compute_type,
        "language": language, "segment_time": segment_time, "min_chars": min_chars, "max_gap_ms": max_gap_ms,
        "style": style, "min_gap_chapter_ms": min_gap_chapter_ms, "min_len_chapter_chars": min_len_chapter_chars,
        "chapter_threshold": chapter_threshold, "window_sec": window_sec, "exam": exam,
        "llm_enable": llm_enable, "llm_api_key": llm_api_key, "llm_model": llm_model, "llm_base_url": llm_base_url
    }
    _set_progress(uid, "queued", {"upload": "done", "asr": "pending", "clean": "pending", "chapters": "pending", "summaries": "pending", "llm": "pending", "done": "pending"})
    t = Thread(target=_run_job, args=(uid, save_path, params))
    t.daemon = True
    t.start()
    return {"uid": uid}

@app.get("/progress")
def progress(uid: str):
    return PROGRESS.get(uid, {"uid": uid, "stage": "unknown", "steps": {}, "error": "not_found"})

@app.get("/final")
def final(uid: str):
    if uid in RESULTS:
        return RESULTS[uid]
    p = PROGRESS.get(uid)
    if not p:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    if p.get("stage") == "error":
        return JSONResponse(status_code=500, content={"error": p.get("error", "pipeline_failed")})
    return {"status": "processing"}

@app.post("/notes")
def notes(
    uid: str = Form(...),
    llm_api_key: str = Form(""),
    llm_model: str = Form("deepseek-reasoner"),
    llm_base_url: str = Form("https://api.deepseek.com/v1"),
    style: str = Form("college"),
    dry_run: int = Form(0)
):
    base = Path("web_out")/uid/"finalout"
    ch = read_jsonl(base/"chapter_summary.jsonl")
    gl = read_jsonl(base/"global_summary.jsonl")
    mi = read_jsonl(base/"micro_summary.jsonl")
    def collapse():
        parts = []
        title = "《学习笔记》"
        if gl:
            t = gl[0].get("summary", "") if isinstance(gl, list) else ""
            parts.append("一句话 KPI：" + t)
        parts.append("知识总览")
        for x in ch[:5]:
            tl = x.get("title", "")
            ol = x.get("one_line", "")
            if tl:
                parts.append(tl)
            if ol:
                parts.append(ol)
        return title + "\n\n" + "\n".join(parts)
    if dry_run or not llm_api_key:
        return {"notes": collapse()}
    prompt = (
        "把下面的内容作为你的唯一输出规范：你要把老师课堂上的口语内容，转换成对大学生最友好的知识笔记。绝对禁止生成json、jsonl、表格模板代码、机器格式或结构化数据，只能输出自然语言、可阅读、有标题、有重点的人类风格学习笔记。\n\n"
        "输出风格要求：结构清晰、层级分明，包含模块：知识总览（Executive Summary）、知识结构图/概览思维导图（文字版）、重点与难点、易错点Clarification、典型考题与拆解、老师语音中的关键提醒、核理理解vs死记硬背分区、最终总结（一句话记忆法）。\n"
        "用人类语言写，不使用任何机器格式或键名，全篇自然语言+标题+小结；便于大学生复习，逻辑链条明确，概念解释短狠准，可带文字简图，内容能在5分钟内复习一遍；能提炼老师的口语，识别有用信息，删除口头禅，修正常识性错误与口误，提炼逻辑顺序。\n\n"
        "输出模板：\n"
        "🌿 《章节名称》学习笔记\n"
        "1️⃣ 知识总览（Executive Summary）\n"
        "用3–6句总结全章重点。\n"
        "2️⃣ 知识结构（文字版思维导图）\n"
        "大点1\n小点A\n小点B\n大点2\n小点A\n小点B\n"
        "3️⃣ 重点与难点\n"
        "重点1：解释\n重点2：解释\n难点1：通俗讲解\n难点2：通俗讲解\n"
        "4️⃣ 老师语音里的关键提醒\n"
        "老师特别强调了……\n老师反复说的重点是……\n"
        "5️⃣ 易错点（纠正常见误解）\n"
        "易错点1：正确解释\n易错点2：正确解释\n"
        "6️⃣ 典型题型拆解\n"
        "例题：（题目重述）\n正确思路：\n陷阱：\n为什么错：\n"
        "7️⃣ 一句话记忆法\n"
        "一句诗性记忆句，让概念永不忘。\n"
        "8️⃣ 本章复习Checklist\n"
        "是否理解……\n是否能画出……\n是否能解释……\n"
        "9️⃣ 结尾（抒情收尾）\n"
        "用一句话把学习和人生连起来，让笔记有灵魂。\n\n"
        "请根据全局摘要、章节摘要与微段摘要，生成以上格式的学习笔记，仅输出自然语言。"
    )
    src = {
        "global": gl,
        "chapters": ch,
        "micro": mi
    }
    user = json.dumps(src, ensure_ascii=False)
    try:
        url = llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"}
        payload = {"model": llm_model, "messages": [{"role":"system","content": prompt},{"role":"user","content": user}], "temperature": 0.2}
        body = la._http_post(url, headers, payload)
        obj = json.loads(body)
        content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"notes": content}
    except Exception:
        return {"notes": collapse()}

@app.post("/chat")
def chat(
    prompt: str = Form(...),
    history: str = Form(""),
    llm_api_key: str = Form(""),
    llm_model: str = Form("deepseek-chat"),
    llm_base_url: str = Form("https://api.deepseek.com/v1"),
    system_prompt: str = Form("你是一名专业的助教，回答要简洁、准确。")
):
    msgs = []
    if history:
        try:
            arr = json.loads(history)
            for m in arr:
                r = str(m.get("role", "")).strip()
                c = str(m.get("content", ""))
                if r in ("user", "assistant") and c:
                    msgs.append({"role": r, "content": c})
        except Exception:
            pass
    msgs.insert(0, {"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    try:
        url = llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"}
        payload = {"model": llm_model, "messages": msgs, "temperature": 0.2}
        body = la._http_post(url, headers, payload)
        obj = json.loads(body)
        content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"reply": content}
    except Exception:
        return {"reply": "当前无法连接到模型，请稍后重试。"}

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception:
        print("uvicorn not installed")
