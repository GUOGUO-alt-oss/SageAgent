import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ==========================
# 1. 更强 SYSTEM PROMPT
# ==========================
SYSTEM_PROMPT = """
你是一名专业教育内容分析助手。你的任务是“逐句分析老师的讲课内容”，并严格输出 JSON 数组。

【必须严格遵守以下规范】

数组中的每个元素必须包含字段：

1. "句子": 原始句子
2. "类别": "重点" 或 "非重点"
3. 若为重点：
    - "总结": 1-3 句中文总结
    - "重要性说明": 为什么是重点（考试点/核心概念/高频考点）
    - "非重点原因": 必须为空字符串
4. 若为非重点：
    - "总结": 必须为空字符串
    - "重要性说明": 必须为空字符串
    - "非重点原因": 为什么不重要（闲聊/举例/非考试范围/过渡语句等）

【必须遵守】
- 只能输出 JSON 数组
- 禁止出现解释、额外文本、Markdown、代码块
"""



# ==========================
# 2. 分句器（保持你原结构）
# ==========================
def _split_sentences(s):
    parts = []
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
            if not re.search(r"[。！？.!?]$", tail):
                if re.search(r"[吗呢吧啊嘛]$", tail):
                    tail += "？"
                else:
                    tail += "。"
            parts.append(tail)
    return parts



# ==========================
# 3. 统一用户 prompt：更结构化
# ==========================
def _build_user_prompt(sentences):
    obj = {
        "任务描述": "请逐句分析这些教师讲课内容。",
        "句子列表": sentences,
        "输出要求": "严格输出 JSON 数组，每个元素对应输入句子序号，不得改变顺序。"
    }
    return json.dumps(obj, ensure_ascii=False)

PROMPT_CLASSROOM = (
    "你是一名训练有素的教育内容分析引擎。" 
    "输入是一段老师的真实授课内容，请你进行结构化内容分析，输出 JSON。"
    "\n你的任务：\n"
    "识别重点内容：包含关键定理、结论、易错点，或老师强调‘重要’‘考试会考’‘必须记住’ → 用 type: key_content\n"
    "识别次要内容（可略过、不重要）：如‘这课不考’‘简单’‘随便看看’‘跳过也行’ → 用 type: minor_content\n"
    "识别基础定义/概念：用 type: definition\n"
    "识别例题讲解：用 type: example\n"
    "检测老师的教学意图/元信息（提醒、总结、转场）：用 type: meta\n"
    "\n每句话都要输出一句分析，不要遗漏。只输出 JSON 数组，每个元素包含 text 与 type。"
)

def _build_user_prompt_lines(sentences):
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

def _heuristic_type(s):
    s2 = str(s)
    if any(w in s2 for w in ["定义", "概念", "称为", "是指"]):
        return "definition"
    if any(w in s2 for w in ["例题", "例子", "例如", "举例", "比如"]):
        return "example"
    if any(w in s2 for w in ["总结", "提醒", "注意", "首先", "其次", "最后", "转场", "接着", "所以"]):
        return "meta"
    if any(w in s2 for w in ["不考", "简单", "随便看看", "跳过", "略过"]):
        return "minor_content"
    if any(w in s2 for w in ["定理", "结论", "易错点", "重要", "考试会考", "必须记住"]):
        return "key_content"
    return "meta"



# ==========================
# 4. HTTP 请求
# ==========================
def _http_post(url, headers, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _call_deepseek(system_prompt, user_prompt, api_key, base_url, model):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    try:
        body = _http_post(url, headers, payload)
        obj = json.loads(body)
        content = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content
    except Exception:
        return ""



# ==========================
# 5. JSON 提取器（更聪明）
# ==========================
def _extract_json_array(text):
    if not text:
        return None

    # 先清理：“```json ... ```” 类格式
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # 直接解析
    if text.startswith("["):
        try:
            return json.loads(text)
        except:
            pass

    # 从任意位置提取
    m1 = text.find("[")
    m2 = text.rfind("]")
    if m1 != -1 and m2 != -1 and m2 > m1:
        try:
            return json.loads(text[m1:m2+1])
        except:
            pass

    return None



# ==========================
# 6. 本地兜底规则（LLM 失败时用）
# ==========================
def _heuristic_item(s):
    kw = ["定义", "定理", "性质", "公式", "证明", "概念", "结论", "注意", "重点", "考试"]
    for w in kw:
        if w in s:
            return {
                "句子": s,
                "类别": "重点",
                "总结": s[:50],
                "重要性说明": "涉及考试或核心概念",
                "非重点原因": ""
            }
    return {
        "句子": s,
        "类别": "非重点",
        "总结": "",
        "重要性说明": "",
        "非重点原因": "举例/过渡/非考试范围"
    }



# ==========================
# 7. 主句子分析器
# ==========================
def analyze_sentences(sentences, api_key, base_url, model, dry_run=False):
    if not sentences:
        return []

    if dry_run:
        return [_heuristic_item(s) for s in sentences]

    user_prompt = _build_user_prompt(sentences)
    content = _call_deepseek(SYSTEM_PROMPT, user_prompt, api_key, base_url, model)

    arr = _extract_json_array(content)

    # 若 LLM 成功，确保 index 对齐
    if isinstance(arr, list) and len(arr) == len(sentences):
        return arr

    # LLM 失败 → 启用兜底
    return [_heuristic_item(s) for s in sentences]

def analyze_sentences_custom(sentences, api_key, base_url, model, dry_run=False):
    if not sentences:
        return []
    if dry_run:
        return [{"text": s, "type": _heuristic_type(s)} for s in sentences]
    user_prompt = _build_user_prompt_lines(sentences) + "\n请按每句输出一个对象，字段包含 text 与 type，只输出 JSON 数组。"
    content = _call_deepseek(PROMPT_CLASSROOM, user_prompt, api_key, base_url, model)
    arr = _extract_json_array(content)
    if isinstance(arr, list):
        return arr
    return [{"text": s, "type": _heuristic_type(s)} for s in sentences]



# ==========================
# 8. 按段落分析 → 写入 jsonl
# ==========================
def analyze_file(clean_paragraphs_path, output_jsonl_path, api_key, base_url, model, dry_run=False):
    outp = Path(output_jsonl_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(clean_paragraphs_path, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue

            text = str(obj.get("text", ""))
            sentences = _split_sentences(text)
            items = analyze_sentences(sentences, api_key, base_url, model, dry_run=dry_run)

            rec = {
                "video_id": obj.get("video_id"),
                "paragraph_id": obj.get("paragraph_id"),
                "start_ms": obj.get("start_ms"),
                "end_ms": obj.get("end_ms"),
                "items": items
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

def analyze_file_custom(clean_paragraphs_path, output_jsonl_path, api_key, base_url, model, dry_run=False):
    outp = Path(output_jsonl_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(clean_paragraphs_path, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue
            text = str(obj.get("text", ""))
            sentences = _split_sentences(text)
            items = analyze_sentences_custom(sentences, api_key, base_url, model, dry_run=dry_run)
            rec = {
                "video_id": obj.get("video_id"),
                "paragraph_id": obj.get("paragraph_id"),
                "start_ms": obj.get("start_ms"),
                "end_ms": obj.get("end_ms"),
                "items": items
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")



def write_analysis_text(analysis_jsonl_path, text_output_path):
    """将LLM分析结果写入易读的文本格式"""
    Path(text_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    key_sentences = []
    non_key_sentences = []
    total_sentences = 0
    
    with open(analysis_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                items = obj.get("items", [])
                
                for item in items:
                    total_sentences += 1
                    category = item.get("类别", "非重点")
                    sentence = item.get("句子", "")
                    
                    if category == "重点":
                        key_sentences.append({
                            "sentence": sentence,
                            "summary": item.get("总结", ""),
                            "importance": item.get("重要性说明", ""),
                            "video_id": obj.get("video_id"),
                            "start_ms": obj.get("start_ms")
                        })
                    else:
                        non_key_sentences.append({
                            "sentence": sentence,
                            "reason": item.get("非重点原因", ""),
                            "video_id": obj.get("video_id"),
                            "start_ms": obj.get("start_ms")
                        })
            except Exception:
                continue
    
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("🎯 LLM重点分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 统计信息
        key_count = len(key_sentences)
        non_key_count = len(non_key_sentences)
        key_percentage = (key_count / total_sentences * 100) if total_sentences > 0 else 0
        
        f.write("📊 统计信息：\n")
        f.write(f"   • 总句子数：{total_sentences}\n")
        f.write(f"   • 重点句子：{key_count} ({key_percentage:.1f}%)\n")
        f.write(f"   • 非重点句子：{non_key_count} ({100-key_percentage:.1f}%)\n\n")
        
        # 重点内容
        if key_sentences:
            f.write("🔥 重点内容分析：\n")
            f.write("=" * 50 + "\n\n")
            
            for i, item in enumerate(key_sentences, 1):
                start_ms = item.get("start_ms", 0)
                start_min = start_ms // 60000
                start_sec = (start_ms % 60000) // 1000
                
                f.write(f"【重点 {i:02d}】 {start_min:02d}:{start_sec:02d}\n")
                f.write(f"📝 原句：{item['sentence']}\n")
                f.write(f"💎 总结：{item['summary']}\n")
                f.write(f"⭐ 重要性：{item['importance']}\n")
                f.write("-" * 40 + "\n\n")
        
        # 非重点内容（仅显示前10个作为示例）
        if non_key_sentences:
            f.write("📝 非重点内容示例（前10个）：\n")
            f.write("=" * 50 + "\n\n")
            
            for i, item in enumerate(non_key_sentences[:10], 1):
                start_ms = item.get("start_ms", 0)
                start_min = start_ms // 60000
                start_sec = (start_ms % 60000) // 1000
                
                f.write(f"【非重点 {i:02d}】 {start_min:02d}:{start_sec:02d}\n")
                f.write(f"📝 原句：{item['sentence']}\n")
                f.write(f"📄 原因：{item['reason']}\n")
                f.write("-" * 30 + "\n\n")
            
            if len(non_key_sentences) > 10:
                f.write(f"... 还有 {len(non_key_sentences) - 10} 个非重点句子未显示\n\n")
        
        f.write("✅ 分析完成！建议重点关注标为🔥的内容。\n")

# ==========================
# 9. 对外接口
# ==========================
def run_llm_analysis(clean_path, out_path, api_key, base_url, model, dry_run=False):
    analyze_file(clean_path, out_path, api_key, base_url, model, dry_run)
