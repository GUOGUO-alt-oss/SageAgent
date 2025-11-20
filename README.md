# SageAgent 项目交付指南

## 项目简介
- 将老师课堂视频/音频进行切片转写（ASR），清洗与分段，自动生成章节与多层次总结，并支持调用 LLM 进行重点分析与学习笔记生成。
- 提供两种使用方式：命令行流水线与 Web UI。

## 环境准备
- 系统依赖：需安装 `ffmpeg` 与 `ffprobe`（Linux/Ubuntu 示例：`sudo apt-get install ffmpeg`）。
- Python 版本：建议 Python 3.12。
- 安装依赖：
  - 创建并激活虚拟环境（可选）：
    - `python3 -m venv .venv && source .venv/bin/activate`
  - 安装 Python 包：
    - `pip install -r requirements.txt`

### ASR 引擎选择
- faster-whisper（推荐，GPU/CPU 皆可）：已在 `requirements.txt` 中。
- openai-whisper（纯 Python 推理）：已在 `requirements.txt` 中。
- MSLite 引擎（可选）：需自行安装 `mindspore_lite`，并准备对应模型文件；代码位于 `mslite_whisper.py`。如使用该引擎，`transformers`/`numpy` 也会用到。

## 使用方式一：Web UI
1. 启动服务：
   - `uvicorn web_app:app --host 0.0.0.0 --port 8000`
   - 或 `python3 web_app.py`（若已安装 `uvicorn`）。
2. 打开浏览器：访问 `http://localhost:8000/process_ui`。
3. 在界面上传视频，选择 ASR 引擎与参数；如需启用 LLM 分析，填入 `llm_api_key`、`llm_model`（默认 `deepseek-reasoner`）、`llm_base_url`（默认 `https://api.deepseek.com/v1`）。
4. 处理完成后，结果保存在 `web_out/<uid>/`（含 `out/`、`cleanout/`、`finalout/`）。

## 使用方式二：命令行流水线
示例完整流程（自动 ASR→清洗→章节→总结）：
```
python3 client.py \
  --videos /path/to/videos_or_dir \
  --outdir out \
  --cleanout cleanout \
  --finalout finalout \
  --tmp_audio_dir tmp_audio \
  --engine auto \
  --model_size medium \
  --device cuda \
  --compute_type float16 \
  --language zh \
  --segment_time 120
```
运行结束后，主要结果位于 `finalout/`：
- `micro_summary.jsonl`、`chapter_summary.jsonl`、`global_summary.jsonl`
- 章节摘要（两种风格）：`chapters_summary.jsonl`、`chapters_summary_exam.jsonl`

## LLM 分析与学习笔记
- 通过 Web UI 可直接在处理流程中启用 LLM 分析生成 `focus_analysis.jsonl`。
- 也可使用接口：
  - `/analyze_clean`：对清洗后的段落进行分析（输入 `clean_paragraphs.jsonl` 路径）。
  - `/notes`：基于汇总内容生成学习笔记（需要 `llm_api_key`）。

## 代码入口与结构
- Web 服务入口：`web_app.py`（挂载静态页 `static/`，提供处理与进度、结果查询接口）。
- 命令行主控：`client.py`（封装各阶段处理）。
- 核心模块：
  - `build_dataset.py`：切片与 ASR（依赖 `ffmpeg/ffprobe`）。
  - `clean_text.py`：清洗与合并段落（可启动为 FastAPI 服务）。
  - `chapter_segmenter.py`：章节分段（可启动为 FastAPI 服务）。
  - `summarizer.py`、`summarize_chapters.py`：生成多层次摘要与章节摘要（支持 FastAPI 服务/uvicorn）。
  - `llm_analyzer.py`：逐句重点分析（DeepSeek API，或本地兜底规则）。

## 交付给他人的步骤（建议）
1. 打包代码：保留整个 `SageAgent/` 目录，包含 `requirements.txt`、`README.md`、`static/`、`web_app.py` 等；可通过 Git 仓库或压缩包交付。
2. 对方环境部署：
   - 安装 `ffmpeg`；
   - `python3 -m venv .venv && source .venv/bin/activate`；
   - `pip install -r requirements.txt`；
   - 运行 Web 或命令行。
3. 如需 GPU 推理：确保 CUDA 驱动与 `faster-whisper` 支持的 `compute_type`（如 `float16`）。无 GPU 时自动降级。
4. 如要使用 MSLite：按需安装 `mindspore_lite` 并准备 `models/mslite/<size>/` 模型文件。

## 常见问题
- FastAPI 上传文件失败：请确认已安装 `python-multipart`。
- 未检测到 `ffmpeg/ffprobe`：安装系统级依赖后重试。
- LLM 分析报错：检查 `llm_api_key`、`llm_base_url` 与网络连通性；可切换到 `dry_run` 进行本地兜底分析。