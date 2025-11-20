#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import json

class LiteWhisperASR:
    def __init__(self, model_size, device, compute_type):
        self.device = device
        self.compute_type = compute_type
        try:
            import mindspore_lite as msl
            self.msl = msl
        except Exception:
            raise RuntimeError("mindspore_lite not installed")
        try:
            from transformers import WhisperProcessor
            self.processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}" if os.path.isdir(str(model_size)) is False else None)
        except Exception:
            self.processor = None
        self.root = Path("models/mslite") / str(model_size)
        enc_ms = self.root / "encoder.ms"
        dec_init_ms = self.root / "decoder_init.ms"
        enc_mindir = self.root / "encoder.mindir"
        dec_init_mindir = self.root / "decoder_init.mindir"
        ctx = self._context()
        if enc_ms.exists() and dec_init_ms.exists():
            self.enc = self.msl.Model()
            self.enc.build_from_file(str(enc_ms), self.msl.ModelType.MS, ctx)
            self.dec_init = self.msl.Model()
            self.dec_init.build_from_file(str(dec_init_ms), self.msl.ModelType.MS, ctx)
        elif enc_mindir.exists() and dec_init_mindir.exists():
            self.enc = self.msl.Model()
            self.enc.build_from_file(str(enc_mindir), self.msl.ModelType.MINDIR, ctx)
            self.dec_init = self.msl.Model()
            self.dec_init.build_from_file(str(dec_init_mindir), self.msl.ModelType.MINDIR, ctx)
        else:
            raise RuntimeError("Lite models not found. Convert ONNX to encoder.ms/decoder_init.ms or provide MindIR.")
        try:
            dec_past_path = self.root / "decoder_with_past.mindir"
            if dec_past_path.exists():
                self.dec_past = self.msl.Model()
                self.dec_past.build_from_file(str(dec_past_path), self.msl.ModelType.MINDIR, ctx)
            else:
                self.dec_past = None
        except Exception:
            self.dec_past = None
        self.eos_id = 50257
        self.max_len = 128
        self.beam_size = 3

    def _context(self):
        ctx = self.msl.Context()
        try:
            ctx.target = ["gpu"] if self.device == "cuda" else ["cpu"]
        except Exception:
            ctx.target = ["cpu"]
        return ctx

    def _load_audio(self, path):
        import numpy as np
        from scipy.io import wavfile
        sr, y = wavfile.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32) / (np.iinfo(y.dtype).max if y.dtype != np.float32 else 1.0)
        # pad or trim to 30s
        target = 16000 * 30
        if sr != 16000:
            # assume input is 16k since we extract with ffmpeg; if not, simple resample ratio
            import math
            ratio = 16000 / float(sr)
            idx = (np.arange(int(len(y) * ratio)) / ratio).astype(np.int64)
            idx = np.minimum(idx, len(y) - 1)
            y = y[idx]
        if len(y) > target:
            y = y[:target]
        elif len(y) < target:
            pad = np.zeros(target, dtype=np.float32)
            pad[:len(y)] = y
            y = pad
        # compute log-mel
        n_fft = 400
        hop = 160
        win = np.hanning(n_fft).astype(np.float32)
        frames = 1 + (len(y) - n_fft) // hop if len(y) >= n_fft else 1
        X = np.zeros((frames, n_fft), dtype=np.float32)
        for i in range(frames):
            s = i * hop
            e = s + n_fft
            seg = np.zeros(n_fft, dtype=np.float32)
            seg[:max(0, min(n_fft, len(y) - s))] = y[s:e]
            X[i] = seg * win
        spec = np.fft.rfft(X, n=n_fft, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        # mel filterbank
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
        n_mels = 80
        mels = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), n_mels + 2)
        hz = mel_to_hz(mels)
        bins = np.floor((n_fft + 1) * hz / 16000.0).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            l = bins[i]; c = bins[i+1]; r = bins[i+2]
            if c > l:
                fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
            if r > c:
                fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
        mel = np.dot(power, fb.T)
        mel = np.maximum(mel, 1e-10)
        mel = np.log10(mel).T
        return mel

    def _softmax(self, x):
        x = x.astype(np.float32)
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        return e / (s + 1e-6)

    def _topk(self, probs, k):
        idx = np.argpartition(-probs, k)[:k]
        vals = probs[idx]
        order = np.argsort(-vals)
        return idx[order], vals[order]

    def _greedy_decode(self, enc_hidden, ids):
        ein = self.msl.Tensor(enc_hidden.astype(np.float32))
        for _ in range(self.max_len):
            din = self.msl.Tensor(ids)
            dec_out = []
            self.dec_init.predict([din, ein], dec_out)
            logits = dec_out[0].get_data_to_numpy()
            next_id = int(logits[0, ids.shape[1]-1].argmax())
            ids = np.concatenate([ids, np.array([[next_id]], dtype=np.int32)], axis=1)
            if next_id == self.eos_id:
                break
        return ids

    def _beam_decode(self, enc_hidden, ids, beam_size):
        ein = self.msl.Tensor(enc_hidden.astype(np.float32))
        beams = [(ids, 0.0, False)]
        for _ in range(self.max_len):
            new_beams = []
            all_finished = True
            for seq, score, finished in beams:
                if finished:
                    new_beams.append((seq, score, True))
                    continue
                din = self.msl.Tensor(seq)
                dec_out = []
                self.dec_init.predict([din, ein], dec_out)
                logits = dec_out[0].get_data_to_numpy()
                last = logits[0, seq.shape[1]-1]
                probs = self._softmax(last)
                idxs, vals = self._topk(probs, beam_size)
                for nid, pv in zip(idxs, vals):
                    nseq = np.concatenate([seq, np.array([[int(nid)]], dtype=np.int32)], axis=1)
                    nfin = int(nid) == self.eos_id
                    new_beams.append((nseq, score + float(np.log(pv + 1e-8)), nfin))
                    if not nfin:
                        all_finished = False
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            if all_finished:
                break
        return beams[0][0]

    def transcribe(self, audio_path, language="zh", mode="greedy", beam_size=3):
        mel = self._load_audio(audio_path)
        enc_in = self.msl.Tensor(mel.astype(np.float32))
        enc_out = []
        self.enc.predict([enc_in], enc_out)
        enc_hidden = enc_out[0].get_data_to_numpy()
        prompt = None
        if self.processor is not None:
            prompt = np.array([self.processor.get_decoder_prompt_ids(language=language, task="transcribe")], dtype=np.int32)
        else:
            prompt = np.array([[50258, 50259]], dtype=np.int32)
        if mode == "beam":
            ids = self._beam_decode(enc_hidden, prompt, beam_size)
        else:
            ids = self._greedy_decode(enc_hidden, prompt)
        text = ""
        try:
            from transformers import WhisperTokenizer
            tok = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
            text = tok.decode(ids[0])
        except Exception:
            try:
                import whisper
                from whisper.tokenizer import get_tokenizer
                tok2 = get_tokenizer(multilingual=True)
                text = tok2.decode(list(map(int, ids[0])))
            except Exception:
                pass
        return [{"segment_id": "00000", "start_ms": 0, "end_ms": int(30*1000), "text": text}]

def export_mindir(model_size, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperConfig
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
        config = WhisperConfig.from_pretrained(f"openai/whisper-{model_size}")
    except Exception:
        raise RuntimeError("transformers not available")
    enc = model.model.encoder
    dec = model.model.decoder
    audio = torch.randn(1, config.num_mel_bins, 3000)
    torch.onnx.export(enc, audio, str(out/"encoder.onnx"), input_names=["audio_features"], output_names=["encoder_hidden_states"], opset_version=13, dynamic_axes={"audio_features":{0:"batch"}, "encoder_hidden_states":{0:"batch"}})
    dec_in_ids = torch.randint(0, config.vocab_size, (1, 2), dtype=torch.int32)
    enc_hidden = torch.randn(1, config.max_source_positions, config.d_model)
    torch.onnx.export(dec, (dec_in_ids, enc_hidden), str(out/"decoder_init.onnx"), input_names=["decoder_input_ids", "encoder_hidden_states"], output_names=["logits"], opset_version=13, dynamic_axes={"decoder_input_ids":{0:"batch",1:"seq"}, "encoder_hidden_states":{0:"batch"}, "logits":{0:"batch",1:"seq"}})
    try:
        from mindspore.tools import converter
        converter(step="onnx_to_mindir", model_file=str(out/"encoder.onnx"), output_file=str(out/"encoder.mindir"))
        converter(step="onnx_to_mindir", model_file=str(out/"decoder_init.onnx"), output_file=str(out/"decoder_init.mindir"))
    except Exception:
        raise RuntimeError("mindspore.tools.converter not available")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--model_size", default="tiny")
    ap.add_argument("--out", default="models/mslite/tiny")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--audio")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--beam_size", type=int, default=3)
    args = ap.parse_args()
    if args.export:
        export_mindir(args.model_size, args.out)
        return
    if args.infer:
        asr = LiteWhisperASR(args.model_size, args.device, args.compute_type)
        res = asr.transcribe(args.audio, language="zh", mode=args.mode, beam_size=args.beam_size)
        print(json.dumps(res, ensure_ascii=False))

if __name__ == "__main__":
    main()
