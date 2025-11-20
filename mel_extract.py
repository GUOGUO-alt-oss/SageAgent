#!/usr/bin/env python3
import argparse
import json
import math
import numpy as np
import soundfile as sf
from pathlib import Path

def hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        l = bins[i]
        c = bins[i + 1]
        r = bins[i + 2]
        if c > l:
            fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb

def stft(y, n_fft=400, hop_length=160, win_length=400):
    y = np.asarray(y, dtype=np.float32)
    w = np.hanning(win_length).astype(np.float32)
    n_frames = 1 + (len(y) - win_length) // hop_length if len(y) >= win_length else 1
    frames = np.zeros((n_frames, win_length), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + win_length
        seg = np.zeros(win_length, dtype=np.float32)
        seg[:max(0, min(win_length, len(y) - start))] = y[start:end]
        frames[i] = seg * w
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    return spec

def log_mel_spectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=80):
    spec = stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = np.dot(power, fb.T)
    mel = np.maximum(mel, 1e-10)
    mel = np.log10(mel)
    mel = mel.T
    return mel

def pad_or_trim(y, length):
    if len(y) > length:
        return y[:length]
    if len(y) < length:
        out = np.zeros(length, dtype=np.float32)
        out[:len(y)] = y
        return out
    return y

def load_audio(path, sr=16000):
    y, s = sf.read(path, dtype="float32")
    if s != sr:
        import resampy
        y = resampy.resample(y, s, sr)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    y = load_audio(args.audio, sr=16000)
    y = pad_or_trim(y, 16000 * 30)
    mel = log_mel_spectrogram(y, sr=16000, n_fft=400, hop_length=160, n_mels=80)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, mel.astype(np.float32))
    print(json.dumps({"shape": mel.shape}))

if __name__ == "__main__":
    main()

