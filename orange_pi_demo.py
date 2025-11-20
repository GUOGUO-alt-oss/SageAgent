#!/usr/bin/env python3
import argparse
import time
import json
import numpy as np
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

def overlay_subtitle(frame, text):
    if cv2 is None:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h-80), (w-10, h-10), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, text[:80], (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--sub", help="jsonl subtitles with start_ms,end_ms,text")
    args = ap.parse_args()
    if cv2 is None:
        print(json.dumps({"error": "opencv not installed"}))
        return
    cap = cv2.VideoCapture(args.video)
    subs = []
    with open(args.sub, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                subs.append(obj)
            except Exception:
                continue
    i = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = int((time.time() - start) * 1000)
        text = ""
        while i < len(subs) and subs[i].get("start_ms", 0) <= t_ms:
            if subs[i].get("end_ms", 0) >= t_ms:
                text = subs[i].get("text", "")
                break
            i += 1
        frame = overlay_subtitle(frame, text)
        cv2.imshow("OrangePi Subtitle Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

