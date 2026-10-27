"""
batch_tren.py — run the dialogue-first moment pipeline over every video in the
"tren yt" folder, sequentially (host RAM is tight; no parallelism).

For each video:
  out/tren/<slug>/segments.json  (whisper transcript, cached — reused if present)
  out/tren/<slug>/moments.json   (ranked moments)
  out/tren/<slug>/clips/*.mp4     (cut clips + clips.json manifest)

Progress is printed AND appended to out/tren/_batch.log so a background run
can be tailed.
"""
from __future__ import annotations

import re
import sys
import time
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from find_moments import build_moments
from cut import cut_clip

SRC = Path(r"D:\caaapct\Edits\tren yt")
OUT = Path(r"D:\ai\clipcafe\out\tren")
LOG = OUT / "_batch.log"
TOP = 25
VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov"}


def _slug(text: str, n: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:n] or "video"


def _clip_slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:n] or "moment"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def process(video: Path):
    slug = _slug(video.stem)
    cache = OUT / slug
    clip_dir = cache / "clips"
    done_marker = cache / "clips" / "clips.json"
    if done_marker.exists():
        log(f"SKIP (already done): {video.name}")
        return
    log(f"START: {video.name}")
    t0 = time.time()
    moments = build_moments(str(video), str(cache))
    picks = [m for m in moments if 1.5 <= m["duration"] <= 40.0][:TOP]
    clip_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    ok = 0
    for i, m in enumerate(picks, 1):
        label = m["text"] if m["text"] else m["reason"]
        name = f"{i:02d}_{m['kind']}_{_clip_slug(label)}.mp4"
        try:
            cut_clip(str(video), m["start"], m["end"], str(clip_dir / name))
            ok += 1
        except Exception as e:
            log(f"   clip {i} FAILED: {e}")
            continue
        manifest.append({
            "rank": i, "score": m["score"], "kind": m["kind"],
            "start": m["start"], "end": m["end"], "duration": m["duration"],
            "peak_db": m.get("peak_db"), "text": m["text"],
            "reason": m["reason"], "file": name,
        })
    (clip_dir / "clips.json").write_text(
        json.dumps({"video": video.name, "clips": manifest},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    dlg = sum(1 for m in picks if m["kind"] == "dialogue")
    eng = sum(1 for m in picks if m["kind"] == "energy")
    log(f"DONE: {video.name} — {ok} clips ({dlg} dialogue, {eng} lift) in {time.time()-t0:.0f}s")


def main():
    vids = sorted(p for p in SRC.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    log(f"==== BATCH START: {len(vids)} videos ====")
    for i, v in enumerate(vids, 1):
        log(f"---- [{i}/{len(vids)}] ----")
        try:
            process(v)
        except Exception:
            log(f"ERROR on {v.name}:\n{traceback.format_exc()}")
    log("==== BATCH COMPLETE ====")


if __name__ == "__main__":
    main()
