"""
find_moments.py — combine AUDIO-ENERGY spikes + TRANSCRIPT dialogue into one
ranked list of "good moments" for vlog/gym/action video, then cut them.

Why separate from intro_finder.py: movies are dialogue-driven (subtitle -> quote).
Vlogs are moment-driven — the best parts are PR lifts, shouts, throwing hands,
hype — which are LOUD, not necessarily wordy. This fuses both tracks:

  energy track  : moments.energy_moments()  -> loud/intense windows (no words)
  dialogue track: whisper transcript -> hook-scored quotable lines
  fusion        : if a dialogue line lands inside a loud window, it gets a big
                  boost (funny line SAID during an intense moment = gold). Loud
                  windows with no words still stand on their own energy score.

Usage:
  python find_moments.py detect  --video "...webm" --out "D:/ai/clipcafe/out/tren/<name>"
  python find_moments.py clips   --video "...webm" --out "..." --top 40
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

from moments import energy_moments
from hooks import rank_segments
from subs import whisper_segments, get_segments
from cut import cut_clip


def _fmt_ts(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:n] or "moment"


def _overlaps(a_start, a_end, b_start, b_end) -> float:
    """seconds of overlap between two ranges"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def build_moments(video: str, cache_dir: str,
                  energy_boost: float = 0.35,
                  keep_pure_energy: int = 3) -> list[dict]:
    """
    DIALOGUE-FIRST fusion (vlog-tuned).

    Lesson from probing the gym vlogs: pure loudness is a bad "good moment"
    detector here — most dB spikes are background music / car talk, not PR lifts.
    What actually makes a hook clip on this channel is the TALK (banter, rage,
    trash-talk) plus the occasional real lift. So:

      - dialogue scenes are the primary candidates, ranked by hook score.
      - a scene that ALSO overlaps a loud spike gets a small `energy_boost`
        (loud + funny = better), but energy never creates a clip on its own...
      - ...except we keep the top `keep_pure_energy` genuinely loud, wordless
        windows as bonus candidates (a real silent lift/scream), clearly tagged.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # --- energy track (for boosting only) ---
    energy = energy_moments(video)
    for e in energy:
        e["escore"] = min(1.0, max(0.0, e["loud_over_baseline"] / 15.0))

    # --- dialogue track (whisper), grouped tight for vlog pacing ---
    seg_cache = cache / "segments.json"
    if seg_cache.exists():
        segs = json.loads(seg_cache.read_text(encoding="utf-8"))
    else:
        segs = whisper_segments(video)
        seg_cache.write_text(json.dumps(segs, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    scenes = rank_segments(segs, scene_gap=1.2, max_scene_dur=30.0) if segs else []

    moments: list[dict] = []
    for sc in scenes:
        boost = 0.0
        peak_db = None
        for e in energy:
            ov = _overlaps(sc["start"], sc["end"], e["start"], e["end"])
            if ov > 0 and e["escore"] > boost:
                boost = e["escore"]
                peak_db = e["peak_db"]
        # dialogue score is primary; loudness only nudges it up
        fused = sc["score"] + boost * energy_boost
        reason = "dialogue"
        if boost > 0.5:
            reason = "dialogue during a loud/intense moment"
        moments.append({
            "start": sc["start"], "end": sc["end"],
            "duration": sc["duration"], "score": round(fused, 4),
            "kind": "dialogue", "text": sc["text"],
            "peak_db": peak_db, "reason": reason,
        })

    # --- keep only the few loudest WORDLESS windows as bonus lift/scream clips ---
    wordless = []
    for e in energy:
        covered = any(_overlaps(e["start"], e["end"], m["start"], m["end"]) > 0.5
                      for m in moments if m["kind"] == "dialogue")
        if not covered:
            wordless.append(e)
    wordless.sort(key=lambda x: x["peak_db"], reverse=True)
    for e in wordless[:keep_pure_energy]:
        moments.append({
            "start": e["start"], "end": e["end"], "duration": e["duration"],
            "score": round(e["escore"] * 0.9, 4),
            "kind": "energy", "text": "",
            "peak_db": e["peak_db"],
            "reason": f"loud wordless moment +{e['loud_over_baseline']}dB (likely lift/scream)",
        })

    moments.sort(key=lambda x: x["score"], reverse=True)

    # --- 5. dedup overlapping moments: keep the highest-scoring, drop any later
    # moment that overlaps an already-kept one by more than 40% of its length.
    kept: list[dict] = []
    for m in moments:
        dup = False
        for k in kept:
            ov = _overlaps(m["start"], m["end"], k["start"], k["end"])
            if ov > 0.4 * min(m["duration"], k["duration"]):
                dup = True
                break
        if not dup:
            kept.append(m)
    moments = kept
    (cache / "moments.json").write_text(
        json.dumps(moments, ensure_ascii=False, indent=1), encoding="utf-8")
    return moments


def cmd_detect(a):
    ms = build_moments(a.video, a.out)
    dlg = sum(1 for m in ms if m["kind"] == "dialogue")
    eng = sum(1 for m in ms if m["kind"] == "energy")
    print(f"# {len(ms)} moments  ({dlg} dialogue, {eng} pure-energy)  -> {a.out}/moments.json\n")
    for i, m in enumerate(ms[:a.top], 1):
        tag = "🔊" if m["kind"] == "energy" else "💬"
        db = f" {m['peak_db']}dB" if m.get("peak_db") is not None else ""
        print(f"{i:>3}. [{m['score']:.3f}] {tag}{db} {_fmt_ts(m['start'])} ({m['duration']}s)")
        if m["text"]:
            print(f"      {m['text'][:90]}")
        else:
            print(f"      {m['reason']}")


def cmd_clips(a):
    cache = Path(a.out)
    mj = cache / "moments.json"
    if mj.exists() and not a.rebuild:
        ms = json.loads(mj.read_text(encoding="utf-8"))
    else:
        ms = build_moments(a.video, a.out)

    picks = [m for m in ms if 1.5 <= m["duration"] <= 40.0][:a.top]
    out_dir = cache / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    print(f"# cutting {len(picks)} moment clips -> {out_dir}\n")
    for i, m in enumerate(picks, 1):
        label = m["text"] if m["text"] else m["reason"]
        name = f"{i:02d}_{m['kind']}_{_slug(label)}.mp4"
        out_path = str(out_dir / name)
        try:
            cut_clip(a.video, m["start"], m["end"], out_path)
            status = "ok"
        except Exception as ex:
            status = f"FAILED: {ex}"
        tag = "energy" if m["kind"] == "energy" else "dialogue"
        print(f"{i:>3}. [{m['score']:.3f}] {tag} {_fmt_ts(m['start'])} ({m['duration']}s) -> {name}  {status}")
        manifest.append({
            "rank": i, "score": m["score"], "kind": m["kind"],
            "start": m["start"], "end": m["end"], "duration": m["duration"],
            "peak_db": m.get("peak_db"), "text": m["text"],
            "reason": m["reason"], "file": name,
        })
    (out_dir / "clips.json").write_text(
        json.dumps({"video": Path(a.video).name, "clips": manifest},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nmanifest: {out_dir / 'clips.json'}")


def main():
    p = argparse.ArgumentParser(description="find good moments (energy + dialogue) in vlog/action video")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("detect")
    sp.add_argument("--video", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--top", type=int, default=40)

    sp = sub.add_parser("clips")
    sp.add_argument("--video", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--top", type=int, default=40)
    sp.add_argument("--rebuild", action="store_true")

    a = p.parse_args()
    {"detect": cmd_detect, "clips": cmd_clips}[a.cmd](a)


if __name__ == "__main__":
    main()
