"""
intro_finder.py — a local, offline "clip.cafe" for one movie.

Pipeline: subtitle/transcript -> merged dialogue candidates -> hook ranking ->
search by dialogue OR export the top-N most quotable clips as mp4 files.

Commands
--------
  index   : build the dialogue segment cache (OpenSubtitles or faster-whisper)
  rank    : print the top-N hook-ranked lines (no video cutting)
  search  : find lines matching a query, ranked by hook score
  clips   : cut the top-N hook clips to mp4 (this is the clip.cafe-style output)

Examples
--------
  python intro_finder.py index  --video "D:/ai/IdlixDownloader/The Wolf of Wall Street.mp4" --title "The Wolf of Wall Street" --year 2013
  python intro_finder.py rank   --video "...mp4" --top 30
  python intro_finder.py search --video "...mp4" --query "sell me this pen"
  python intro_finder.py clips  --video "...mp4" --top 25 --out "D:/ai/clipcafe/out/wow"
"""
from __future__ import annotations

import re
import os
import sys
import json
import argparse
from pathlib import Path

from subs import get_segments, parse_srt
from hooks import rank_segments, score_segment
from cut import cut_clip

try:
    from intro import rank_segments_intro, get_intro_duration_range
    _INTRO_AVAILABLE = True
except ImportError:
    _INTRO_AVAILABLE = False


def _read_srt_file(path: str) -> list[dict]:
    from pathlib import Path
    raw = Path(path).read_bytes()
    # srt files are often latin-1/cp1252; try utf-8 then fall back
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return parse_srt(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return parse_srt(raw.decode("latin-1", errors="replace"))


def _fmt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "clip"


def load_ranked(video: str, title: str, year: int | None, force_whisper: bool,
                srt: str | None = None, scorer: str = "heuristic"):
    if srt:
        segs = _read_srt_file(srt)
        source = f"srt:{__import__('pathlib').Path(srt).name}"
    else:
        segs, source = get_segments(video, title, year, force_whisper=force_whisper)
    
    if scorer == "intro" and _INTRO_AVAILABLE:
        ranked = rank_segments_intro(segs)
        source += " (intro scorer)"
    else:
        ranked = rank_segments(segs)
    return ranked, source


def cmd_index(a):
    if a.srt:
        segs = _read_srt_file(a.srt)
        source = f"srt:{Path(a.srt).name}"
    else:
        segs, source = get_segments(a.video, a.title, a.year, force_whisper=a.force_whisper)
    
    if a.scorer == "intro" and _INTRO_AVAILABLE:
        ranked = rank_segments_intro(segs)
        source += " (intro scorer)"
    else:
        ranked = rank_segments(segs)
    
    print(f"source       : {source}")
    print(f"raw cues     : {len(segs)}")
    print(f"merged lines : {len(ranked)}")


def cmd_rank(a):
    ranked, source = load_ranked(a.video, a.title, a.year, a.force_whisper, a.srt, a.scorer)
    print(f"# source={source}  candidates={len(ranked)}  showing top {a.top}\n")
    for i, r in enumerate(ranked[:a.top], 1):
        print(f"{i:>3}. [{r['score']:.3f}] {_fmt_ts(r['start'])}  ({r['duration']}s)")
        print(f"     {r['text']}")


def cmd_search(a):
    ranked, source = load_ranked(a.video, a.title, a.year, a.force_whisper, a.srt, a.scorer)
    q = a.query.lower()
    q_words = set(re.findall(r"[a-z']+", q))
    hits = []
    for r in ranked:
        tl = r["text"].lower()
        if q in tl:
            rel = 2.0
        else:
            tw = set(re.findall(r"[a-z']+", tl))
            overlap = len(q_words & tw) / max(1, len(q_words))
            if overlap < 0.5:
                continue
            rel = overlap
        hits.append((rel + r["score"], r))
    hits.sort(key=lambda x: x[0], reverse=True)
    print(f"# query={a.query!r}  source={source}  {len(hits)} matches\n")
    for combined, r in hits[:a.top]:
        print(f"[{r['score']:.3f}] {_fmt_ts(r['start'])}  ({r['duration']}s)")
        print(f"   {r['text']}")


def cmd_export(a):
    ranked, source = load_ranked(a.video, a.title, a.year, a.force_whisper, a.srt, a.scorer)
    
    # filter to sensible lengths
    if a.scorer == "intro" and _INTRO_AVAILABLE:
        min_dur, max_dur = get_intro_duration_range()
    else:
        min_dur, max_dur = 1.5, 70.0
    
    picks = [r for r in ranked if min_dur <= r["duration"] <= max_dur][:a.top]
    out = []
    for i, r in enumerate(picks):
        cues = [{"start": round(c["start"], 2), "end": round(c["end"], 2)}
                for c in r.get("cues", [])]
        out.append({
            "id": i,
            "start": round(r["start"], 2),
            "end": round(r["end"], 2),
            "last_cue_end": cues[-1]["end"] if cues else round(r["end"], 2),
            "cues": cues,
            "duration": r["duration"],
            "heuristic_score": r["score"],
            "text": r["text"],
        })
    Path(a.outfile).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    print(f"exported {len(out)} scenes -> {a.outfile}  (source={source})")


def cmd_cut_reranked(a):
    """Cut clips in the AI subagent's reranked order (reranked.json + scenes.json)."""
    rr = json.loads(Path(a.reranked).read_text(encoding="utf-8"))
    # Accept either a bare list, or an object wrapping the list under
    # "ranked" / "clips" / "items" (subagents often add a movie header).
    if isinstance(rr, dict):
        for key in ("ranked", "clips", "items"):
            if isinstance(rr.get(key), list):
                rr = rr[key]
                break
        else:
            raise SystemExit("reranked.json is an object with no ranked/clips/items list")
    scenes = {s["id"]: s for s in json.loads(Path(a.scenes).read_text(encoding="utf-8"))}

    # AUDIO MODE: default = music-free clips (vocal-only audio), because the user
    # adds their own music. Use the Demucs vocals.wav (same timeline) as the clip
    # audio while keeping video from the original. Set CLIPCAFE_CLIP_WITH_MUSIC=1
    # to keep the original mixed audio instead.
    audio_source = None
    if os.environ.get("CLIPCAFE_CLIP_WITH_MUSIC", "0") != "1":
        voc = str(Path(a.video).with_name(Path(a.video).stem + "_vocals.wav"))
        if Path(voc).is_file():
            audio_source = voc
            print(f"# audio: VOCAL-ONLY (no music) from {voc}")
        else:
            print(f"# audio: original mix (no vocals.wav found at {voc})")
    else:
        print("# audio: original mix (CLIPCAFE_CLIP_WITH_MUSIC=1)")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rr = sorted(rr, key=lambda x: x.get("rank", 999))[:a.top]
    manifest = []
    print(f"# cutting {len(rr)} AI-reranked clips -> {out_dir}\n")
    for r in rr:
        sc = scenes.get(r["id"])
        if not sc:
            continue
        rank = r.get("rank", 0)
        # hook_start (from the rubric) trims a skippable opening; tail-trim at the
        # last cue's end so the clip stops when the dialogue stops, not 40s later
        # when scene-merge bled into the next shot's silence.
        hook_start = float(r.get("hook_start", 0) or 0)
        clip_start = sc["start"] + hook_start
        hook_len = float(r.get("hook_len", 0) or 0)
        if hook_len > 0:
            # cut only the hook moment (3-10s typical), not the whole scene.
            clip_end = min(sc["end"], clip_start + hook_len)
            # AUTO-SNAP: never end mid-sentence. If clip_end lands INSIDE a spoken
            # cue, push it to that cue's end (+0.8s breath) so the peak line always
            # finishes. Snaps forward at most ~4s; beyond that, trust the target.
            for c in sc.get("cues", []):
                cs, ce = c.get("start", 0), c.get("end", 0)
                if cs < clip_end < ce and (ce - clip_end) <= 4.0:
                    clip_end = min(sc["end"], ce + 0.8)
                    break
        else:
            # fallback: tail-trim at the last spoken cue (whole-scene behaviour)
            tail = sc.get("last_cue_end", sc["end"])
            clip_end = min(sc["end"], tail + 1.2)
        if clip_end <= clip_start:
            clip_end = sc["end"]
        name = f"{rank:02d}_{_slug(sc['text'])}.mp4"
        out_path = str(out_dir / name)
        try:
            cut_clip(a.video, clip_start, clip_end, out_path, audio_source=audio_source)
            status = "ok"
        except Exception as e:
            status = f"FAILED: {e}"
        dur = round(clip_end - clip_start, 2)
        print(f"{rank:>3}. [{r.get('clip_score', 0):.2f}] {_fmt_ts(clip_start)} ({dur}s) -> {name}  {status}")
        print(f"      {r.get('reason','')}")
        manifest.append({
            "rank": rank, "clip_score": r.get("clip_score"), "reason": r.get("reason"),
            "start": round(clip_start, 2), "end": round(clip_end, 2), "duration": dur,
            "scene_end": sc["end"], "text": sc["text"], "file": name,
        })
    (out_dir / "clips.json").write_text(
        json.dumps({"source": "ai-reranked", "clips": manifest}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\nmanifest: {out_dir / 'clips.json'}")


def cmd_clips(a):
    ranked, source = load_ranked(a.video, a.title, a.year, a.force_whisper, a.srt, a.scorer)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # filter to sensible clip lengths
    if a.scorer == "intro" and _INTRO_AVAILABLE:
        # intro scorer ranks by learned similarity, no duration filter
        min_dur, max_dur = 0.0, 999.0
    else:
        min_dur, max_dur = 1.5, 70.0
    
    picks = [r for r in ranked if min_dur <= r["duration"] <= max_dur][:a.top]

    manifest = []
    print(f"# source={source}  cutting {len(picks)} clips -> {out_dir}\n")
    for i, r in enumerate(picks, 1):
        name = f"{i:02d}_{_slug(r['text'])}.mp4"
        out_path = str(out_dir / name)
        clip_end = r.get("hook_end", r["end"])
        try:
            cut_clip(a.video, r["start"], clip_end, out_path)
            status = "ok"
        except Exception as e:
            status = f"FAILED: {e}"
        print(f"{i:>3}. [{r['score']:.3f}] {_fmt_ts(r['start'])} ({r['duration']}s) -> {name}  {status}")
        manifest.append({
            "rank": i, "score": r["score"], "start": r["start"],
            "hook_end": clip_end, "scene_end": r.get("scene_end", r["end"]),
            "duration": r["duration"], "text": r["text"], "file": name,
            "signals": r["signals"],
        })

    (out_dir / "clips.json").write_text(
        json.dumps({"source": source, "clips": manifest}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\nmanifest: {out_dir / 'clips.json'}")


def main():
    p = argparse.ArgumentParser(description="local clip.cafe for one movie")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--video", required=True)
        sp.add_argument("--title", default="")
        sp.add_argument("--year", type=int, default=None)
        sp.add_argument("--srt", default=None,
                        help="use this .srt file directly (skips API + whisper)")
        sp.add_argument("--force-whisper", action="store_true",
                        help="skip OpenSubtitles, transcribe locally")
        sp.add_argument("--scorer", default="heuristic", choices=["heuristic", "intro"],
                        help="scoring method: heuristic (default) or intro (corpus-based)")

    sp = sub.add_parser("index"); common(sp)
    sp = sub.add_parser("rank"); common(sp); sp.add_argument("--top", type=int, default=30)
    sp = sub.add_parser("search"); common(sp)
    sp.add_argument("--query", required=True); sp.add_argument("--top", type=int, default=15)
    sp = sub.add_parser("clips"); common(sp)
    sp.add_argument("--top", type=int, default=25)
    sp.add_argument("--out", default="D:/ai/clipcafe/out")
    sp = sub.add_parser("export"); common(sp)
    sp.add_argument("--top", type=int, default=150)
    sp.add_argument("--outfile", default="D:/ai/clipcafe/out/scenes.json")
    sp = sub.add_parser("cut-reranked")
    sp.add_argument("--video", required=True)
    sp.add_argument("--reranked", default="D:/ai/clipcafe/out/reranked.json")
    sp.add_argument("--scenes", default="D:/ai/clipcafe/out/scenes.json")
    sp.add_argument("--top", type=int, default=30)
    sp.add_argument("--out", default="D:/ai/clipcafe/out/wow_ai")

    a = p.parse_args()
    {"index": cmd_index, "rank": cmd_rank, "search": cmd_search,
     "clips": cmd_clips, "export": cmd_export, "cut-reranked": cmd_cut_reranked}[a.cmd](a)


if __name__ == "__main__":
    main()
