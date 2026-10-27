"""
moments.py — find GOOD MOMENTS in a video, not just dialogue.

clipcafe's hook ranking is 100% text-based (subtitle -> quotable line). That works
for movies but MISSES the best parts of a gym / vlog / action video: a PR deadlift
with a scream, throwing hands, an explosive hype moment — those are LOUD, not wordy.

This module adds an AUDIO-ENERGY track: it measures short-window loudness (RMS)
across the whole video with ffmpeg, finds the spikes (sustained bursts well above
the video's own baseline), and turns each into a candidate clip window. Combined
with the whisper transcript, the pipeline now ranks on BOTH what is said and what
EXPLODES — so a silent-but-intense PR lift ranks alongside a funny line.

No external model. Pure ffmpeg (astats per-frame) + numpy-free math.

Public entry:
    energy_moments(video, ...) -> [{"start","end","peak_db","mean_db","kind":"energy"}]
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _extract_frame_db(video: str, win: float = 0.5) -> list[tuple[float, float]]:
    """
    Return [(t_seconds, rms_db), ...] sampled every `win` seconds across the video.

    Uses ffmpeg's astats filter with metadata injection: we resample audio into
    `win`-second frames and read each frame's RMS level (dBFS) from stderr.
    Silence ~ -70..-90 dB; loud shouting/impact ~ -15..-5 dB.
    """
    # asetnsamples groups audio into fixed windows; astats prints per-frame RMS.
    sr = 16000
    nsamp = int(sr * win)
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", video,
        "-map", "0:a:0",
        "-af",
        f"aresample={sr},asetnsamples=n={nsamp}:p=0,"
        f"astats=metadata=1:reset=1,"
        f"ametadata=print:key=lavfi.astats.Overall.RMS_level",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stderr

    # ametadata prints lines like:
    #   frame:123  pts:... pts_time:61.5
    #   lavfi.astats.Overall.RMS_level=-23.145
    times: list[float] = []
    dbs: list[float] = []
    cur_t = None
    for line in out.splitlines():
        mt = re.search(r"pts_time:([\d.]+)", line)
        if mt:
            cur_t = float(mt.group(1))
            continue
        md = re.search(r"RMS_level=(-?[\d.]+|-?inf)", line)
        if md and cur_t is not None:
            val = md.group(1)
            db = -90.0 if "inf" in val else float(val)
            times.append(cur_t)
            dbs.append(db)
            cur_t = None
    return list(zip(times, dbs))


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def energy_moments(video: str, win: float = 0.5,
                   thresh_pct: float = 88.0,
                   min_dur: float = 1.5, max_dur: float = 30.0,
                   merge_gap: float = 1.5,
                   pre: float = 2.5, post: float = 2.0) -> list[dict]:
    """
    Detect loudness spikes and return candidate moment windows.

    - thresh_pct : a frame counts as "hot" if its dB is above this percentile of
                   the whole video's loudness (adaptive per-video, so a quiet vlog
                   and a loud one both work).
    - merge_gap  : hot frames closer than this merge into one moment.
    - pre / post : context padding so a clip starts a beat before the spike and
                   ends after it (a PR lift needs the wind-up + the scream).
    """
    frames = _extract_frame_db(video, win=win)
    if not frames:
        return []

    dbs = [d for _, d in frames]
    thresh = _percentile(dbs, thresh_pct)
    baseline = _percentile(dbs, 50.0)

    # collect hot frames
    hot = [(t, d) for (t, d) in frames if d >= thresh]
    if not hot:
        return []

    # merge consecutive hot frames into moments
    moments: list[dict] = []
    cur_start = hot[0][0]
    cur_end = hot[0][0] + win
    peak = hot[0][1]
    dbsum = hot[0][1]
    cnt = 1
    for t, d in hot[1:]:
        if t - cur_end <= merge_gap:
            cur_end = t + win
            peak = max(peak, d)
            dbsum += d
            cnt += 1
        else:
            moments.append({"start": cur_start, "end": cur_end,
                            "peak_db": peak, "mean_db": dbsum / cnt})
            cur_start, cur_end, peak, dbsum, cnt = t, t + win, d, d, 1
    moments.append({"start": cur_start, "end": cur_end,
                    "peak_db": peak, "mean_db": dbsum / cnt})

    # pad + length-filter + shape output
    out: list[dict] = []
    total = frames[-1][0]
    for m in moments:
        st = max(0.0, m["start"] - pre)
        en = min(total, m["end"] + post)
        dur = en - st
        if dur < min_dur:
            continue
        if dur > max_dur:
            en = st + max_dur
            dur = max_dur
        out.append({
            "start": round(st, 2), "end": round(en, 2),
            "duration": round(dur, 2),
            "peak_db": round(m["peak_db"], 1),
            "mean_db": round(m["mean_db"], 1),
            "loud_over_baseline": round(m["peak_db"] - baseline, 1),
            "kind": "energy",
        })
    out.sort(key=lambda x: x["peak_db"], reverse=True)
    return out


if __name__ == "__main__":
    import sys, json
    v = sys.argv[1]
    ms = energy_moments(v)
    print(f"{len(ms)} energy moments (loudest first)")
    for m in ms[:30]:
        print(f"  {m['start']:>8.1f}-{m['end']:<8.1f} peak={m['peak_db']}dB "
              f"(+{m['loud_over_baseline']} over baseline) {m['duration']}s")
    print(json.dumps(ms, indent=1))
