"""
subs.py — get a dialogue transcript for a movie.

Two sources, in priority order:
  1. OpenSubtitles REST API (api.opensubtitles.com) — official subtitles (.srt),
     the same source that makes clip.cafe accurate. Needs an API key.
  2. faster-whisper — local, free, no key. Transcribes the movie's own audio.
     Used when no official subtitle is found or no API key is set.

Output is a normalized list of dialogue segments:
    [{"start": float_sec, "end": float_sec, "text": "..."}]

Usage:
    from subs import get_segments
    segs = get_segments(video_path="D:/.../movie.mp4",
                        title="The Wolf of Wall Street", year=2013)
"""
from __future__ import annotations

import os
import re
import gzip
import json
import tempfile
from pathlib import Path

import requests

OS_API = "https://api.opensubtitles.com/api/v1"
# Set your key in env: OPENSUBTITLES_API_KEY   (free at opensubtitles.com/consumers)
OS_KEY = os.environ.get("OPENSUBTITLES_API_KEY", "").strip()
OS_USER_AGENT = "clipcafe-proto/0.1"

# --- promo / non-dialogue junk filters -------------------------------------
_JUNK_PATTERNS = [
    re.compile(r"opensubtitles", re.I),
    re.compile(r"www\.", re.I),
    re.compile(r"https?://", re.I),
    re.compile(r"subtitle[sd]?\s+by", re.I),
    re.compile(r"sync(hroni[sz]ed)?\s+by", re.I),
    re.compile(r"\bripped?\s+by\b", re.I),
    re.compile(r"advertise your product", re.I),
    re.compile(r"api\.opensubtitles", re.I),
]


def _is_junk(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    return any(p.search(t) for p in _JUNK_PATTERNS)


def _clean(text: str) -> str:
    # strip srt formatting tags, speaker cues, music notes
    t = re.sub(r"<[^>]+>", "", text)          # <i>...</i>
    t = re.sub(r"\{[^}]+\}", "", t)            # {\an8}
    t = re.sub(r"^\s*[-–]\s*", "", t)          # leading dash for dialogue
    t = t.replace("♪", "").strip()
    return t


# --- SRT parsing ------------------------------------------------------------
_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def _ts_to_sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_text: str) -> list[dict]:
    segs: list[dict] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        m = None
        text_lines = []
        for ln in lines:
            mm = _TS.search(ln)
            if mm and m is None:
                m = mm
            elif m is not None:
                text_lines.append(ln)
        if not m or not text_lines:
            continue
        start = _ts_to_sec(*m.group(1, 2, 3, 4))
        end = _ts_to_sec(*m.group(5, 6, 7, 8))
        text = _clean(" ".join(text_lines))
        if _is_junk(text):
            continue
        segs.append({"start": start, "end": end, "text": text})
    return segs


# --- OpenSubtitles path -----------------------------------------------------
def _os_headers() -> dict:
    return {
        "Api-Key": OS_KEY,
        "User-Agent": OS_USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def opensubtitles_srt(title: str, year: int | None, lang: str = "en") -> str | None:
    """Return raw SRT text from OpenSubtitles, or None if unavailable."""
    if not OS_KEY:
        return None
    try:
        params = {"query": title, "languages": lang, "order_by": "download_count"}
        if year:
            params["year"] = year
        r = requests.get(f"{OS_API}/subtitles", params=params,
                         headers=_os_headers(), timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        file_id = data[0]["attributes"]["files"][0]["file_id"]
        # request a download link
        dl = requests.post(f"{OS_API}/download", json={"file_id": file_id},
                          headers=_os_headers(), timeout=30)
        dl.raise_for_status()
        link = dl.json()["link"]
        raw = requests.get(link, timeout=60)
        raw.raise_for_status()
        content = raw.content
        if content[:2] == b"\x1f\x8b":  # gzipped
            content = gzip.decompress(content)
        return content.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[subs] OpenSubtitles failed: {e}")
        return None


# --- demucs vocal separation ------------------------------------------------
def _demucs_vocals(video_path: str) -> str | None:
    """Separate vocals from the video's audio with Demucs (CPU, small segment)
    so whisper hears clean speech with no music. Returns path to a vocals wav,
    or None on failure. Low-RAM via --segment (default 7s). Cached beside input
    as <stem>_vocals.wav so re-runs skip the (slow) separation.
    """
    import subprocess, sys, tempfile, shutil
    stem = Path(video_path).stem
    cache = Path(video_path).parent / (stem + "_vocals.wav")
    if cache.exists() and cache.stat().st_size > 0:
        print(f"[subs] demucs: using cached vocals -> {cache}")
        return str(cache)
    model = os.environ.get("CLIPCAFE_DEMUCS_MODEL", "htdemucs")
    segment = os.environ.get("CLIPCAFE_DEMUCS_SEGMENT", "7")
    scratch = os.environ.get("KIROCREW_SCRATCH") or tempfile.gettempdir()
    work = tempfile.mkdtemp(prefix="demucs_", dir=scratch)
    print(f"[subs] demucs separating vocals (model={model} segment={segment}s)...")
    try:
        cmd = [sys.executable, "-m", "demucs", "-n", model,
               "--two-stems", "vocals", "--segment", str(segment),
               "-d", "cpu", "-o", work, video_path]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[subs] demucs failed rc={r.returncode}; falling back to raw audio")
            return None
        voc = Path(work) / model / stem / "vocals.wav"
        if not voc.is_file():
            for root, _, files in os.walk(work):
                if "vocals.wav" in files:
                    voc = Path(root) / "vocals.wav"
                    break
        if not Path(voc).is_file():
            print("[subs] demucs produced no vocals.wav; falling back to raw audio")
            return None
        shutil.move(str(voc), str(cache))
        return str(cache)
    except Exception as e:
        print(f"[subs] demucs error: {e}; falling back to raw audio")
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- faster-whisper fallback ------------------------------------------------
def _run_whisper(model, audio_path, vocal_only):
    """Transcribe one audio file; apply confidence filter unless vocal_only.
    Returns (segments_list, mean_logprob, mean_no_speech) for auto-trigger.
    """
    if vocal_only:
        vad_params = {"min_silence_duration_ms": 300, "speech_pad_ms": 300}
    else:
        vad_params = {"min_silence_duration_ms": 500, "speech_pad_ms": 200}
    segments, _ = model.transcribe(
        audio_path, language="en", beam_size=5, vad_filter=True,
        vad_parameters=vad_params, condition_on_previous_text=False,
    )
    max_no_speech = float(os.environ.get("CLIPCAFE_MAX_NO_SPEECH", "0.6"))
    min_logprob = float(os.environ.get("CLIPCAFE_MIN_LOGPROB", "-1.0"))
    out, dropped = [], 0
    alps, nsps = [], []
    for s in segments:
        text = _clean(s.text)
        nsp = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
        alp = float(getattr(s, "avg_logprob", 0.0) or 0.0)
        alps.append(alp); nsps.append(nsp)
        if _is_junk(text):
            continue
        if not vocal_only and (nsp > max_no_speech or alp < min_logprob):
            dropped += 1
            continue
        out.append({
            "start": float(s.start), "end": float(s.end), "text": text,
            "no_speech_prob": round(nsp, 3), "avg_logprob": round(alp, 3),
        })
    mean_alp = sum(alps) / len(alps) if alps else 0.0
    mean_nsp = sum(nsps) / len(nsps) if nsps else 0.0
    if vocal_only:
        print(f"[subs] vocal-only: kept {len(out)} segments (confidence filter OFF)")
    else:
        print(f"[subs] confidence filter: kept {len(out)}, dropped {dropped} "
              f"(no_speech>{max_no_speech} or logprob<{min_logprob}); "
              f"mean_logprob={mean_alp:.2f} mean_no_speech={mean_nsp:.2f}")
    return out, mean_alp, mean_nsp


def whisper_segments(video_path: str, model_size: str | None = None) -> list[dict]:
    from faster_whisper import WhisperModel

    # Three paths for handling music-vs-speech:
    #  1. VOCAL-ONLY (manual, CLIPCAFE_VOCAL_ONLY=1): always Demucs-separate first,
    #     transcribe clean vocals, confidence filter OFF, loosened VAD. For gym/vlog.
    #  2. AUTO (CLIPCAFE_VOCAL_AUTO=1): transcribe raw first (fast); if the mean
    #     confidence is low (dialogue buried under score/noise), AUTO-rerun
    #     vocal-only. Best of both — clean films stay fast, noisy ones get boosted.
    #  3. DEFAULT: raw audio + confidence filter (films are mixed for clear dialogue).
    vocal_only = os.environ.get("CLIPCAFE_VOCAL_ONLY", "1") == "1"
    vocal_auto = os.environ.get("CLIPCAFE_VOCAL_AUTO", "0") == "1"

    model_size = model_size or os.environ.get("CLIPCAFE_FW_MODEL", "small")
    print(f"[subs] transcribing with faster-whisper ({model_size}) — this takes a while...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Path 1: explicit vocal-only
    if vocal_only:
        voc = _demucs_vocals(video_path)
        if voc:
            print(f"[subs] vocal-only mode: transcribing isolated vocals")
            out, _, _ = _run_whisper(model, voc, vocal_only=True)
            return out
        print(f"[subs] vocal-only requested but separation failed; using raw audio + filter")

    # Path 2/3: raw first
    out, mean_alp, mean_nsp = _run_whisper(model, video_path, vocal_only=False)

    # Path 2: auto-trigger — raw confidence poor => dialogue buried => Demucs rerun.
    # Thresholds: mean_logprob below -0.85 OR mean_no_speech above 0.45 signals a
    # noisy mix where separation will materially help. Tunable via env.
    if vocal_auto and not vocal_only:
        auto_alp = float(os.environ.get("CLIPCAFE_AUTO_LOGPROB", "-0.85"))
        auto_nsp = float(os.environ.get("CLIPCAFE_AUTO_NO_SPEECH", "0.45"))
        if mean_alp < auto_alp or mean_nsp > auto_nsp:
            print(f"[subs] AUTO-TRIGGER: raw transcript weak "
                  f"(mean_logprob={mean_alp:.2f} / mean_no_speech={mean_nsp:.2f}) "
                  f"-> re-running vocal-only via Demucs")
            voc = _demucs_vocals(video_path)
            if voc:
                out2, _, _ = _run_whisper(model, voc, vocal_only=True)
                if len(out2) >= len(out):   # keep the richer transcript
                    print(f"[subs] AUTO: vocal-only won ({len(out2)} vs {len(out)} segments)")
                    return out2
                print(f"[subs] AUTO: raw kept ({len(out)} >= {len(out2)} vocal segments)")
            else:
                print(f"[subs] AUTO: separation failed; keeping raw transcript")
    return out


# --- public entry -----------------------------------------------------------
def get_segments(video_path: str, title: str, year: int | None = None,
                 lang: str = "en", cache_dir: str | None = None,
                 force_whisper: bool = False) -> tuple[list[dict], str]:
    """
    Returns (segments, source_label). Caches the parsed segments as JSON.
    source_label is "opensubtitles" or "faster-whisper".
    """
    cache_dir = cache_dir or str(Path(video_path).parent)
    cache = Path(cache_dir) / (Path(video_path).stem + ".segments.json")
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["segments"], data["source"]

    segs, source = [], ""
    if not force_whisper:
        srt = opensubtitles_srt(title, year, lang)
        if srt:
            segs = parse_srt(srt)
            source = "opensubtitles"

    if not segs:
        segs = whisper_segments(video_path)
        source = "faster-whisper"

    cache.write_text(json.dumps({"source": source, "segments": segs},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return segs, source
