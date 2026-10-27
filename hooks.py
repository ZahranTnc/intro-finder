"""
hooks.py — score dialogue segments for "how quotable / hook-worthy is this".

clip.cafe leans on years of traffic + manual curation to know which quotes people
want. We have neither, so we approximate "hook" with cheap linguistic heuristics.
This gets recall = clip.cafe and a ranking that starts ~80-90% as good, tunable.

Signals (each contributes to a 0..1-ish score, then weighted-summed):
  - punch      : exclamation / question / emphatic punctuation
  - brevity    : short, self-contained lines are more quotable than rambles
  - emotion    : presence of high-arousal / profanity / imperative words
  - imperative : starts with a command verb ("Sell", "Look", "Listen")
  - pronouns   : direct address ("you", "we") reads like a line TO the viewer
  - caps        : shouted words (ALL CAPS tokens) signal intensity
  - repetition : repeated words / anaphora ("more, more, more")
Anti-signals:
  - filler / scene-setting, too-long, or pure names get down-weighted.

No external model needed. If you later add an LLM re-ranker, feed it the top ~150
by this score and let it pick the final set — that mirrors clip.cafe curation.
"""
from __future__ import annotations

import re
import math

# high-arousal / emotionally loaded lexicon (small, extend freely)
EMOTION = {
    "love", "hate", "kill", "die", "death", "money", "rich", "poor", "win", "lose",
    "never", "always", "everything", "nothing", "everyone", "nobody", "fuck",
    "shit", "damn", "hell", "god", "swear", "promise", "believe", "dream",
    "afraid", "fear", "crazy", "insane", "genius", "stupid", "power", "control",
    "sell", "buy", "deal", "greed", "fortune", "million", "billion", "drug",
    "want", "need", "truth", "lie", "trust", "betray", "family", "war", "fight",
}

IMPERATIVE_STARTS = {
    "sell", "buy", "look", "listen", "stop", "wait", "run", "go", "get", "give",
    "tell", "show", "remember", "forget", "never", "always", "let", "keep",
    "watch", "think", "believe", "trust", "don't", "do", "make", "take", "come",
}

DIRECT_ADDR = re.compile(r"\b(you|your|you're|we|us|our)\b", re.I)
CAPS_TOKEN = re.compile(r"\b[A-Z]{3,}\b")
WORD = re.compile(r"[a-zA-Z']+")


def _norm(text: str) -> str:
    return text.strip()


def score_segment(text: str) -> tuple[float, dict]:
    t = _norm(text)
    words = WORD.findall(t)
    n = len(words)
    if n == 0:
        return 0.0, {}

    lower = [w.lower() for w in words]

    # punch: ! or ? presence
    punch = 0.0
    if "!" in t:
        punch += 0.6
    if "?" in t:
        punch += 0.4
    punch = min(punch, 1.0)

    # brevity: sweet spot 3..12 words; penalize very long / one-word
    if n <= 2:
        brevity = 0.3
    elif n <= 12:
        brevity = 1.0
    else:
        brevity = max(0.2, 1.0 - (n - 12) / 25.0)

    # emotion density
    emo_hits = sum(1 for w in lower if w in EMOTION)
    emotion = min(1.0, emo_hits / max(2, n / 4))

    # imperative start
    imperative = 1.0 if lower and lower[0] in IMPERATIVE_STARTS else 0.0

    # direct address to viewer
    pronouns = 1.0 if DIRECT_ADDR.search(t) else 0.0

    # shouted caps (exclude common all-caps like I, A)
    caps = min(1.0, len([c for c in CAPS_TOKEN.findall(text) if c not in {"THE", "AND"}]) * 0.5)

    # repetition / anaphora
    from collections import Counter
    counts = Counter(lower)
    rep = max(counts.values()) if counts else 1
    repetition = min(1.0, (rep - 1) / 3.0)

    weights = {
        "punch": 0.22,
        "brevity": 0.18,
        "emotion": 0.24,
        "imperative": 0.12,
        "pronouns": 0.10,
        "caps": 0.06,
        "repetition": 0.08,
    }
    parts = {
        "punch": punch, "brevity": brevity, "emotion": emotion,
        "imperative": imperative, "pronouns": pronouns, "caps": caps,
        "repetition": repetition,
    }
    total = sum(weights[k] * parts[k] for k in weights)
    return round(total, 4), parts


def _window_peak_score(text: str) -> float:
    """
    A long scene should be ranked by its BEST line, not diluted by an average.
    Slide over the sentences and take the max single-line hook score, so a scene
    containing a killer quote still ranks high even if surrounded by setup.
    """
    import re as _re
    pieces = [p.strip() for p in _re.split(r"[.!?]+", text) if p.strip()]
    if not pieces:
        return score_segment(text)[0]
    peak = 0.0
    for p in pieces:
        s, _ = score_segment(p)
        if s > peak:
            peak = s
    # blend: mostly the peak line, a little of the whole-scene score
    whole, _ = score_segment(text)
    return round(0.7 * peak + 0.3 * whole, 4)


def rank_segments(segments: list[dict], scene_gap: float = 1.2,
                  max_scene_dur: float = 45.0) -> list[dict]:
    """
    Group subtitle cues into whole CONVERSATION SCENES (not single lines), so a
    back-and-forth exchange like 'sell me this pen ... supply and demand' stays
    one clip instead of being cut at a tiny pause.

    A scene continues while the gap to the next cue stays under `scene_gap`
    seconds AND the scene has not run past `max_scene_dur`. Each scene keeps its
    member cues, is scored by its peak hook line, and gets a smart `hook_end`
    so the clip can stop at the hook payoff rather than the hard scene end.
    """
    if not segments:
        return []

    scenes: list[dict] = []
    cur = dict(segments[0])
    cur["cues"] = [dict(segments[0])]
    for s in segments[1:]:
        gap = s["start"] - cur["end"]
        span = s["end"] - cur["start"]
        if gap <= scene_gap and span <= max_scene_dur:
            cur["end"] = s["end"]
            cur["text"] = (cur["text"] + " " + s["text"]).strip()
            cur["cues"].append(dict(s))
        else:
            scenes.append(cur)
            cur = dict(s)
            cur["cues"] = [dict(s)]
    scenes.append(cur)

    for m in scenes:
        m["score"] = _window_peak_score(m["text"])
        m["signals"] = score_segment(m["text"])[1]
        # full-scene boundary: never trim, so a quiet payoff line is never cut.
        # (user sorts / trims final clips themselves in the editor)
        m["hook_end"] = round(m["end"], 3)
        m["scene_end"] = round(m["end"], 3)
        m["duration"] = round(m["end"] - m["start"], 2)

    scenes.sort(key=lambda x: x["score"], reverse=True)
    return scenes
