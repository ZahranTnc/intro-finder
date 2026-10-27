"""
intro.py — score movie scenes by similarity to successful TikTok intros + Laya hook labels.

Scorer for intro_finder.py that learns from corpus/intros.json instead of hardcoded heuristics.

Three signals:
  1. similarity: embedding distance to top-viewed intros (sentence-transformers)
  2. laya_hook: hook-type probability (threat/confession/reveal/funny/question)
  3. heuristic: existing hooks.py score (punch/brevity/emotion)

Final score = 0.5·similarity + 0.3·laya_hook + 0.2·heuristic

Also learns dynamic intro length from the corpus p10-p90 range.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Optional

_CORPUS_FILE = Path(__file__).parent / "corpus" / "movie_intros.json"
_EMBEDDER = None
_LAYA_ROUTER = None
_TOP_INTROS = None
_DUR_RANGE = None


def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _load_laya():
    global _LAYA_ROUTER
    if _LAYA_ROUTER is None:
        from laya import Router
        _LAYA_ROUTER = Router(preload=False)
    return _LAYA_ROUTER


def _load_corpus():
    global _TOP_INTROS, _DUR_RANGE
    # Force reload to pick up changes
    
    if not _CORPUS_FILE.exists():
        raise FileNotFoundError(f"corpus not built: {_CORPUS_FILE}\nRun: python build_corpus_manual.py")
    
    intros = json.loads(_CORPUS_FILE.read_text(encoding="utf-8"))
    if not intros:
        raise ValueError("corpus is empty")
    
    # manual tier weighting: all movie conversations equal weight
    for intro in intros:
        intro["tier_weight"] = 1.0
        if intro.get("views", 0) == 0:
            intro["view_rate"] = 1.0
    
    # top 50% by view_rate (or all equal if no views)
    intros.sort(key=lambda x: x.get("view_rate", x.get("tier_weight", 1.0)), reverse=True)
    cutoff = max(1, len(intros) // 2)
    _TOP_INTROS = [i for i in intros[:cutoff] if i.get("text")]
    
    # duration range p10-p90
    durs = sorted(i["duration"] for i in intros if i.get("duration") and i["duration"] > 0)
    if durs:
        p10 = max(5.0, durs[len(durs) // 10])  # floor at 5s
        p90 = min(60.0, durs[int(len(durs) * 0.9)])  # cap at 60s
    else:
        p10, p90 = 5.0, 60.0  # fallback wider range
    _DUR_RANGE = (p10, p90)


def score_intro_similarity(text: str) -> float:
    """Cosine similarity to top-viewed intros, 0-1."""
    _load_corpus()
    if not _TOP_INTROS:
        return 0.5
    
    emb = _load_embedder()
    from sklearn.metrics.pairwise import cosine_similarity
    
    scene_vec = emb.encode([text])
    top_vecs = emb.encode([i["text"] for i in _TOP_INTROS])
    sims = cosine_similarity(scene_vec, top_vecs)[0]
    return float(max(sims))


def score_laya_hook(text: str) -> float:
    """Laya hook-type probability, 0-1."""
    router = _load_laya()
    questions = {
        "hook_type": {
            "type": "choice",
            "instructions": "Which hook type best describes this line?",
            "criteria": {
                "threat": "danger, warning, escalation",
                "confession": "admission, secret revealed",
                "reveal": "plot twist, surprise, unveiling truth",
                "funny": "joke, comedic line, sarcasm",
                "question": "rhetorical or direct question to viewer",
                "none": "no strong hook"
            }
        }
    }
    result = router.predict(text, questions)
    choice = result["answers"]["hook_type"]["choice"]
    conf = result["answers"]["hook_type"]["confidence"]
    # treat "none" as 0, others as conf
    return 0.0 if choice == "none" else conf


def score_segment_intro(seg: dict) -> tuple[float, dict]:
    """
    Score a clipcafe segment for intro quality.
    Returns (score, parts) where parts = {similarity, laya_hook, heuristic, length_bonus}.
    """
    from hooks import score_segment as heuristic_score
    
    text = seg["text"]
    duration = seg.get("duration", 0)
    
    sim = score_intro_similarity(text)
    laya = score_laya_hook(text)
    heur, _ = heuristic_score(text)
    
    # Length bonus: STRONGLY prefer 15-60s conversations
    if duration >= 20 and duration <= 60:
        length_bonus = 2.0  # strong boost for full conversations
    elif duration >= 15 and duration < 20:
        length_bonus = 1.6  # good boost
    elif duration >= 10 and duration < 15:
        length_bonus = 1.2  # medium boost
    elif duration >= 5 and duration < 10:
        length_bonus = 0.5  # strong penalty
    elif duration < 5:
        length_bonus = 0.3  # very strong penalty for one-liners
    else:  # >60s
        length_bonus = 1.0  # neutral
    
    base_score = 0.5 * sim + 0.3 * laya + 0.2 * heur
    final_score = base_score * length_bonus
    
    parts = {"similarity": sim, "laya_hook": laya, "heuristic": heur, "length_bonus": length_bonus}
    return round(final_score, 4), parts


def get_intro_duration_range() -> tuple[float, float]:
    """Return (min, max) duration in seconds learned from corpus."""
    # Always recompute to pick up changes
    intros = json.loads(_CORPUS_FILE.read_text(encoding="utf-8"))
    durs = sorted(i["duration"] for i in intros if i.get("duration") and i["duration"] > 0)
    if durs:
        p10 = max(5.0, durs[len(durs) // 10])
        p90 = min(60.0, durs[int(len(durs) * 0.9)])
    else:
        p10, p90 = 5.0, 60.0
    return (p10, p90)


def rank_segments_intro(segments: list[dict], prefilter_top: int = None) -> list[dict]:
    """
    Re-rank segments using intro scorer.
    No pre-filtering — scores ALL segments with similarity+Laya+heuristic+length_bonus.
    Length bonus prioritizes 15-60s conversations over short one-liners.
    """
    from hooks import score_segment as heuristic_score
    
    _load_corpus()
    
    # Score all segments
    for seg in segments:
        if "duration" not in seg:
            seg["duration"] = round(seg["end"] - seg["start"], 2)
        score, parts = score_segment_intro(seg)
        seg["score"] = score
        seg["signals"] = parts
        seg["hook_end"] = round(seg["end"], 3)
        seg["scene_end"] = round(seg["end"], 3)
    
    segments.sort(key=lambda x: x["score"], reverse=True)
    return segments
