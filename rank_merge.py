"""
rank_merge.py — deterministic backbone for the clipcafe-rank skill.

The SKILL.md rubric is the JUDGMENT; this file is everything deterministic around
it so the skill is portable (any model does step 3, this script does the rest):

  --split   scenes.json  -> batch_000.json, batch_001.json, ...   (30-40 scenes)
  --prompt  batch_NNN.json -> prints the exact LLM prompt (rubric + batch)
  --merge   batch_*.ranked.json -> validate, merge, sort -> reranked.json
  --eval    reranked.json vs a review folder's eligible labels -> recall@20

The LLM step is NOT done here. The agent reads SKILL.md + a batch and writes
batch_NNN.ranked.json (the strict JSON array from the contract). Then --merge
validates and assembles. This keeps the model swappable (Kiro / free tier).

Usage
-----
  python rank_merge.py --split  --scenes out/scenes.json --batch-dir out/batches
  python rank_merge.py --prompt --batch out/batches/batch_000.json
  python rank_merge.py --merge  --batch-dir out/batches --scenes out/scenes.json \
                       --out out/reranked.json
  python rank_merge.py --eval   --reranked out/reranked.json \
                       --review out/oceans12_review --top 20
"""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / ".agents" / "skills" / "clipcafe-rank"
# Mode -> rubric file. "film" is the default (SKILL.md); each new genre adds one
# rubric file here, the deterministic backbone (split/merge/cut/snap) is shared.
RUBRIC_BY_MODE = {
    "film": SKILL_DIR / "SKILL.md",
    "gym":  SKILL_DIR / "SKILL-gym.md",
}
BATCH_SIZE = 35          # 30-40 window; 35 fits free-tier context with the rubric
TIER_RANK = {"strong": 0, "mid": 1, "skip": 2}


def _rubric_path(mode: str) -> Path:
    path = RUBRIC_BY_MODE.get(mode)
    if path is None:
        raise SystemExit(f"unknown --mode {mode!r}; choices: {', '.join(RUBRIC_BY_MODE)}")
    return path


# ---------------------------------------------------------------- split

def cmd_split(a):
    scenes = json.loads(Path(a.scenes).read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes") or scenes.get("clips") or []
    bdir = Path(a.batch_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    # clear old batches so a re-split never merges stale ones
    for old in bdir.glob("batch_*.json"):
        old.unlink()

    n = 0
    for i in range(0, len(scenes), a.size):
        chunk = scenes[i:i + a.size]
        # only the fields the LLM needs — keep the prompt lean
        lean = [{"id": s["id"], "text": s["text"],
                 "duration": round(s.get("duration", s.get("end", 0) - s.get("start", 0)), 1)}
                for s in chunk]
        (bdir / f"batch_{n:03d}.json").write_text(
            json.dumps(lean, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(f"split {len(scenes)} scenes -> {n} batches of <= {a.size} in {bdir}")


# ---------------------------------------------------------------- prompt

def _rubric_text(mode: str = "film") -> str:
    path = _rubric_path(mode)
    if not path.exists():
        raise SystemExit(f"rubric for mode {mode!r} not found at {path}")
    return path.read_text(encoding="utf-8")


def build_prompt(batch: list[dict], mode: str = "film") -> str:
    rubric = _rubric_text(mode)
    scenes_json = json.dumps(batch, ensure_ascii=False, indent=1)
    return (
        f"{rubric}\n\n"
        "=== TASK ===\n"
        "Apply the rubric above to EVERY scene in the batch below. Return ONLY the "
        "strict JSON array from the output contract — no prose, no markdown fence. "
        "Every id must appear exactly once.\n\n"
        f"=== BATCH ({len(batch)} scenes) ===\n{scenes_json}\n"
    )


def cmd_prompt(a):
    batch = json.loads(Path(a.batch).read_text(encoding="utf-8"))
    print(build_prompt(batch, getattr(a, "mode", "film")))


# ---------------------------------------------------------------- validate + merge

def _extract_json_array(raw: str):
    """Pull a JSON array out of a model response that may wrap it in prose/fences."""
    raw = raw.strip()
    # strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if m:
        raw = m.group(1).strip()
    # find the outermost [ ... ]
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in response")
    return json.loads(raw[start:end + 1])


def validate_batch(ranked: list[dict], batch_ids: set[str]) -> list[str]:
    """Return a list of problems; empty list = valid. Caller retries on problems."""
    problems = []
    seen = set()
    for r in ranked:
        rid = r.get("id")
        if rid not in batch_ids:
            problems.append(f"unknown id: {rid!r}")
            continue
        if rid in seen:
            problems.append(f"duplicate id: {rid!r}")
        seen.add(rid)
        if r.get("tier") not in TIER_RANK:
            problems.append(f"bad tier for {rid!r}: {r.get('tier')!r}")
    missing = batch_ids - seen
    if missing:
        problems.append(f"missing {len(missing)} ids, e.g. {list(missing)[:3]}")
    return problems


def cmd_merge(a):
    scenes = json.loads(Path(a.scenes).read_text(encoding="utf-8"))
    if isinstance(scenes, dict):
        scenes = scenes.get("scenes") or scenes.get("clips") or []
    scene_by_id = {s["id"]: s for s in scenes}

    bdir = Path(a.batch_dir)
    ranked_files = sorted(bdir.glob("batch_*.ranked.json"))
    if not ranked_files:
        raise SystemExit(
            f"no batch_*.ranked.json in {bdir}. The agent must write each batch's "
            "rubric output there first (see SKILL.md step 3).")

    merged = []
    for rf in ranked_files:
        batch_file = bdir / rf.name.replace(".ranked.json", ".json")
        batch_ids = {s["id"] for s in json.loads(batch_file.read_text(encoding="utf-8"))}
        try:
            ranked = _extract_json_array(rf.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError) as e:
            raise SystemExit(f"{rf.name}: unparseable JSON ({e}). Re-run the agent on "
                             f"{batch_file.name} and overwrite {rf.name}.")
        problems = validate_batch(ranked, batch_ids)
        if problems:
            raise SystemExit(f"{rf.name}: validation failed -> {problems}. Re-run the "
                             f"agent on {batch_file.name} and overwrite {rf.name}.")
        for r in ranked:
            r["_batch"] = rf.name
        merged.extend(ranked)

    # drop skips, sort: tier (strong<mid), then in-batch rank, then duration desc
    kept = [r for r in merged if r.get("tier") != "skip"]
    for r in kept:
        sc = scene_by_id.get(r["id"], {})
        r["duration"] = round(sc.get("end", 0) - sc.get("start", 0), 2)
    kept.sort(key=lambda r: (TIER_RANK.get(r.get("tier"), 9),
                             r.get("rank", 999),
                             -r.get("duration", 0)))

    # assign global rank + attach hook_start, pass start/end through for cutting
    out = []
    for i, r in enumerate(kept, 1):
        sc = scene_by_id.get(r["id"], {})
        out.append({
            "id": r["id"],
            "rank": i,
            "tier": r["tier"],
            "hook_start": round(r.get("hook_start", 0.0), 2),
            "hook_len": round(r.get("hook_len", 0.0), 2),
            "reason": r.get("reason", ""),
            "clip_score": {"strong": 9, "mid": 6}.get(r["tier"], 0),
            "start": sc.get("start"),
            "end": sc.get("end"),
        })
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    strong = sum(1 for r in out if r["tier"] == "strong")
    skipped = len(merged) - len(kept)
    print(f"merged {len(ranked_files)} batches, {len(merged)} scenes: "
          f"{strong} strong, {len(out) - strong} mid, {skipped} skipped")
    print(f"-> {a.out}  (feed to: intro_finder.py cut-reranked --reranked {a.out})")


# ---------------------------------------------------------------- eval

def _eligible_ids_from_review(review_dir: Path) -> tuple[set[str], set[str]]:
    """
    Infer labels from the review folder: files inside Eligible/ are positives;
    everything in the top folder is a candidate. Match by text-slug prefix so it
    lines up with scene ids (both come from the same _slug()).
    Returns (eligible_slugs, all_slugs).
    """
    def slugset(d: Path):
        out = set()
        for f in d.glob("*.mp4"):
            stem = f.stem
            # strip leading rank prefix "00_" and any " = note" suffix
            stem = re.sub(r"^\d+_", "", stem)
            stem = re.split(r"\s*=", stem)[0].strip()
            out.add(stem)
        return out

    elig_dir = next((p for p in review_dir.iterdir()
                     if p.is_dir() and p.name.lower() == "eligible"), None)
    eligible = slugset(elig_dir) if elig_dir else set()
    allslugs = slugset(review_dir) | eligible
    return eligible, allslugs


def _slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "clip"


def cmd_eval(a):
    reranked = json.loads(Path(a.reranked).read_text(encoding="utf-8"))
    eligible, _ = _eligible_ids_from_review(Path(a.review))
    if not eligible:
        raise SystemExit(f"no Eligible/ folder with mp4s under {a.review}")

    # reranked ids are scene ids; map each to its text slug via scenes if present,
    # else assume the id is already slug-like. We compare slug prefixes.
    top = reranked[:a.top]
    top_slugs = set()
    for r in top:
        rid = str(r["id"])
        rid = re.sub(r"^\d+_", "", rid)
        top_slugs.add(rid[:40])

    hits = sum(1 for e in eligible if any(e[:30] in t or t[:30] in e for t in top_slugs))
    total = len(eligible)
    pct = 100.0 * hits / total if total else 0.0
    print(f"eval @ top-{a.top}:  {hits}/{total} eligible recovered  ({pct:.0f}%)")
    print(f"  eligible clips : {total}")
    print(f"  in top-{a.top}      : {hits}")
    if pct < 60:
        print("  VERDICT: rubric likely wrong — a strong model should recover most. "
              "Fix the rubric or swap a few-shot example (do NOT grow examples).")
    else:
        print("  VERDICT: rubric is tracking the user's taste.")


# ---------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description="deterministic backbone for clipcafe-rank")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", action="store_true")
    g.add_argument("--prompt", action="store_true")
    g.add_argument("--merge", action="store_true")
    g.add_argument("--eval", action="store_true")

    p.add_argument("--scenes", default="D:/ai/clipcafe/out/scenes.json")
    p.add_argument("--batch-dir", default="D:/ai/clipcafe/out/batches")
    p.add_argument("--batch", help="single batch file for --prompt")
    p.add_argument("--out", default="D:/ai/clipcafe/out/reranked.json")
    p.add_argument("--reranked", default="D:/ai/clipcafe/out/reranked.json")
    p.add_argument("--review", help="review folder with Eligible/ for --eval")
    p.add_argument("--size", type=int, default=BATCH_SIZE)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--mode", default="film", choices=list(RUBRIC_BY_MODE),
                   help="rubric mode: film (default) or gym")
    a = p.parse_args()

    if a.split:
        cmd_split(a)
    elif a.prompt:
        if not a.batch:
            raise SystemExit("--prompt needs --batch <batch_000.json>")
        cmd_prompt(a)
    elif a.merge:
        cmd_merge(a)
    elif a.eval:
        if not a.review:
            raise SystemExit("--eval needs --review <folder with Eligible/>")
        cmd_eval(a)


if __name__ == "__main__":
    main()
