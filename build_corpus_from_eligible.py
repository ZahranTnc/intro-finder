"""
build_corpus_from_eligible.py — build movie conversation corpus from eligible folders.

Scans D:/ai/clipcafe/out/oceans12_review/Eligible and oceans13_review/eligible,
transcribes each clip, builds corpus/movie_intros.json.

Usage:
    python build_corpus_from_eligible.py
"""
import json, re, subprocess
from pathlib import Path
from faster_whisper import WhisperModel

def get_duration(mp4_path):
    """Get video duration with ffprobe."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)]
        dur = subprocess.check_output(cmd, text=True).strip()
        return float(dur)
    except Exception:
        return 0.0

def transcribe_clip(mp4_path, model):
    """Transcribe with silence detection (>3s = end)."""
    segs, _ = model.transcribe(str(mp4_path), vad_filter=True, beam_size=1, language="en")
    text_parts = []
    last_end = 0.0
    
    for s in segs:
        gap = s.start - last_end
        if gap > 3.0 and text_parts:
            break
        text_parts.append(s.text.strip())
        last_end = s.end
    
    return " ".join(text_parts).strip()

def build_corpus():
    eligible_dirs = [
        Path("D:/ai/clipcafe/out/oceans12_review/Eligible"),
        Path("D:/ai/clipcafe/out/oceans13_review/eligible")
    ]
    
    # ensure dirs exist
    for d in eligible_dirs:
        if not d.exists():
            d.mkdir(parents=True)
            print(f"Created {d} — move eligible clips there")
    
    all_clips = []
    for d in eligible_dirs:
        all_clips.extend(d.glob("*.mp4"))
    
    if not all_clips:
        print(f"\nNo clips found in eligible folders.")
        print(f"Move good clips to:")
        for d in eligible_dirs:
            print(f"  {d}")
        return
    
    print(f"\nFound {len(all_clips)} eligible clips")
    print("Transcribing...")
    
    model = WhisperModel("small", device="cpu", compute_type="int8")
    intros = []
    
    for i, clip_path in enumerate(all_clips, 1):
        movie = "Ocean's Twelve" if "oceans12" in str(clip_path) else "Ocean's Thirteen"
        print(f"  [{i}/{len(all_clips)}] {clip_path.name[:50]}")
        
        text = transcribe_clip(clip_path, model)
        duration = get_duration(clip_path)
        
        intros.append({
            "id": clip_path.stem,
            "movie": movie,
            "text": text,
            "duration": duration,
            "views": 0,  # placeholder
            "view_rate": 1.0,  # all equal weight
            "account": "movie_conversations",
        })
    
    # write corpus
    corpus_file = Path("corpus/movie_intros.json")
    corpus_file.parent.mkdir(exist_ok=True)
    corpus_file.write_text(json.dumps(intros, ensure_ascii=False, indent=1), encoding="utf-8")
    
    durs = [i["duration"] for i in intros if i["duration"] > 0]
    if durs:
        durs.sort()
        p10 = durs[len(durs)//10]
        p90 = durs[int(len(durs)*0.9)]
        print(f"\nDone: {len(intros)} movie conversations -> {corpus_file}")
        print(f"Duration range: {p10:.1f}s - {p90:.1f}s (p10-p90)")
    else:
        print(f"\nDone: {len(intros)} movie conversations -> {corpus_file}")

if __name__ == "__main__":
    build_corpus()
