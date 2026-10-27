"""
build_corpus_manual.py — build corpus from manually downloaded videos.

Usage:
    python build_corpus_manual.py
    python build_corpus_manual.py --views-file views.json

Scans corpus/<acc>/*.mp4, transcribes with silence detection (>3s = end of dialog),
and builds corpus/intros.json. View counts come from views.json or manual input.

views.json format:
{
  "grmz.edits": {
    "1790892288": 150000,
    "1790892302": 200000
  }
}
"""
import argparse, json, re
from pathlib import Path
from collections import defaultdict

def transcribe_with_silence_detection(mp4_path: str, silence_threshold: float = 3.0) -> str:
    """Transcribe and stop at silence > threshold seconds."""
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(mp4_path, vad_filter=True, beam_size=1, language="en")
    
    text_parts = []
    last_end = 0.0
    
    for s in segs:
        gap = s.start - last_end
        if gap > silence_threshold and text_parts:
            # silence detected, stop here
            break
        text_parts.append(s.text.strip())
        last_end = s.end
    
    return " ".join(text_parts).strip()


def extract_video_id(filename: str) -> str:
    """Extract video ID from musicaldown format: <acc>[HD]_musicaldown.com_<id>.mp4"""
    m = re.search(r'_(\d+)\.mp4$', filename)
    return m.group(1) if m else filename.replace('.mp4', '')


def load_views(views_file: str | None) -> dict:
    """Load view counts from JSON file."""
    if not views_file or not Path(views_file).exists():
        return {}
    return json.loads(Path(views_file).read_text(encoding="utf-8"))


def build_corpus_manual(views_file: str | None):
    corpus_root = Path("corpus")
    if not corpus_root.exists():
        corpus_root.mkdir()
        print("Created corpus/ directory. Put videos in corpus/<account>/")
        return
    
    views_db = load_views(views_file)
    all_intros = []
    acc_views = defaultdict(list)
    
    accounts = [d for d in corpus_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not accounts:
        print("No account folders found in corpus/")
        return
    
    for acc_dir in accounts:
        handle = acc_dir.name
        videos = list(acc_dir.glob("*.mp4"))
        
        if not videos:
            print(f"\n=== {handle}: no mp4 files ===")
            continue
        
        print(f"\n=== {handle}: {len(videos)} videos ===")
        
        for i, vid_path in enumerate(videos, 1):
            vid_id = extract_video_id(vid_path.name)
            meta_file = acc_dir / f"{vid_id}.json"
            
            if meta_file.exists():
                print(f"  [{i}/{len(videos)}] {vid_id} (cached)")
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                all_intros.append({**meta, "account": handle})
                acc_views[handle].append(meta.get("views", 0))
                continue
            
            print(f"  [{i}/{len(videos)}] {vid_id} transcribing...")
            text = transcribe_with_silence_detection(str(vid_path), silence_threshold=3.0)
            
            # view count: from views.json or default to 0
            views = views_db.get(handle, {}).get(vid_id, 0)
            
            duration = 0.0  # would need ffprobe, skip for now
            
            meta = {
                "id": vid_id,
                "url": f"https://www.tiktok.com/@{handle}/video/{vid_id}",
                "title": text[:80],
                "duration": duration,
                "views": views,
                "likes": 0,
                "text": text,
            }
            
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            all_intros.append({**meta, "account": handle})
            acc_views[handle].append(views)
    
    # compute view_rate per intro
    acc_medians = {}
    for acc, vlist in acc_views.items():
        vlist = [v for v in vlist if v > 0]
        acc_medians[acc] = sorted(vlist)[len(vlist)//2] if vlist else 1
    
    for intro in all_intros:
        med = acc_medians.get(intro["account"], 1)
        intro["view_rate"] = intro["views"] / max(1, med)
    
    # write intros.json
    intros_file = corpus_root / "intros.json"
    intros_file.write_text(json.dumps(all_intros, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDone: {len(all_intros)} intros -> {intros_file}")
    
    # get duration range for the scorer
    import subprocess
    for intro in all_intros:
        if intro["duration"] == 0.0:
            vid_path = corpus_root / intro["account"] / f"{intro['id']}.mp4"
            if not vid_path.exists():
                vid_path = list((corpus_root / intro["account"]).glob(f"*{intro['id']}.mp4"))
                vid_path = vid_path[0] if vid_path else None
            if vid_path:
                try:
                    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                           "-of", "default=noprint_wrappers=1:nokey=1", str(vid_path)]
                    dur = subprocess.check_output(cmd, text=True).strip()
                    intro["duration"] = float(dur)
                except Exception:
                    intro["duration"] = 10.0  # fallback
    
    intros_file.write_text(json.dumps(all_intros, ensure_ascii=False, indent=1), encoding="utf-8")
    
    durs = [i["duration"] for i in all_intros if i["duration"] > 0]
    if durs:
        durs.sort()
        p10 = durs[len(durs)//10]
        p90 = durs[int(len(durs)*0.9)]
        print(f"Duration range: {p10:.1f}s - {p90:.1f}s (p10-p90)")


def main():
    p = argparse.ArgumentParser(description="build corpus from manual downloads")
    p.add_argument("--views-file", default=None, help="JSON file with view counts")
    args = p.parse_args()
    build_corpus_manual(args.views_file)


if __name__ == "__main__":
    main()
