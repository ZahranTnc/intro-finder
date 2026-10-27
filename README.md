# intro-finder

Python toolkit for turning long videos into short vertical clips.

## What it does

Finds hook moments in long videos (movies, vlogs, podcasts) and cuts them into short clips ready for TikTok/Reels.

**Pipeline:**
1. Extract scenes with timestamps (Whisper transcription)
2. Rank scenes by hook quality (heuristic, embedding, or LLM)
3. Cut top clips with auto-snap to sentence boundaries
4. Optional: vocal separation for music-heavy content

## Modules

- `intro-finder.py` — main CLI (export, cut)
- `subs.py` — Whisper transcription + vocal separation (Demucs)
- `find_moments.py`, `moments.py` — scene segmentation
- `hooks.py` — heuristic ranker (keyword + energy)
- `intro.py` — embedding similarity ranker
- `rank_merge.py` — batch LLM ranking workflow
- `cut.py` — clip cutting with cue auto-snap
- `vocal_sep.py` — Demucs wrapper
- `tiktok.py`, `batch_tren.py` — TikTok export utilities

## Quick start

```bash
# Export scenes from a video
python intro-finder.py export --video movie.mp4 --srt movie.srt

# Rank scenes (heuristic)
python intro-finder.py clips --video movie.mp4 --srt movie.srt --top 20

# Cut ranked clips
python intro-finder.py cut-reranked --video movie.mp4 --reranked out/reranked.json
```

## Dependencies

- Python 3.10+
- faster-whisper (transcription)
- demucs (vocal separation, optional)
- ffmpeg, ffprobe
- sentence-transformers (for embedding ranker)

## Output

- `out/scenes.json` — extracted scenes with timestamps
- `out/reranked.json` — ranked scene IDs
- `out/clips/` — cut MP4 clips

## Notes

Built for local/free tooling (faster-whisper, Demucs). No paid APIs.

Vocal separation (`CLIPCAFE_VOCAL_AUTO=1`) auto-triggers when music buries dialogue (logprob < -0.85 or no_speech > 0.45). Clips are cut from the original full-audio video by default, not the vocal-only track.

Cue auto-snap extends clips to the end of the spoken sentence + 0.8s breath, preventing mid-sentence cuts.
