"""
cut.py — cut a time range out of a video into its own mp4 with ffmpeg.

Uses a small pre-roll/post-roll pad so a line doesn't start/end abruptly, and
re-encodes (not -c copy) so cuts land on exact frames rather than the nearest
keyframe — important for short dialogue clips.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def cut_clip(video: str, start: float, end: float, out_path: str,
             pad: float = 0.35, reencode: bool = True,
             audio_source: str | None = None) -> str:
    """Cut [start,end] from video. If audio_source is given (e.g. a Demucs
    vocals.wav of the same timeline), the CLIP VIDEO comes from `video` but the
    CLIP AUDIO is taken from `audio_source` at the same timestamps — producing a
    music-free (vocal-only) clip the user can score manually. Omit for original mix.
    """
    start = max(0.0, start - pad)
    dur = (end + pad) - start
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if audio_source:
        # video from `video`, audio from `audio_source`, both seeked to `start`
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-i", video,
            "-ss", f"{start:.3f}", "-i", audio_source,
            "-t", f"{dur:.3f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path

    if reencode:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video,
            "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video,
            "-t", f"{dur:.3f}", "-c", "copy", out_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
