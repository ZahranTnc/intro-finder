import os, sys, json, subprocess
from faster_whisper import WhisperModel

clips = ["new1-3.mp4","new1-41.mp4","new1-47.mp4"]
base = r"D:\caaapct\Edits\Tren\new1"
outdir = r"D:\ai\clipcafe\_example_probe"
os.makedirs(outdir, exist_ok=True)

print("[load] small model", flush=True)
model = WhisperModel("small", device="cpu", compute_type="int8")

for c in clips:
    path = os.path.join(base, c)
    print("\n==== %s ====" % c, flush=True)
    segs, info = model.transcribe(path, language="en", beam_size=5,
                                  vad_filter=True,
                                  condition_on_previous_text=False)
    for s in segs:
        nsp = getattr(s, "no_speech_prob", 0.0) or 0.0
        alp = getattr(s, "avg_logprob", 0.0) or 0.0
        dropped = (nsp > 0.6 or alp < -1.0)
        flag = "  <-- DROPPED by my filter" if dropped else ""
        print("  [%6.2f-%6.2f] nsp=%.2f alp=%.2f | %s%s" % (
            s.start, s.end, nsp, alp, s.text.strip(), flag), flush=True)
    # mid frame
    try:
        dur = float(subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration",
             "-of","default=noprint_wrappers=1:nokey=1", path]).decode().strip())
        mid = dur/2
        frame = os.path.join(outdir, c.replace(".mp4",".jpg"))
        subprocess.run(["ffmpeg","-y","-ss",str(mid),"-i",path,"-frames:v","1",
                        "-q:v","2", frame],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("  frame err:", e, flush=True)

print("\nDONE", flush=True)
