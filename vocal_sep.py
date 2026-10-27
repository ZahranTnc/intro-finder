"""Separate vocals from a video/audio file using Demucs (CPU, small segment).
Outputs <stem>_vocals.wav next to the input (or in --out dir).
Keeps RAM low via --segment and two-stem vocal mode.
Usage: python vocal_sep.py <input> [--out DIR] [--segment 7] [--model htdemucs]
"""
import os, sys, argparse, subprocess, tempfile, shutil, time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None, help="output dir (default: beside input)")
    ap.add_argument("--segment", type=int, default=7, help="segment seconds (lower=less RAM)")
    ap.add_argument("--model", default="htdemucs")
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        print("ERR: input not found:", inp); sys.exit(2)
    stem = os.path.splitext(os.path.basename(inp))[0]
    out_dir = os.path.abspath(args.out) if args.out else os.path.dirname(inp)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, stem + "_vocals.wav")

    scratch = os.environ.get("KIROCREW_SCRATCH") or tempfile.gettempdir()
    work = tempfile.mkdtemp(prefix="demucs_", dir=scratch)
    t0 = time.time()
    print(f"[demucs] model={args.model} segment={args.segment}s -> {final}", flush=True)
    # two-stem vocals: produces vocals.wav + no_vocals.wav; we keep vocals
    cmd = [sys.executable, "-m", "demucs",
           "-n", args.model,
           "--two-stems", "vocals",
           "--segment", str(args.segment),
           "-d", "cpu",
           "-o", work,
           inp]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("ERR: demucs failed rc", r.returncode); sys.exit(r.returncode)
    # demucs writes work/<model>/<stem>/vocals.wav
    voc = os.path.join(work, args.model, stem, "vocals.wav")
    if not os.path.isfile(voc):
        # fallback: find any vocals.wav
        for root, _, files in os.walk(work):
            if "vocals.wav" in files:
                voc = os.path.join(root, "vocals.wav"); break
    if not os.path.isfile(voc):
        print("ERR: vocals.wav not produced"); sys.exit(3)
    shutil.move(voc, final)
    shutil.rmtree(work, ignore_errors=True)
    print(f"[demucs] DONE in {time.time()-t0:.1f}s -> {final}", flush=True)

if __name__ == "__main__":
    main()
