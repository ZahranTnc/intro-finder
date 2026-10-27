from faster_whisper import WhisperModel
import os
voc = r"D:\ai\intro-finder\_vocal_test\new1-47_vocals.wav"
print("size MB:", round(os.path.getsize(voc)/1e6,2))
m = WhisperModel("small", device="cpu", compute_type="int8")
segs, info = m.transcribe(voc, language="en", beam_size=5, vad_filter=True,
                          condition_on_previous_text=False)
for s in segs:
    nsp = getattr(s,"no_speech_prob",0.0) or 0.0
    alp = getattr(s,"avg_logprob",0.0) or 0.0
    print("  [%6.2f-%6.2f] nsp=%.2f alp=%.2f | %s" % (s.start,s.end,nsp,alp,s.text.strip()))
print("DONE")
