"""Voice dataset recorder for fine-tuning Orpheus on YOUR voice.

Shows you a sentence, records you reading it (in your browser, your mic), decodes to
clean 24 kHz mono WAV, and saves it with the transcript. Read ~all of them and you have
a training set. Record in a quiet room, consistent mic distance, and — importantly —
speak in the exact calm, one-to-one tone you want the agent to have.

Run:  ./run_record.sh    then open  http://localhost:7861
Dataset lands in:  voice_dataset/clips/NNN.wav  +  voice_dataset/metadata.csv
"""
from __future__ import annotations

import csv
import io
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import av.audio.resampler
import numpy as np
import soundfile as sf

PORT = int(os.environ.get("REC_PORT", "7861"))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_dataset")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

# ~100 sentences: phonetically varied + conversational + questions + numbers + sales tone.
SENTENCES = [
    "Hey, thanks for taking my call, I'll keep it quick.",
    "So, how's your day going so far?",
    "Yeah, I totally get that, no worries at all.",
    "Honestly, I was just curious how things are going on your end.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on sunny days.",
    "I think we can probably help you save a bit of money here.",
    "Let me pull that up real quick, give me one second.",
    "That makes sense, and I appreciate you being straight with me.",
    "Would next Tuesday or Thursday work better for a quick chat?",
    "We usually see people cut their bills by around thirty percent.",
    "It's about two hundred and fifty dollars a month, give or take.",
    "No pressure at all, I just wanted to lay out the options.",
    "Right, so here's the thing, and I'll be honest with you.",
    "Can you hear me okay? The line sounds a little quiet.",
    "Perfect, that works great on my end too.",
    "I hear you, that's a really common concern actually.",
    "Let's say we start small and see how it feels.",
    "You know what, that's a fair point, I hadn't thought of that.",
    "Alright, so what matters most to you in a setup like this?",
    "The weather's been kind of all over the place lately.",
    "A little rain never hurt anybody, right?",
    "I grew up in a small town, so this is pretty familiar.",
    "We've got about seven or eight options, but only two really fit.",
    "Give me a ballpark, what are you paying right now?",
    "Okay, cool, that's actually lower than I expected.",
    "Between you and me, the second plan is the better deal.",
    "It only takes about fifteen minutes to get set up.",
    "I promise I'm not going to read you a long script here.",
    "So tell me, what got you thinking about this in the first place?",
    "Sure, take your time, there's no rush on my side.",
    "That's totally your call, whatever feels right to you.",
    "We could knock this out today if you're up for it.",
    "The numbers speak for themselves, but I'll walk you through them.",
    "Bright sunlight streamed through the tall kitchen window.",
    "He packed five dozen jars of fresh orange marmalade.",
    "Please call me back whenever you get a free minute.",
    "I'll send over a quick summary so you have it in writing.",
    "Honestly, most folks are surprised by how simple it is.",
    "Let's circle back on the pricing in just a second.",
    "Is now still a good time, or should I catch you later?",
    "Great question, and the answer is actually pretty simple.",
    "We handle all the paperwork, so you don't have to worry.",
    "My whole goal here is just to make your life easier.",
    "You'd be locked in at that rate for the next two years.",
    "Think of it like a trial run, no strings attached.",
    "I'll be around all afternoon if anything comes up.",
    "Wow, okay, that's a bigger place than I pictured.",
    "Let me double check that I've got your details right.",
    "So your address is forty two Maple Street, is that correct?",
    "We can start as early as this Friday if you'd like.",
    "I know these calls can be annoying, so thanks for hearing me out.",
    "The kids are back in school, so mornings are a bit hectic.",
    "Coffee first, then anything is possible, in my opinion.",
    "Let's keep this simple and just focus on what you need.",
    "I'll flag the best three and you can pick from there.",
    "That's the part people usually like the most, actually.",
    "Fair enough, let's park that idea for now.",
    "Just so you know, there's no cancellation fee either.",
    "How many people are we talking about, roughly?",
    "Okay, so about a dozen, that's totally manageable.",
    "The last thing I want is to waste your time.",
    "I really appreciate your patience with all my questions.",
    "Let me be upfront, this isn't right for everyone.",
    "But for your situation, I think it's a strong fit.",
    "Sound good so far, or do you have any concerns?",
    "We can always adjust it later if your needs change.",
    "The install team is friendly, they'll be in and out.",
    "I'll text you a reminder the morning before.",
    "Honestly, I'd rather under promise and over deliver.",
    "Zebras and giraffes wandered across the wide open plain.",
    "The old clock on the wall ticked a little too loudly.",
    "Seven thirty works, or is eight o'clock easier?",
    "Cool, I've got you down for eight then.",
    "You've been super easy to talk to, I appreciate that.",
    "Let's make sure this actually saves you money first.",
    "If it doesn't, I'll be the first to tell you.",
    "I'm not going anywhere, so ask me anything.",
    "Right, let me summarize so we're on the same page.",
    "You mentioned the noise earlier, does that still bother you?",
    "Got it, so quieter is a priority for you.",
    "We can definitely work with that budget, no problem.",
    "Between the two of us, I'd go with the middle option.",
    "Alright, I think we've got a plan, how do you feel?",
    "Awesome, I'm glad this actually made sense for you.",
    "One more thing and then I'll let you go.",
    "Thanks again, seriously, have a great rest of your day.",
    "Talk soon, and don't hesitate to reach out.",
    "Um, yeah, let me think about that for a sec.",
    "Hmm, that's a good one, I'm not totally sure.",
    "Oh nice, that's actually really cool.",
    "Wait, sorry, could you say that part again?",
    "Right, right, okay, that makes a lot more sense now.",
    "No way, that's exactly what my neighbor said too.",
    "Ha, yeah, tell me about it.",
    "Okay so, big picture, here's what I'm thinking.",
]

_recorded = set(int(f[:-4]) for f in os.listdir(CLIPS_DIR)
                if f.endswith(".wav") and f[:-4].isdigit())


def webm_to_wav24k(data: bytes) -> np.ndarray:
    container = av.open(io.BytesIO(data))
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=24000)
    out = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            out.append(rf.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def write_metadata():
    with open(os.path.join(DATA_DIR, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        for i in sorted(_recorded):
            w.writerow([f"clips/{i:03d}.wav", SENTENCES[i]])


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Calliope — Voice Recorder</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:20px auto;padding:0 16px;background:#0f1115;color:#e6e6e6}
 h1{font-size:19px} .prog{color:#9aa0aa;font-size:13px;margin:6px 0 16px}
 .card{background:#171a21;border:1px solid #2a2f3a;border-radius:12px;padding:26px;margin:10px 0;min-height:90px;
       display:flex;align-items:center;justify-content:center;text-align:center;font-size:22px;line-height:1.5}
 .btns{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}
 button{padding:11px 18px;font-size:15px;border-radius:9px;border:1px solid #2a2f3a;background:#1c2029;color:#e6e6e6;cursor:pointer}
 button.rec{background:#c0392b;border:0;color:#fff;font-weight:600;min-width:150px}
 button.rec.on{background:#e74c3c;animation:pulse 1s infinite}
 @keyframes pulse{50%{opacity:.6}}
 button.nav{background:#1c2740}
 button.go{background:#5b8cff;border:0;color:#04122e;font-weight:600}
 .done{color:#8fd0a0} .todo{color:#e6a3c9}
 audio{width:100%;margin-top:14px}
 .hint{font-size:12.5px;color:#7a828f;margin-top:14px;line-height:1.6}
 .bar{height:6px;background:#232833;border-radius:4px;overflow:hidden;margin-top:6px}
 .bar>i{display:block;height:100%;background:#5b8cff}
</style></head><body>
<h1>🎙️ Record your voice</h1>
<div class=prog><span id=count></span> recorded · sentence <span id=idx></span> of <span id=total></span>
 <span id=state></span></div>
<div class=bar><i id=barfill></i></div>
<div class=card id=sentence>…</div>
<div class=btns>
 <button class=nav id=prev>← Prev</button>
 <button class=rec id=rec>● Record</button>
 <button class=nav id=next>Next →</button>
 <button class=go id=nextun>Next unrecorded</button>
</div>
<audio id=player controls></audio>
<div class=hint>
 <b>Space</b> = record / stop · <b>←/→</b> = move · re-record any time (it overwrites).<br>
 Speak in the <b>calm, natural, one-to-one tone</b> you want the agent to use — like talking to one person, not presenting.<br>
 Quiet room, keep a steady distance from the mic. You don't need all of them, but more is better (aim for most).
</div>
<script>
const S=%%SENTENCES%%, DONE=new Set(%%DONE%%);
let i=0, mr=null, chunks=[], stream=null, recording=false;
const $=id=>document.getElementById(id);
function render(){
  $('sentence').textContent=S[i];
  $('idx').textContent=i+1; $('total').textContent=S.length;
  $('count').textContent=DONE.size;
  $('state').innerHTML = DONE.has(i)?'· <span class=done>✓ recorded</span>':'· <span class=todo>not yet</span>';
  $('barfill').style.width=(100*DONE.size/S.length)+'%';
  const p=$('player'); p.src = DONE.has(i)?('/clip?i='+i+'&t='+Date.now()):'';
}
async function ensureMic(){
  if(stream)return;
  stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
}
async function toggle(){
  if(recording){ mr.stop(); return; }
  await ensureMic();
  chunks=[]; mr=new MediaRecorder(stream,{mimeType:'audio/webm'});
  mr.ondataavailable=e=>chunks.push(e.data);
  mr.onstop=async()=>{
    recording=false; $('rec').classList.remove('on'); $('rec').textContent='● Record';
    const blob=new Blob(chunks,{type:'audio/webm'});
    $('state').textContent='· saving…';
    const r=await fetch('/save?i='+i,{method:'POST',body:blob});
    if(r.ok){ DONE.add(i); render(); $('player').play?.(); }
    else { $('state').textContent='· save failed'; }
  };
  mr.start(); recording=true;
  $('rec').classList.add('on'); $('rec').textContent='■ Stop';
}
$('rec').onclick=toggle;
$('prev').onclick=()=>{i=(i-1+S.length)%S.length;render()};
$('next').onclick=()=>{i=(i+1)%S.length;render()};
$('nextun').onclick=()=>{for(let k=1;k<=S.length;k++){let j=(i+k)%S.length;if(!DONE.has(j)){i=j;break}}render()};
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();toggle();}
  else if(e.code==='ArrowLeft')$('prev').click();
  else if(e.code==='ArrowRight')$('next').click();
});
render();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == "/":
            import json
            body = (PAGE
                    .replace("%%SENTENCES%%", json.dumps(SENTENCES))
                    .replace("%%DONE%%", json.dumps(sorted(_recorded)))
                    ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p.path == "/clip":
            q = urllib.parse.parse_qs(p.query)
            i = int(q.get("i", ["-1"])[0])
            path = os.path.join(CLIPS_DIR, f"{i:03d}.wav")
            if not os.path.exists(path):
                self.send_error(404)
                return
            data = open(path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == "/save":
            q = urllib.parse.parse_qs(p.query)
            i = int(q.get("i", ["-1"])[0])
            if not (0 <= i < len(SENTENCES)):
                self.send_error(400, "bad index")
                return
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length)
            try:
                pcm = webm_to_wav24k(data)
                if len(pcm) < 2400:  # < 0.1s => probably empty
                    self.send_error(400, "clip too short")
                    return
                sf.write(os.path.join(CLIPS_DIR, f"{i:03d}.wav"), pcm, 24000, subtype="PCM_16")
                _recorded.add(i)
                write_metadata()
                print(f"saved {i:03d} ({len(pcm)/24000:.1f}s)  [{len(_recorded)}/{len(SENTENCES)}]", flush=True)
                self.send_response(200)
                self.end_headers()
            except Exception as e:
                self.send_error(500, str(e))
            return
        self.send_error(404)


if __name__ == "__main__":
    print(f"Recorder ready — open http://localhost:{PORT}", flush=True)
    print(f"Saving to {DATA_DIR}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
