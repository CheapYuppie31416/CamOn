#!/usr/bin/env python3
"""
cam_on.py (CamOn) - a webcam conversation bot for the NVIDIA DGX Spark (or any Linux box).
Version 1.0 by John W. Phillips.

When the camera sees a face, CamOn runs a conversation defined in a YAML flow
file: it speaks, listens, lets a local vLLM model interpret answers and write
replies, and branches. Flows live in the flows/ folder; the default, joke.yaml,
offers a joke, asks for a topic and tells one - or roasts people who say no.
See flows/joke.yaml for the format.

Everything runs locally:
  faces   - OpenCV YuNet (model file downloaded next to this script on first run)
  ears    - Silero VAD (bundled with faster-whisper) to hear when someone talks,
            faster-whisper speech-to-text for what they said
  brain   - a vLLM server's OpenAI-compatible API (default http://localhost:8000/v1)
  voice   - Kokoro neural TTS via kokoro-onnx (espeak-ng if Kokoro can't start)

--------------------------------------------------------------------------- setup
    sudo apt install -y espeak-ng alsa-utils v4l-utils libportaudio2 python3-venv
    python3 -m venv ~/greeter-venv && source ~/greeter-venv/bin/activate
    pip install opencv-python kokoro-onnx faster-whisper sounddevice numpy pyyaml

The Kokoro model files (~340 MB) download into ./kokoro on first run.

Start vLLM separately (see NVIDIA's DGX Spark vLLM playbook) so that
    curl http://localhost:8000/v1/models
returns your model. The bot uses the first model the server lists unless you
pass --llm-model.

--------------------------------------------------------------------------- run
    python cam_on.py --camera 0 --mic BRIO       # runs flows/joke.yaml
    python cam_on.py --flow greeter              # flows/greeter.yaml (or give any path)
    python cam_on.py --record                    # save video + audio + log of each conversation
    python cam_on.py --say "Testing, one two three" --voice am_michael   # just try a voice
    python cam_on.py --list-voices
    python cam_on.py --voice bm_george --speed 1.1
    python cam_on.py --list-mics                 # pick a microphone
    python cam_on.py --mic BRIO                  # mic by name (substring) or index
    python cam_on.py --whisper-model small.en    # more accurate ears, a bit slower
    python cam_on.py --llm-url http://localhost:8000/v1 --llm-model Qwen/Qwen3-8B

Press "q" in the video window (or Ctrl+C) to quit.
Less common settings are constants near the top of this file.
"""

import argparse
import collections
import contextlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave

import cv2
import numpy as np

__version__ = "1.0"
__author__ = "John W. Phillips"
APP_NAME = "CamOn"
VERSION_LINE = f"{APP_NAME} Version {__version__}, made by {__author__}."   # spoken by the easter egg

SAMPLE_RATE = 16000          # what Whisper wants
BLOCK_SECONDS = 0.1          # microphone chunk size

# ---- tuning constants (edit here; the everyday settings are command-line options)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
DETECT_FPS = 5               # face detections per second
FACE_CONFIDENCE = 0.8        # minimum YuNet face score (0-1)
CONFIRM_FRAMES = 3           # consecutive detections before a face counts
MIC_MIN_LEVEL = 0.01         # loudness floor (RMS) for the talk-over-the-prompt check
WHISPER_DEVICE = "cpu"       # faster-whisper's ARM build is CPU-only
WHISPER_COMPUTE = "int8"
KOKORO_DIR = os.path.join(SCRIPT_DIR, "kokoro")
FLOWS_DIR = os.path.join(SCRIPT_DIR, "flows")
RECORD_FPS = 15              # frame rate of saved recordings
RECORD_PREROLL = 2.0         # seconds kept from before a conversation starts

YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]
YUNET_FILE = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")

# =========================================================================== faces
class YuNetFaceDetector:
    def __init__(self, model_path, confidence, min_size):
        self.min_size = min_size
        self.net = cv2.FaceDetectorYN.create(model_path, "", (320, 320),
                                             score_threshold=confidence, nms_threshold=0.3, top_k=50)
        self._size = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        if self._size != (w, h):
            self.net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.net.detect(frame)
        out = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (int(v) for v in f[:4])
                if min(fw, fh) >= self.min_size:
                    out.append(((x, y, x + fw, y + fh), float(f[14])))
        return out


def ensure_yunet_model():
    if os.path.exists(YUNET_FILE):
        return YUNET_FILE
    print("Downloading YuNet face model (one time)...")
    tmp = YUNET_FILE + ".part"
    for url in YUNET_URLS:
        try:
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) > 100_000:
                os.replace(tmp, YUNET_FILE)
                return YUNET_FILE
        except Exception as e:
            print(f"  {url.split('/')[2]}: {e}")
    if os.path.exists(tmp):
        os.remove(tmp)
    raise SystemExit(f"Could not download the YuNet face model. Download it manually from\n  {YUNET_URLS[0]}\n"
                     f"and save it as {YUNET_FILE}")


def build_face_detector(confidence, min_size):
    if not hasattr(cv2, "FaceDetectorYN"):
        raise SystemExit(f"This OpenCV ({cv2.__version__}) has no YuNet face detector; it needs 4.8 or newer: "
                         f"pip install -U opencv-python")
    path = ensure_yunet_model()
    try:
        det = YuNetFaceDetector(path, confidence, min_size)
    except cv2.error as e:
        raise SystemExit(f"Could not load the YuNet model at {path} ({e}). Delete it and run again to "
                         f"re-download it.")
    print("Faces: YuNet")
    return det


def open_camera(source, width, height):
    """Open a V4L2 webcam by index (0) or device path (/dev/video0)."""
    src = str(source)
    if not src.isdigit():
        real = os.path.realpath(src)
        num = real.rsplit("video", 1)[-1] if "/dev/video" in real else ""
        if not num.isdigit():
            raise SystemExit(f"Camera '{source}' is not a /dev/videoN device. "
                             f"List cameras with `v4l2-ctl --list-devices`.")
        src = num
    cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {source}. Try another index from `v4l2-ctl --list-devices` "
                         f"and check you are in the 'video' group.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# =========================================================================== voice (TTS)
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}


def ensure_kokoro_files(folder):
    os.makedirs(folder, exist_ok=True)
    for name, url in KOKORO_FILES.items():
        path = os.path.join(folder, name)
        if os.path.exists(path):
            continue
        print(f"Downloading {name} (one time, this can take a minute)...")
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f"could not download {url} ({e}); download it by hand into {folder}")
    return os.path.join(folder, "kokoro-v1.0.onnx"), os.path.join(folder, "voices-v1.0.bin")


def play_samples(samples, sample_rate):
    """Play float32 audio through the default output (PipeWire/ALSA) with aplay."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        subprocess.run(["aplay", "-q", tmp.name], check=False)


class Speaker:
    """Blocking text-to-speech: say() returns when the sentence has finished playing.

    Uses Kokoro; if Kokoro can't start, falls back to espeak-ng so the bot still talks.
    """

    def __init__(self, voice="af_heart", speed=1.0):
        self.kokoro = None
        self.voice = voice
        self.speed = speed
        try:
            from kokoro_onnx import Kokoro
            model, voices = ensure_kokoro_files(KOKORO_DIR)
            self.kokoro = Kokoro(model, voices)
            if voice not in self.kokoro.get_voices():
                print(f"Kokoro has no voice '{voice}'; using af_heart. See --list-voices.")
                self.voice = "af_heart"
            print(f"Voice: Kokoro ({self.voice}, speed {speed})")
        except Exception as e:
            print(f"Could not start Kokoro ({e}); falling back to espeak-ng.")
            if not shutil.which("espeak-ng"):
                print("WARNING: espeak-ng not found either (sudo apt install espeak-ng); the bot will be silent.")

    def say(self, text):
        print(f"  BOT: {text}")
        try:
            if self.kokoro is not None:
                samples, sr = self.kokoro.create(text, voice=self.voice, speed=self.speed, lang="en-us")
                play_samples(samples, sr)
            elif shutil.which("espeak-ng"):
                subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", text], check=False)
        except Exception as e:
            print(f"Speech error: {e}")


# =========================================================================== ears (STT)
class SileroVAD:
    """Silero voice-activity detector, run one 32 ms frame at a time with onnxruntime.

    Uses the Silero model that ships inside faster-whisper (no download, no PyTorch).
    Feed it 16 kHz audio with block_prob(); it keeps its own state between calls,
    so call reset() at the start of each new recording.
    """
    FRAME = 512      # samples per frame at 16 kHz (32 ms), what Silero expects
    CONTEXT = 64     # samples of the previous frame the model also looks at

    def __init__(self, threshold=0.5, path=None):
        import onnxruntime
        path = path or self._bundled_model()
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        names = {i.name for i in self.session.get_inputs()}
        if {"h", "c"} <= names:
            self.kind = "hc"          # faster-whisper's export
        elif "state" in names:
            self.kind = "state"       # official silero_vad.onnx (v5+)
        else:
            raise SystemExit(f"Unrecognised Silero VAD model inputs {sorted(names)} in {path}")
        self.threshold = threshold
        self.path = path
        self.reset()

    @staticmethod
    def _bundled_model():
        import glob
        import faster_whisper
        found = sorted(glob.glob(os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad*.onnx")))
        if not found:
            raise SystemExit("Could not find the Silero VAD model inside faster-whisper. "
                             "Upgrade it: pip install -U faster-whisper")
        return found[-1]

    def reset(self):
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0

    def frame_prob(self, frame):
        x = np.concatenate([self.context, frame]).astype(np.float32)[None, :]
        if self.kind == "hc":
            out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        else:
            out, self.state = self.session.run(None, {"input": x, "state": self.state,
                                                      "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self.context = frame[-self.CONTEXT:].astype(np.float32)
        return float(np.ravel(out)[0])

    def block_prob(self, audio16k):
        """Highest speech probability among the complete frames in this chunk of 16 kHz audio."""
        self.pending = np.concatenate([self.pending, audio16k.astype(np.float32)])
        probs = []
        while len(self.pending) >= self.FRAME:
            probs.append(self.frame_prob(self.pending[:self.FRAME]))
            self.pending = self.pending[self.FRAME:]
        if probs:
            self.last_prob = max(probs)
        return self.last_prob


def endpoint_speech(blocks, is_speech, block_seconds=BLOCK_SECONDS, wait_seconds=6.0, max_seconds=10.0,
                    silence_seconds=1.0):
    """Pull audio blocks until someone has spoken and then gone quiet.

    `blocks` is an iterator of 1-D float32 arrays and `is_speech(block)` says
    whether a block contains speech (Silero VAD). Returns the utterance (with a
    little lead-in) or None if nobody started talking within wait_seconds.
    """
    preroll, speech = [], []
    started, quiet_run, elapsed = False, 0, 0.0
    for block in blocks:
        elapsed += block_seconds
        talking = is_speech(block)
        if not started:
            preroll = (preroll + [block])[-3:]
            if talking:
                started, speech = True, list(preroll)
            elif elapsed >= wait_seconds:
                return None
            continue
        speech.append(block)
        quiet_run = 0 if talking else quiet_run + 1
        if quiet_run * block_seconds >= silence_seconds or len(speech) * block_seconds >= max_seconds:
            break
    return np.concatenate(speech) if speech else None


def resample(audio, src_rate, dst_rate):
    """Resample a mono float32 signal (e.g. 48 kHz mic audio -> 16 kHz for Whisper)."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio.astype(np.float32)
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
    except ImportError:
        pass
    # numpy-only fallback: windowed-sinc low-pass below the new Nyquist, then interpolate
    if dst_rate < src_rate:
        cutoff = 0.45 * dst_rate / src_rate                  # fraction of the source rate
        n = np.arange(-64, 65)
        taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
        audio = np.convolve(audio, taps / taps.sum(), mode="same")
    t_src = np.arange(len(audio)) / src_rate
    t_dst = np.arange(int(len(audio) * dst_rate / src_rate)) / dst_rate
    return np.interp(t_dst, t_src, audio).astype(np.float32)


class Listener:
    # The microphone is opened once and stays open. Each block goes to:
    #   - the queue of whoever is listening right now (ask), if anyone, and
    #   - every "tap" (e.g. the recorder), always.
    stream = None
    active_q = None
    taps = ()

    def __init__(self, whisper_model, mic, wait_seconds, silence_seconds, barge_factor=3.0, vad_threshold=0.5):
        device, compute_type = WHISPER_DEVICE, WHISPER_COMPUTE
        self.barge_factor = barge_factor
        self.vad = SileroVAD(vad_threshold)
        print(f"Speech detection: Silero VAD ({os.path.basename(self.vad.path)}, threshold {vad_threshold})")
        import sounddevice as sd
        from faster_whisper import WhisperModel
        self.sd = sd
        self.mic = self._resolve_mic(mic)
        self.wait_seconds = wait_seconds
        self.silence_seconds = silence_seconds
        self.rate, self.channels = self._pick_format()
        name = sd.query_devices(self.mic, "input")["name"]
        note = "" if self.rate == SAMPLE_RATE else f" (records at {self.rate} Hz, converted to {SAMPLE_RATE} Hz)"
        print(f"Ears: faster-whisper {whisper_model} on {device} ({compute_type}); mic: {name}{note}")
        self.model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        self.taps = []
        self._start_stream()

    def add_tap(self, fn):
        """Call fn(block) for every microphone block from now on (keep it fast)."""
        self.taps = list(self.taps) + [fn]

    def _pick_format(self):
        """Find a sample rate / channel count the mic accepts, preferring 16 kHz mono.

        Raw USB mics (e.g. "USB PnP Sound Device" as hw:N,0) often only support
        44.1 or 48 kHz, so fall back to the device's own rate and resample later.
        """
        native = int(self.sd.query_devices(self.mic, "input")["default_samplerate"])
        for rate in dict.fromkeys([SAMPLE_RATE, native, 48000, 44100]):
            for ch in (1, 2):
                try:
                    self.sd.check_input_settings(device=self.mic, samplerate=rate, channels=ch, dtype="float32")
                    return rate, ch
                except Exception:
                    pass
        raise SystemExit("The microphone accepted none of 16000/44100/48000 Hz. Try a different --mic, "
                         "or leave --mic out and choose the mic as the system default in Settings -> Sound.")

    def _start_stream(self):
        if self.stream is None:
            self.stream = self.sd.InputStream(samplerate=self.rate, channels=self.channels, dtype="float32",
                                              blocksize=int(self.rate * BLOCK_SECONDS), device=self.mic,
                                              callback=self._callback)
            self.stream.start()

    def _callback(self, indata, frames, t, status):
        block = indata[:, 0].copy()               # first channel only
        q = self.active_q
        if q is not None:
            q.put(block)
        for tap in self.taps:
            try:
                tap(block)
            except Exception as e:
                print(f"Microphone tap error: {e}")

    @contextlib.contextmanager
    def _open_stream(self, q):
        """Route microphone blocks into q for the duration of a with-block."""
        self._start_stream()
        self.active_q = q
        try:
            yield
        finally:
            self.active_q = None

    def _resolve_mic(self, mic):
        if mic is None:
            return None
        if str(mic).isdigit():
            return int(mic)
        for i, d in enumerate(self.sd.query_devices()):
            if d["max_input_channels"] > 0 and str(mic).lower() in d["name"].lower():
                return i
        raise SystemExit(f"No microphone matching '{mic}'. Run with --list-mics.")

    def _blocks(self, q):
        while True:
            yield q.get()

    def _is_speech(self, block):
        """Silero VAD verdict for one microphone block (any sample rate)."""
        return self.vad.block_prob(resample(block, self.rate, SAMPLE_RATE)) >= self.vad.threshold

    def _transcribe(self, audio):
        audio = resample(audio, self.rate, SAMPLE_RATE)
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1,
                                            condition_on_previous_text=False)
        text = " ".join(s.text for s in segments).strip()
        print(f"  HEARD: {text!r}")
        return text

    def ask(self, speaker, prompt):
        """Speak `prompt` and return the reply, listening WHILE the prompt plays.

        Silero VAD decides what is speech. While the prompt is playing, the bot's
        own voice coming back through the mic is speech too, so during playback a
        block only counts if Silero says "speech" AND it is louder than
        barge_factor x the room noise level.

        Phase A records from the moment the prompt starts. If someone talks over
        the prompt it keeps recording until they stop, transcribes everything,
        and removes any words of the prompt itself that leaked in as echo.
        Phase B: if nothing was said over the prompt, wait for an answer as usual.
        """
        q = queue.Queue()
        self.vad.reset()
        with self._open_stream(q):
            blocks = self._blocks(q)
            rms = lambda b: float(np.sqrt(np.mean(b ** 2)))
            noise = []
            for _ in range(3):                                       # 0.3 s of room noise
                b = next(blocks)
                self._is_speech(b)                                   # warm up the VAD state
                noise.append(rms(b))
            barge_level = max(np.median(noise) * 3.0, MIC_MIN_LEVEL) * self.barge_factor

            talker = threading.Thread(target=speaker.say, args=(prompt,), daemon=True)
            talker.start()
            print("  (listening while speaking...)")
            text = self._overlap_phase(blocks, talker, barge_level, prompt, rms)
            if text:
                return text

            print("  (listening...)")
            audio = endpoint_speech(blocks, self._is_speech, wait_seconds=self.wait_seconds,
                                    silence_seconds=self.silence_seconds)
        if audio is None:
            print("  (heard nothing)")
            return ""
        return strip_echo(self._transcribe(audio), prompt)

    def _overlap_phase(self, blocks, talker, barge_level, prompt, rms, max_after=10.0):
        recorded, speech_run, quiet_run, heard, after = [], 0, 0, False, 0.0
        for block in blocks:
            recorded.append(block)
            playing = talker.is_alive()
            voice = self._is_speech(block)                           # always run, keeps VAD state current
            talking = voice and (rms(block) > barge_level if playing else True)
            speech_run = speech_run + 1 if talking else 0
            quiet_run = 0 if talking else quiet_run + 1
            if speech_run >= (2 if playing else 1):
                heard = True
            if playing:
                continue
            after += BLOCK_SECONDS
            if not heard:
                return ""                              # nobody talked over the prompt
            if quiet_run * BLOCK_SECONDS >= self.silence_seconds or after >= max_after:
                break
        text = strip_echo(self._transcribe(np.concatenate(recorded)), prompt)
        if text:
            print(f"  ANSWER (minus echo): {text!r}")
        return text


# =========================================================================== brain (vLLM)
class LLM:
    def __init__(self, base_url, model=None, timeout=60, disable_thinking=True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.model = model or self._first_model()
        print(f"Brain: {self.model} at {self.base_url}")

    def _request(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _first_model(self):
        try:
            return self._request("/models")["data"][0]["id"]
        except Exception as e:
            raise SystemExit(f"Can't reach vLLM at {self.base_url} ({e}). Is the server running? "
                             f"Test with: curl {self.base_url}/models")

    def chat(self, system, user, max_tokens=150, temperature=0.9):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.disable_thinking:   # Qwen3-style reasoning models: answer directly
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            out = self._request("/chat/completions", payload)
        except urllib.error.HTTPError as e:
            if self.disable_thinking and e.code == 400:   # server rejected the extra field
                self.disable_thinking = False
                return self.chat(system, user, max_tokens, temperature)
            raise
        return clean_for_speech(out["choices"][0]["message"]["content"] or "")


def clean_for_speech(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"[*_#`>~]|\[.*?\]\(.*?\)", "", text)        # markdown
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)  # emoji
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    return text


# =========================================================================== conversation
FILLER_WORDS = {"um", "umm", "uh", "er", "erm", "hmm", "mm", "ah", "oh", "the", "a", "an", "i", "so", "well"}


def strip_echo(text, prompt):
    """Remove the bot's own prompt from a transcript (the mic hears the speaker too).

    Finds runs of 2+ words that match the prompt in order, then also drops any
    prompt words or filler sitting inside that echoed stretch. Returns '' if
    nothing but echo and filler is left.
    """
    import difflib
    words = re.findall(r"[a-z0-9']+", text.lower())
    prompt_words = re.findall(r"[a-z0-9']+", prompt.lower())
    matcher = difflib.SequenceMatcher(None, words, prompt_words, autojunk=False)
    drop = set()
    for m in matcher.get_matching_blocks():
        if m.size >= 2:
            drop.update(range(m.a, m.a + m.size))
    if drop:
        lo, hi = min(drop), max(drop)
        pw = set(prompt_words) | FILLER_WORDS
        # grow the echoed stretch over neighbouring prompt/filler words ("the joke" misheard for "a joke")
        while lo > 0 and words[lo - 1] in pw:
            lo -= 1
        while hi + 1 < len(words) and words[hi + 1] in pw:
            hi += 1
        drop.update(i for i in range(lo, hi + 1) if words[i] in pw)
    rest = [w for i, w in enumerate(words) if i not in drop]
    if all(w in FILLER_WORDS for w in rest):
        return ""
    return " ".join(rest)


# =========================================================================== conversation flow (YAML)
TOP_KEYS = {"name", "version", "author", "start", "steps", "easter_eggs", "on_error"}
STEP_KEYS = {"say": {"say", "next"},
             "ask": {"ask", "save_as", "default", "classify", "next"},
             "generate": {"generate", "next"}}
GENERATE_KEYS = {"system", "prompt", "temperature", "max_tokens", "fallback", "save_as"}
BUILTIN_VARS = {"name", "version", "author"}
NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
VAR_RE = re.compile(r"\{(\w+)\}")
MAX_STEPS_PER_CONVERSATION = 50      # safety net in case a flow loops with nobody there
DEFAULT_ON_ERROR = "Oops, something went wrong. Try again in a moment."
CLASSIFY_TEMPLATE = (
    "Someone was asked: \"{question}\"\n"
    "You get a speech-to-text transcript of their reply, which may contain small recognition errors. "
    "Choose the ONE label that best describes what they meant:\n{options}\n"
    "Answer with the label only.")


def _yaml():
    """PyYAML, with a loader that keeps unquoted yes/no/on/off as words instead of booleans."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("Conversation flows need PyYAML: pip install pyyaml")

    class Loader(yaml.SafeLoader):
        pass

    Loader.yaml_implicit_resolvers = {
        first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    Loader.add_implicit_resolver("tag:yaml.org,2002:bool",
                                 re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
    return yaml, Loader


def _texts(step):
    """All the text in a step that can contain {variables}."""
    out = [step.get(k) for k in ("say", "ask", "default")]
    gen = step.get("generate")
    if isinstance(gen, dict):
        out += [gen.get(k) for k in ("system", "prompt", "fallback")]
    return [t for t in out if isinstance(t, str)]


def validate_flow(flow):
    """Return a list of human-readable problems with a flow (empty list = OK)."""
    if not isinstance(flow, dict):
        return ["the file must be a mapping with at least 'start:' and 'steps:'"]
    errors = [f"unknown top-level key '{k}' (allowed: {', '.join(sorted(TOP_KEYS))})"
              for k in flow if k not in TOP_KEYS]
    steps = flow.get("steps")
    if not isinstance(steps, dict) or not steps:
        return errors + ["'steps:' must be a mapping of step names to steps"]
    names = set(steps)
    if flow.get("start") not in names:
        errors.append(f"'start: {flow.get('start')}' is not one of the steps ({', '.join(steps)})")

    def check_next(where, target):
        if target is not None and target not in names:
            errors.append(f"{where}: 'next: {target}' is not a step")

    saved = set()
    for name, step in steps.items():
        where = f"step '{name}'"
        if not isinstance(step, dict):
            errors.append(f"{where}: must be a mapping"); continue
        kinds = [k for k in STEP_KEYS if k in step]
        if len(kinds) != 1:
            errors.append(f"{where}: needs exactly one of say:, ask:, generate: (found {kinds or 'none'})"); continue
        kind = kinds[0]
        errors += [f"{where}: unknown key '{k}' for a {kind} step (allowed: {', '.join(sorted(STEP_KEYS[kind]))})"
                   for k in step if k not in STEP_KEYS[kind]]
        check_next(where, step.get("next"))
        if kind in ("say", "ask") and not (isinstance(step[kind], str) and step[kind].strip()):
            errors.append(f"{where}: '{kind}:' needs some text")
        if kind == "ask":
            if "save_as" in step:
                if NAME_RE.match(str(step["save_as"])):
                    saved.add(step["save_as"])
                else:
                    errors.append(f"{where}: save_as '{step['save_as']}' must be a simple name like topic")
            elif "default" in step:
                errors.append(f"{where}: 'default:' only makes sense with save_as")
            if "classify" in step:
                labels = step["classify"]
                if not isinstance(labels, dict) or len(labels) < 2:
                    errors.append(f"{where}: classify needs at least two labels")
                else:
                    for label, opt in labels.items():
                        lw = f"{where}, label '{label}'"
                        if not LABEL_RE.match(str(label)):
                            errors.append(f"{lw}: labels must be single words")
                        if not isinstance(opt, dict) or not isinstance(opt.get("means"), str):
                            errors.append(f"{lw}: needs 'means:' describing it for the model"); continue
                        errors += [f"{lw}: unknown key '{k}' (allowed: means, next)" for k in opt if k not in ("means", "next")]
                        check_next(lw, opt.get("next"))
                if "next" in step:
                    errors.append(f"{where}: with classify, put 'next:' under each label instead")
        if kind == "generate":
            gen = step["generate"]
            if not isinstance(gen, dict) or not isinstance(gen.get("prompt"), str):
                errors.append(f"{where}: 'generate:' needs at least 'prompt:'"); continue
            errors += [f"{where}: unknown generate key '{k}' (allowed: {', '.join(sorted(GENERATE_KEYS))})"
                       for k in gen if k not in GENERATE_KEYS]
            t = gen.get("temperature", 0.9)
            if not isinstance(t, (int, float)) or not 0 <= t <= 2:
                errors.append(f"{where}: temperature must be a number from 0 to 2")
            m = gen.get("max_tokens", 150)
            if not isinstance(m, int) or m <= 0:
                errors.append(f"{where}: max_tokens must be a positive whole number")
            if "save_as" in gen:
                if NAME_RE.match(str(gen["save_as"])):
                    saved.add(gen["save_as"])
                else:
                    errors.append(f"{where}: save_as '{gen['save_as']}' must be a simple name")

    known = BUILTIN_VARS | saved
    for name, step in steps.items():
        if isinstance(step, dict):
            for text in _texts(step):
                for var in VAR_RE.findall(text):
                    if var not in known:
                        errors.append(f"step '{name}': uses {{{var}}} but no step saves it (save_as: {var})")

    eggs = flow.get("easter_eggs", [])
    if not isinstance(eggs, list):
        errors.append("'easter_eggs:' must be a list")
    else:
        for i, egg in enumerate(eggs, 1):
            if not isinstance(egg, dict) or "match" not in egg or not isinstance(egg.get("say"), str):
                errors.append(f"easter egg #{i}: needs 'match:' and 'say:'"); continue
            for pattern in (egg["match"] if isinstance(egg["match"], list) else [egg["match"]]):
                try:
                    re.compile(str(pattern))
                except re.error as e:
                    errors.append(f"easter egg #{i}: bad pattern {pattern!r} ({e})")
    if "on_error" in flow and not isinstance(flow["on_error"], str):
        errors.append("'on_error:' must be text")
    return errors


def unreachable_steps(flow):
    seen, todo = set(), [flow["start"]]
    while todo:
        name = todo.pop()
        if name in seen or name is None:
            continue
        seen.add(name)
        step = flow["steps"][name]
        todo.append(step.get("next"))
        todo += [opt.get("next") for opt in step.get("classify", {}).values()]
    return [n for n in flow["steps"] if n not in seen]


def available_flows():
    if not os.path.isdir(FLOWS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(FLOWS_DIR) if f.endswith((".yaml", ".yml")))


def resolve_flow(name):
    """Turn --flow into a file: an existing path, or a name/filename in the flows/ folder."""
    candidates = [name] if os.path.sep in name or os.path.isfile(name) else []
    candidates += [os.path.join(FLOWS_DIR, name + ext) for ext in ("", ".yaml", ".yml")]
    for path in candidates:
        if os.path.isfile(path):
            return path
    flows = available_flows()
    raise SystemExit(f"No flow called '{name}'. Available in {FLOWS_DIR}: {', '.join(flows) or '(none)'}")


def load_flow(path):
    yaml, Loader = _yaml()
    try:
        with open(path) as f:
            flow = yaml.load(f, Loader=Loader)
    except OSError as e:
        raise SystemExit(f"Can't read flow file {path}: {e}")
    except yaml.YAMLError as e:
        raise SystemExit(f"{path} is not valid YAML:\n{e}")
    errors = validate_flow(flow)
    if errors:
        raise SystemExit(f"Problems in {path}:\n" + "\n".join(f"  - {e}" for e in errors))
    flow["name"] = str(flow.get("name", APP_NAME))
    flow["version"] = str(flow.get("version", __version__))
    flow["author"] = str(flow.get("author", __author__))
    flow["_eggs"] = [([re.compile(str(p), re.I) for p in (e["match"] if isinstance(e["match"], list) else [e["match"]])],
                      e["say"]) for e in flow.get("easter_eggs", [])]
    for name in unreachable_steps(flow):
        print(f"Note: step '{name}' in {os.path.basename(path)} can never be reached.")
    return flow


def match_label(verdict, labels):
    """Find which label the model answered with (case-insensitive, ignores punctuation)."""
    lower = {l.lower(): l for l in labels}
    for word in re.findall(r"[A-Za-z0-9_-]+", verdict):
        if word.lower() in lower:
            return lower[word.lower()]
    return None


class FlowRunner:
    """Runs one conversation from a loaded flow."""

    def __init__(self, flow, speaker, listener, llm):
        self.flow, self.speaker, self.listener, self.llm = flow, speaker, listener, llm

    def fill(self, text, vars):
        return VAR_RE.sub(lambda m: str(vars.get(m.group(1), m.group(0))), text)

    def run(self, state):
        vars = {k: self.flow[k] for k in BUILTIN_VARS}
        name, count = self.flow["start"], 0
        try:
            while name is not None:
                count += 1
                if count > MAX_STEPS_PER_CONVERSATION:
                    print(f"  (stopping: more than {MAX_STEPS_PER_CONVERSATION} steps)")
                    break
                print(f"  [{name}]")
                name = self.do_step(self.flow["steps"][name], vars, state)
        except Exception as e:
            print(f"Conversation error: {e}")
            self.speaker.say(self.fill(self.flow.get("on_error", DEFAULT_ON_ERROR), vars))
        finally:
            state["status"] = "idle"
            state["ended"] = time.time()

    def do_step(self, step, vars, state):
        if "say" in step:
            state["status"] = "speaking"
            self.speaker.say(self.fill(step["say"], vars))
            return step.get("next")

        if "ask" in step:
            question = self.fill(step["ask"], vars)
            reply = self.ask(question, vars, state)
            if "save_as" in step:
                vars[step["save_as"]] = reply or self.fill(str(step.get("default", "")), vars)
            if "classify" in step:
                label = self.classify(question, reply, step["classify"], state)
                return step["classify"][label].get("next")
            return step.get("next")

        gen = step["generate"]
        state["status"] = "thinking"
        try:
            text = self.llm.chat(self.fill(gen.get("system", ""), vars), self.fill(gen["prompt"], vars),
                                 max_tokens=gen.get("max_tokens", 150), temperature=gen.get("temperature", 0.9))
        except Exception as e:
            print(f"  (model failed: {e})")
            text = ""
        text = text or self.fill(gen.get("fallback", ""), vars)
        if "save_as" in gen:
            vars[gen["save_as"]] = text
        if text:
            state["status"] = "speaking"
            self.speaker.say(text)
        return step.get("next")

    def ask(self, question, vars, state):
        """Ask, answering easter eggs (then asking again) until a normal reply arrives."""
        while True:
            state["status"] = "asking + listening"
            reply = self.listener.ask(self.speaker, question)
            egg = next((say for patterns, say in self.flow["_eggs"]
                        if reply and any(p.search(reply) for p in patterns)), None)
            if egg is None:
                return reply
            print("  (easter egg)")
            self.speaker.say(self.fill(egg, vars))

    def classify(self, question, reply, options, state):
        labels = list(options)
        fallback = labels[-1]
        if not reply.strip():
            print(f"  MEANING: {fallback} (nothing heard)")
            return fallback
        state["status"] = "thinking"
        lines = "\n".join(f"- {label}: {opt['means']}" for label, opt in options.items())
        try:
            verdict = self.llm.chat(CLASSIFY_TEMPLATE.format(question=question, options=lines), reply,
                                    max_tokens=10, temperature=0)
        except Exception as e:
            print(f"  (classification failed: {e})")
            verdict = ""
        label = match_label(verdict, labels) or fallback
        print(f"  MEANING: {label} (model said {verdict!r})")
        return label


# =========================================================================== recording
class _TeeLog:
    """Copies everything printed to the console into the current recording's log, with timestamps."""

    def __init__(self, console):
        self.console, self.file, self.line_start, self.lock = console, None, True, threading.Lock()

    def write(self, text):
        self.console.write(text)
        with self.lock:
            if self.file is None:
                return
            for part in text.splitlines(keepends=True):
                if self.line_start:
                    now = time.time()
                    self.file.write(time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 100) % 100:02d}  ")
                self.file.write(part)
                self.line_start = part.endswith("\n")
            self.file.flush()

    def flush(self):
        self.console.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class Recorder:
    """Records camera + microphone for each conversation into its own folder.

    The last RECORD_PREROLL seconds are always buffered, so a recording also shows
    the moment the person walked up. Video is written at a steady RECORD_FPS
    (frames are repeated or skipped to follow the clock) so it stays in sync with
    the audio. When ffmpeg is installed the two are combined into conversation.mp4.
    """

    def __init__(self, root, audio_rate, fps=RECORD_FPS, preroll=RECORD_PREROLL):
        self.root, self.audio_rate, self.fps, self.preroll = root, audio_rate, fps, preroll
        self.frames = collections.deque()        # (time, frame) pre-roll buffer
        self.audio = collections.deque()         # (time, block)  pre-roll buffer
        self.lock = threading.Lock()
        self.active = False
        self.folder = None
        self.ffmpeg = shutil.which("ffmpeg")
        self.log = _TeeLog(sys.stdout)
        sys.stdout = self.log
        os.makedirs(root, exist_ok=True)
        print(f"Recording conversations to {os.path.abspath(root)}"
              + ("" if self.ffmpeg else " (install ffmpeg to also get a combined conversation.mp4)"))

    # ---- inputs (called continuously)
    def add_frame(self, frame, t=None):
        t = time.time() if t is None else t
        with self.lock:
            if self.active:
                self._write_frame(frame, t)
            else:
                self.frames.append((t, frame.copy()))
                while self.frames and t - self.frames[0][0] > self.preroll:
                    self.frames.popleft()

    def add_audio(self, block, t=None):
        t = time.time() if t is None else t
        with self.lock:
            if self.active:
                self._write_audio(block)
            else:
                self.audio.append((t, block))
                while self.audio and t - self.audio[0][0] > self.preroll:
                    self.audio.popleft()

    # ---- start / stop (called by the main loop)
    def start(self):
        with self.lock:
            if self.active:
                return
            self.folder = os.path.join(self.root, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.folder, exist_ok=True)
            self.t0 = min([t for t, _ in self.frames] + [t for t, _ in self.audio] + [time.time()])
            self.video, self.frames_written = None, 0
            self.wav = wave.open(os.path.join(self.folder, "audio.wav"), "wb")
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)
            self.wav.setframerate(self.audio_rate)
            # pad audio so it starts at t0 like the video, then flush the buffers
            if self.audio:
                gap = int((self.audio[0][0] - self.t0) * self.audio_rate)
                self._write_audio(np.zeros(max(gap, 0), dtype=np.float32))
            for _, block in self.audio:
                self._write_audio(block)
            for t, frame in self.frames:
                self._write_frame(frame, t)
            self.frames.clear()
            self.audio.clear()
            self.active = True
        with self.log.lock:
            self.log.file = open(os.path.join(self.folder, "log.txt"), "w")
            self.log.line_start = True
        print(f"  (recording to {self.folder})")

    def stop(self):
        with self.lock:
            if not self.active:
                return
            self.active = False
            had_video = self.video is not None
            if had_video:
                self.video.release()
            self.wav.close()
            folder = self.folder
        print(f"  (recording saved: {folder})")
        with self.log.lock:
            self.log.file.close()
            self.log.file = None
        if self.ffmpeg and had_video:
            threading.Thread(target=self._combine, args=(folder,), daemon=True).start()

    def close(self):
        self.stop()
        sys.stdout = self.log.console

    # ---- internals (lock held)
    def _write_frame(self, frame, t):
        if self.video is None:
            h, w = frame.shape[:2]
            self.video = cv2.VideoWriter(os.path.join(self.folder, "video.avi"),
                                         cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        due = int((t - self.t0) * self.fps) + 1          # frames that should exist by time t
        while self.frames_written < due:
            self.video.write(frame)
            self.frames_written += 1

    def _write_audio(self, block):
        pcm = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)
        self.wav.writeframes(pcm.tobytes())

    def _combine(self, folder):
        video, audio, out = (os.path.join(folder, n) for n in ("video.avi", "audio.wav", "conversation.mp4"))
        result = subprocess.run([self.ffmpeg, "-y", "-loglevel", "error", "-i", video, "-i", audio,
                                 "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                                 "-c:a", "aac", "-shortest", out], capture_output=True, text=True)
        if result.returncode == 0 and os.path.getsize(out) > 0:
            os.remove(video)                             # keep the mp4, audio.wav and log.txt
        else:
            print(f"(ffmpeg could not combine {folder}: {result.stderr.strip()[:200]})")


# =========================================================================== main
def main():
    p = argparse.ArgumentParser(description=f"{APP_NAME} - see a face, run a YAML-defined conversation with it.")
    p.add_argument("--version", action="version", version=VERSION_LINE)

    p.add_argument("--flow", default="joke",
                   help="Conversation to run: a name from the flows/ folder (joke, greeter, ...) or a path "
                        "to a YAML file. Default: joke.")

    g = p.add_argument_group("camera")
    g.add_argument("--camera", default="0", help="Camera index (0) or /dev/videoN.")
    g.add_argument("--min-face", type=int, default=60,
                   help="Ignore faces smaller than this many pixels, so only people close by are asked.")
    g.add_argument("--absent-seconds", type=float, default=1,
                   help="No face for this long before the next arrival can start a new conversation.")
    g.add_argument("--cooldown", type=float, default=3, help="Seconds after a conversation before another.")
    g.add_argument("--no-window", action="store_true", help="Run without the video window.")
    g.add_argument("--record", nargs="?", const=os.path.join(SCRIPT_DIR, "recordings"), default=None,
                   metavar="FOLDER", help="Record camera + microphone for each conversation (default folder: "
                                          "recordings/ next to this script).")

    g = p.add_argument_group("listening")
    g.add_argument("--mic", default=None, help="Microphone index or part of its name (default: system default).")
    g.add_argument("--list-mics", action="store_true", help="List microphones and exit.")
    g.add_argument("--whisper-model", default="base.en",
                   help="Speech-to-text model: tiny.en, base.en, small.en (more accurate, slower) ...")
    g.add_argument("--vad-threshold", type=float, default=0.5,
                   help="Silero speech probability (0-1) that counts as speech. Lower (0.3) if quiet "
                        "answers are missed, higher (0.7) if noises start a recording.")
    g.add_argument("--answer-wait", type=float, default=6, help="Seconds to wait for someone to start answering.")
    g.add_argument("--end-silence", type=float, default=1.0, help="Seconds of quiet that end an answer.")
    g.add_argument("--barge-factor", type=float, default=3.0,
                   help="While the bot is talking, your voice must be this many times louder than room noise "
                        "to count (keeps its own voice from triggering it). ~1.5 with PipeWire echo cancellation.")

    g = p.add_argument_group("language model")
    g.add_argument("--llm-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL.")
    g.add_argument("--llm-model", default=None, help="Model name (default: first model the server lists).")

    g = p.add_argument_group("voice")
    g.add_argument("--voice", default="af_heart",
                   help="Kokoro voice, e.g. af_heart, af_bella, am_michael, am_fenrir, bm_george.")
    g.add_argument("--speed", type=float, default=1.0, help="Speaking speed (0.8 slower, 1.2 faster).")
    g.add_argument("--list-voices", action="store_true", help="List Kokoro voices and exit.")
    g.add_argument("--say", default=None, help="Speak this text with the chosen voice and exit (voice test).")
    args = p.parse_args()
    print(VERSION_LINE)

    if args.list_mics:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"{i:3d}  {d['name']}")
        return

    if args.list_voices:
        from kokoro_onnx import Kokoro
        model, voices = ensure_kokoro_files(KOKORO_DIR)
        names = sorted(Kokoro(model, voices).get_voices())
        print("First letter = accent (a US, b British); second = f female / m male.")
        print("  ".join(names))
        return

    if not args.say:                     # check the flow before loading any models
        flow_path = resolve_flow(args.flow)
        flow = load_flow(flow_path)
        print(f"Flow: {os.path.relpath(flow_path)} ({len(flow['steps'])} steps, starts at '{flow['start']}')")

    speaker = Speaker(args.voice, args.speed)
    if args.say:
        speaker.say(args.say)
        return

    llm = LLM(args.llm_url, args.llm_model)
    listener = Listener(args.whisper_model, args.mic, args.answer_wait, args.end_silence,
                        barge_factor=args.barge_factor, vad_threshold=args.vad_threshold)
    runner = FlowRunner(flow, speaker, listener, llm)
    recorder = None
    if args.record:
        recorder = Recorder(args.record, listener.rate)
        listener.add_tap(recorder.add_audio)
    detector = build_face_detector(FACE_CONFIDENCE, args.min_face)
    cap = open_camera(args.camera, CAMERA_WIDTH, CAMERA_HEIGHT)

    state = {"status": "idle", "ended": -1e9}
    seen_streak, present, last_seen, last_detect = 0, False, 0.0, 0.0
    detect_interval = 1.0 / DETECT_FPS
    boxes = []

    print("Watching for faces. Press 'q' in the window or Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost camera feed.")
                break
            now = time.time()
            if recorder:
                recorder.add_frame(frame, now)          # before anything is drawn on it
                if recorder.active and state["status"] == "idle":
                    recorder.stop()

            if now - last_detect >= detect_interval:
                last_detect = now
                boxes = detector.detect(frame)
                seen_streak = seen_streak + 1 if boxes else 0
                if boxes:
                    last_seen = now

                if (not present and seen_streak >= CONFIRM_FRAMES
                        and state["status"] == "idle" and now - state["ended"] >= args.cooldown):
                    present = True
                    print(f"[{time.strftime('%H:%M:%S')}] Face detected - starting conversation")
                    state["status"] = "starting"
                    if recorder:
                        recorder.start()
                    threading.Thread(target=runner.run, args=(state,), daemon=True).start()

                if present and state["status"] == "idle" and now - last_seen >= args.absent_seconds:
                    present = False
                    print(f"[{time.strftime('%H:%M:%S')}] Nobody here - ready for the next face")

            if not args.no_window:
                for (x1, y1, x2, y2), score in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"faces: {len(boxes)}   {state['status']}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if recorder and recorder.active:
                    cv2.circle(frame, (frame.shape[1] - 70, 25), 8, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (frame.shape[1] - 55, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(flow["name"], frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            recorder.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
