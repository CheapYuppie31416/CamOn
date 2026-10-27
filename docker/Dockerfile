# CamOn runtime image for the DGX Spark (ARM64).
#
# Only the dependencies live in the image. docker-compose.yml mounts ~/CamOn at /app,
# so cam_on.py, flows/, the downloaded models (kokoro/, YuNet, Whisper) and recordings/
# all stay in that folder - editing a flow or the script needs no rebuild.

FROM python:3.12-slim-bookworm

# espeak-ng   - fallback voice, and Kokoro's phonemizer
# alsa-utils  - aplay, used to play speech
# libasound2-plugins - ALSA "pulse" plugin, so audio goes to the host's PipeWire
# libportaudio2 - microphone access for sounddevice
# ffmpeg      - combines recordings into conversation.mp4
# v4l-utils   - v4l2-ctl, handy for finding cameras
# libgl1 ... libdbus-1-3 - only needed for OpenCV's video window (X11)
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng alsa-utils libasound2-plugins libportaudio2 ffmpeg v4l-utils \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libxkbcommon-x11-0 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Send all ALSA audio (aplay out, microphone in) to the host's PipeWire via its PulseAudio socket.
RUN printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf

RUN pip install --no-cache-dir \
        opencv-python kokoro-onnx faster-whisper sounddevice numpy pyyaml scipy

# Run as the same user id as you on the host, so the container can use your PipeWire
# session and write into ~/CamOn. Change with: docker compose build --build-arg UID=$(id -u)
ARG UID=1000
RUN useradd -m -u ${UID} -G video,audio camon
USER camon
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

CMD ["python", "cam_on.py", "--no-window"]
