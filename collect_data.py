import os
import sys
import time
import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
RECORD_SEC = 1.5          # Short capture per single utterance
TARGET_WIN_SEC = 1.0      # Exact 1.0s (16,000 samples) window saved for training
DEFAULT_SAMPLES = 30
OUTPUT_ROOT = "on_off_data"
VALID_CLASSES = ("on", "off", "other")


def normalize_to_target_window(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    win_sec: float = TARGET_WIN_SEC,
    is_command: bool = True,
) -> np.ndarray:
    target_len = int(round(sr * win_sec))
    y = np.asarray(audio, dtype=np.float32).flatten()

    # Remove DC offset
    y = y - float(np.mean(y)) if y.size else np.zeros(target_len, dtype=np.float32)

    if is_command and y.size > 0:
        # Trim leading/trailing silence around the spoken 'on' or 'off'
        trimmed, _ = librosa.effects.trim(y, top_db=25)
        if trimmed.size > 0:
            y = trimmed

    if y.size == target_len:
        out = y
    elif y.size > target_len:
        # Take the highest-energy 1.0s segment so the word 'on'/'off' stays centered
        frame_len = 512
        hop = 160
        if y.size >= frame_len:
            rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
            best_frame = int(np.argmax(rms_frames))
            center_sample = best_frame * hop + frame_len // 2
            start = max(0, min(y.size - target_len, center_sample - target_len // 2))
        else:
            start = (y.size - target_len) // 2
        out = y[start : start + target_len]
    else:
        # Center short speech inside a 1.0s buffer
        pad_total = target_len - y.size
        left = pad_total // 2
        right = pad_total - left
        out = np.pad(y, (left, right), mode="constant")

    # Gentle peak limiting without distorting normal speech
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.95:
        out = out * (0.95 / peak)

    return out.astype(np.float32)


def capture_samples(label: str, count: int = DEFAULT_SAMPLES) -> None:
    save_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.lower().endswith(".wav")])

    print(f"\n=== Capturing '{label}' ({count} single-utterance clips -> {TARGET_WIN_SEC:.1f}s WAVs) ===")
    if label == "on":
        print("Tip: Say 'on' ONCE clearly each time, varying pace, volume, and distance slightly.")
    elif label == "off":
        print("Tip: Say 'off' ONCE clearly each time, varying pace, volume, and distance slightly.")
    else:
        print("Tip: Include room noise, fan/typing, silence, and other words (like 'light', 'up', 'down', 'stop').")

    for idx in range(count):
        print(f"\nClip {idx + 1}/{count}: Get ready...")
        time.sleep(0.6)
        print(f">>> SAY '{label.upper()}' NOW ({RECORD_SEC:.1f}s) <<<" if label != "other" else f">>> RECORDING OTHER ({RECORD_SEC:.1f}s) <<<")

        recording = sd.rec(
            int(round(RECORD_SEC * SAMPLE_RATE)),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        window = normalize_to_target_window(
            recording.flatten(),
            sr=SAMPLE_RATE,
            win_sec=TARGET_WIN_SEC,
            is_command=(label in ("on", "off")),
        )

        rms = float(np.sqrt(np.mean(window ** 2)))
        peak = float(np.max(np.abs(window)))

        filename = f"{label}_{existing + idx + 1:03d}.wav"
        out_path = os.path.join(save_dir, filename)
        sf.write(out_path, window, SAMPLE_RATE, subtype="PCM_16")

        status = "OK"
        if label in ("on", "off") and rms < 0.015:
            status = "LOW VOLUME (speak a bit closer/clearer)"
        elif peak >= 0.94:
            status = "NEAR CLIPPING (step back slightly)"

        print(f"Saved {out_path} [16kHz mono PCM_16, 1.0s, rms={rms:.4f}, peak={peak:.3f}] -> {status}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_CLASSES:
        print(f"Usage: python collect_on_off.py <{'|'.join(VALID_CLASSES)}> [num_clips={DEFAULT_SAMPLES}]")
        sys.exit(1)

    target_class = sys.argv[1]
    num_clips = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_SAMPLES
    if num_clips < 1:
        raise ValueError("num_clips must be at least 1.")

    capture_samples(target_class, num_clips)