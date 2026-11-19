import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5.0          # 5.0s per capture
DEFAULT_SAMPLES = 6     # Low number of raw captures per class
OUTPUT_ROOT = "lights_off_data"
VALID_CLASSES = ("lights_off", "other")


def capture_samples(label: str, count: int = DEFAULT_SAMPLES) -> None:
    save_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.lower().endswith(".wav")])

    print(f"\n=== Capturing '{label}' ({count} clips x {DURATION:.1f}s) ===")
    if label == "lights_off":
        print("Tip: Say 'lights off' 3-4 times during each 5s clip with short pauses,")
        print("     varying volume, pace, and distance from the mic.")
    else:
        print("Tip: Include room noise, fan/typing sounds, silence, random words,")
        print("     and phrases like 'lights on' or 'turn off' so it won't false-trigger.")

    for idx in range(count):
        print(f"\nClip {idx + 1}/{count}: Get ready...")
        time.sleep(0.8)
        print(f">>> RECORDING ({DURATION:.1f}s) <<<")
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        audio = recording.flatten()
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0

        filename = f"{label}_{existing + idx + 1:03d}.wav"
        out_path = os.path.join(save_dir, filename)
        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"Saved {out_path} (rms={rms:.4f}, peak={peak:.3f})")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in VALID_CLASSES:
        print(f"Usage: python collect_lights_off.py <{'|'.join(VALID_CLASSES)}> [num_clips={DEFAULT_SAMPLES}]")
        sys.exit(1)

    target_class = sys.argv[1]
    num_clips = int(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_SAMPLES
    if num_clips < 1:
        raise ValueError("num_clips must be at least 1.")

    capture_samples(target_class, num_clips)