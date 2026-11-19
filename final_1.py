import time
import joblib
import librosa
import numpy as np
import serial
import serial.tools.list_ports
import sounddevice as sd

from data_preparation import extract_features

MODEL_PATH = "on_off_model.joblib"

# Serial / COM port settings (Arduino Uno, ESP32, ESP8266)
COM_PORT = None          # Set e.g. "COM4" to force a port, or leave None to auto-detect
BAUD_RATE = 115200       # Works cleanly on Uno, ESP32, and ESP8266 (match Serial.begin rate)
SERIAL_TIMEOUT = 1.0
MCU_BOOT_DELAY = 1.5
DISABLE_DTR_RTS = False  # Set True if opening the port causes unwanted ESP32/ESP8266 reset/pin toggles

# Detection & relay debounce settings
CONF_THRESHOLD = 0.65
MIN_MARGIN = 0.18
MIN_RMS = 0.020
HOP_FRAMES = 2           # 2 * 80 ms = 160 ms evaluation interval
REQUIRED_HITS = 2
COOLDOWN_SEC = 0.70
SEND_DUPLICATE_COMMANDS = False


def list_available_ports() -> list:
    ports = list(serial.tools.list_ports.comports())
    if ports:
        print("Detected COM ports:")
        for p in ports:
            print(f"  - {p.device}: {p.description} [{p.hwid}]")
    else:
        print("No COM ports currently detected.")
    return ports


def resolve_com_port(preferred_port: str | None) -> str:
    if preferred_port:
        return preferred_port

    ports = list_available_ports()
    board_keywords = (
        "usb", "ch340", "ch341", "cp210", "ftdi",
        "arduino", "esp32", "esp8266", "wemos", "nodemcu", "jtag", "serial"
    )
    for p in ports:
        info = f"{p.description or ''} {p.hwid or ''}".lower()
        if any(k in info for k in board_keywords):
            return p.device

    if ports:
        return ports[0].device

    raise RuntimeError("No COM port found. Connect your Arduino Uno, ESP32, or ESP8266 and try again.")


def open_serial_port(port: str, baudrate: int) -> serial.Serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baudrate
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = SERIAL_TIMEOUT
    ser.write_timeout = 1.0

    if DISABLE_DTR_RTS:
        ser.dtr = False
        ser.rts = False

    ser.open()
    time.sleep(MCU_BOOT_DELAY)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def send_mcu_command(ser: serial.Serial, command: str) -> None:
    msg = f"{command.strip().upper()}\n".encode("utf-8")
    ser.write(msg)
    ser.flush()


def extract_peak_window(ring: np.ndarray, win_len: int) -> np.ndarray:
    """Slice a non-wrapping window aligned near the highest-energy region of the ring buffer."""
    y = np.asarray(ring, dtype=np.float32).flatten()
    if len(y) <= win_len:
        return np.pad(y, (max(0, win_len - len(y)), 0), mode="constant")[-win_len:]

    frame_len = 512
    hop = 160
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    best_frame = int(np.argmax(rms_frames)) if rms_frames.size else len(y) // (2 * hop)
    peak_sample = best_frame * hop + frame_len // 2

    start = peak_sample - int(0.55 * win_len)
    start = max(0, min(len(y) - win_len, start))
    return y[start : start + win_len].astype(np.float32)


def listen_once(
    pipe,
    classes: list[str],
    sr: int,
    win_len: int,
    ring: np.ndarray,
    stream: sd.InputStream,
    state: dict,
) -> tuple[str, dict[str, float]]:
    chunk_len = stream.blocksize or 1280

    while True:
        pcm, overflowed = stream.read(chunk_len)
        if overflowed:
            continue

        state["frame_idx"] += 1
        samples = pcm.flatten().astype(np.float32)
        ring[:] = np.roll(ring, -len(samples))
        ring[-len(samples) :] = samples

        if state["frame_idx"] % HOP_FRAMES != 0:
            continue

        rms = float(np.sqrt(np.mean(ring ** 2)))
        if rms < MIN_RMS:
            state["hit_count"] = 0
            state["last_class"] = None
            continue

        if time.monotonic() - state["last_trigger_time"] < COOLDOWN_SEC:
            continue

        window = extract_peak_window(ring, win_len)
        feat = extract_features(window, sr=sr).reshape(1, -1)
        probs = pipe.predict_proba(feat)[0]
        prob_map = {str(c): float(p) for c, p in zip(classes, probs)}

        ranked = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        best_class, best_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_prob - second_prob

        if best_class in ("on", "off") and best_prob >= CONF_THRESHOLD and margin >= MIN_MARGIN:
            if best_class == state["last_class"]:
                state["hit_count"] += 1
            else:
                state["last_class"] = best_class
                state["hit_count"] = 1

            if state["hit_count"] >= REQUIRED_HITS:
                state["last_trigger_time"] = time.monotonic()
                state["hit_count"] = 0
                state["last_class"] = None
                return best_class, prob_map
        else:
            state["hit_count"] = 0
            state["last_class"] = None


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    classes = [str(c) for c in bundle["classes"]]
    sr = int(bundle["sample_rate"])
    win_len = int(round(float(bundle["win_sec"]) * sr))

    port_name = resolve_com_port(COM_PORT)
    ser = open_serial_port(port_name, BAUD_RATE)
    print(f"Serial ready on {port_name} @ {BAUD_RATE} baud.")
    print("Mic ready. Say 'on' or 'off' (Ctrl+C to quit)...")

    chunk_len = 1280  # 80 ms at 16 kHz
    ring = np.zeros(win_len, dtype=np.float32)
    state = {
        "frame_idx": 0,
        "hit_count": 0,
        "last_class": None,
        "last_trigger_time": 0.0,
    }
    last_sent = None

    try:
        with sd.InputStream(
            samplerate=sr, channels=1, dtype="float32", blocksize=chunk_len
        ) as stream:
            while True:
                heard, prob_map = listen_once(pipe, classes, sr, win_len, ring, stream, state)
                scores = ", ".join(f"{k}={v:.2f}" for k, v in prob_map.items())

                if SEND_DUPLICATE_COMMANDS or heard != last_sent:
                    send_mcu_command(ser, heard)
                    last_sent = heard
                    print(f"Heard: {heard} -> TX '{heard.upper()}'  ({scores})", flush=True)
                else:
                    print(f"Heard: {heard} -> Already {heard.upper()} (skipped)  ({scores})", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if ser.is_open:
            ser.close()
            print(f"Closed {port_name}.")


if __name__ == "__main__":
    main()