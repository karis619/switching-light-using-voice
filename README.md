# Offline Voice-Controlled Relay Bulb System

An offline, local microphone-to-microcontroller voice control system that recognizes spoken **`"on"`** and **`"off"`** commands (while rejecting `"other"` background noise/speech) and switches a relay-driven bulb over a serial COM port without cloud services or external speech APIs.

## Key Features
- **100% Offline & Local Inference:** Captures live audio via `sounddevice`, computes temporal/spectral/onset/peak features in Python, and classifies commands locally using a trained `scikit-learn` pipeline (`on_off_model.joblib`).
- **Onset, Peak & Pronunciation Features:** Evaluates `1.0 s` (`16 kHz`, mono) windows across start, middle, and end segments together with onset zero-crossing/centroid cues, peak position, and energy rise sharpness to separate `"on"` from `"off"`.
- **Leak-Free Dataset & Evaluation:** Uses group-aware splitting (`StratifiedGroupKFold`) over `dataset_prepared.npz` so augmented versions of the same recording never leak across train and test splits.
- **Multi-Station MCU Support:** Transmits newline-terminated `ON\n` and `OFF\n` commands at `115200` baud over USB serial (`pyserial`) to independent relay stations running **Arduino Uno**, **ESP32**, or **ESP8266**.

## Requirements

### Software & Python Libraries
- **Python:** Python `3.10+` (tested on Windows with `pip` / virtual environment or VS Code).
- **Required Python Packages:**
  - `numpy` — array handling, signal math, ring buffers, and `.npz` dataset storage
  - `scipy` — scientific/signal support routines used by audio/ML stacks
  - `librosa` — audio loading, silence trimming, RMS/ZCR/spectral centroid/rolloff/MFCC extraction, and pitch-shift augmentation
  - `soundfile` — reading and writing `16 kHz` mono `PCM_16` `.wav` files
  - `sounddevice` — real-time microphone audio capture (`InputStream` and `rec`)
  - `scikit-learn` — `StandardScaler`, `HistGradientBoostingClassifier`, `StratifiedGroupKFold`, and evaluation metrics
  - `joblib` — saving and loading the trained model bundle (`on_off_model.joblib`)
  - `pyserial` — COM port discovery and serial UART communication (`ON\n` / `OFF\n`) with the MCU

Install all Python dependencies with:

```bash
pip install numpy scipy librosa soundfile sounddevice scikit-learn joblib pyserial
