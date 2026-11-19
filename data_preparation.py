import glob
import os
import librosa
import numpy as np

DATA_ROOT = "on_off_data"
OUT_NPZ = "dataset_prepared.npz"

SAMPLE_RATE = 16000
WIN_SEC = 1.0
N_MFCC = 13
CLASSES = ["other", "on", "off"]  # 0 -> other, 1 -> on, 2 -> off
SEED = 42


def _safe_delta_width(n_frames: int) -> int:
    """Return a valid odd delta filter width less than or equal to n_frames."""
    if n_frames < 3:
        return 1
    w = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    return max(3, w)


def _slice_stats(y_seg: np.ndarray, sr: int) -> np.ndarray:
    """Extract spectral, cepstral, and energy statistics for a temporal segment."""
    y_seg = np.asarray(y_seg, dtype=np.float32).flatten()
    if y_seg.size < 256:
        y_seg = np.pad(y_seg, (0, 256 - y_seg.size), mode="constant")

    n_fft = 512 if y_seg.size >= 512 else 256
    hop = max(64, n_fft // 2)

    mfcc = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)
    width = _safe_delta_width(mfcc.shape[1])
    if width > 1:
        delta1 = librosa.feature.delta(mfcc, order=1, width=width)
        d1_mean = np.mean(delta1, axis=1)
    else:
        d1_mean = np.zeros(N_MFCC, dtype=np.float32)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(y=y_seg, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y_seg, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85
    )[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_seg, frame_length=n_fft, hop_length=hop)[0]
    rms_frames = librosa.feature.rms(y=y_seg, frame_length=n_fft, hop_length=hop)[0]

    rms_val = float(np.sqrt(np.mean(y_seg ** 2)))
    peak_val = float(np.max(np.abs(y_seg)))
    crest = peak_val / (rms_val + 1e-6)

    extra = np.array(
        [
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(rolloff)),
            float(np.mean(zcr)),
            float(np.max(zcr)),
            rms_val,
            peak_val,
            crest,
            float(np.max(rms_frames)) if rms_frames.size else peak_val,
        ],
        dtype=np.float32,
    )
    return np.concatenate([mfcc_mean, mfcc_std, d1_mean, extra]).astype(np.float32)


def extract_onset_and_peak_cues(y_win: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Capture starting-letter articulation, onset timing, peak position, and rise profile."""
    y = np.asarray(y_win, dtype=np.float32).flatten()
    frame_len = 256
    hop = 128

    if y.size < frame_len:
        y = np.pad(y, (0, frame_len - y.size), mode="constant")

    rms_env = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    zcr_env = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_len, hop_length=hop)[0]
    cent_env = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=hop)[0]

    n_frames = max(1, len(rms_env))
    peak_idx = int(np.argmax(rms_env))
    peak_env = float(rms_env[peak_idx])
    peak_pos_ratio = peak_idx / max(1, n_frames - 1)

    # First frame where energy rises to at least 25% of peak envelope
    onset_thresh = 0.25 * peak_env
    above_idx = np.flatnonzero(rms_env >= onset_thresh)
    onset_frame = int(above_idx[0]) if above_idx.size > 0 else 0
    onset_pos_ratio = onset_frame / max(1, n_frames - 1)

    # Starting segment (~130 ms around onset) for initial consonant / vowel-start cues
    onset_sample = onset_frame * hop
    start_seg = y[max(0, onset_sample - 160) : min(len(y), onset_sample + 1920)]
    if start_seg.size < 256:
        start_seg = np.pad(start_seg, (0, 256 - start_seg.size), mode="constant")

    onset_zcr = float(
        np.mean(librosa.feature.zero_crossing_rate(y=start_seg, frame_length=256, hop_length=128))
    )
    onset_cent = float(
        np.mean(librosa.feature.spectral_centroid(y=start_seg, sr=sr, n_fft=512, hop_length=128))
    )
    onset_rms = float(np.sqrt(np.mean(start_seg ** 2)))

    # Energy rise sharpness from onset to main peak
    frames_to_peak = max(1, peak_idx - onset_frame)
    rise_slope = (peak_env - float(rms_env[onset_frame])) / frames_to_peak

    peak_abs = float(np.max(np.abs(y)))
    onset_to_peak_ratio = onset_rms / (peak_abs + 1e-6)

    # Peak segment (~80 ms centered on peak) to compare against starting sound
    peak_sample = peak_idx * hop
    peak_seg = y[max(0, peak_sample - 640) : min(len(y), peak_sample + 640)]
    if peak_seg.size < 256:
        peak_seg = np.pad(peak_seg, (0, 256 - peak_seg.size), mode="constant")

    peak_cent = float(
        np.mean(librosa.feature.spectral_centroid(y=peak_seg, sr=sr, n_fft=512, hop_length=128))
    )
    cent_shift_onset_to_peak = onset_cent - peak_cent

    return np.array(
        [
            onset_pos_ratio,
            peak_pos_ratio,
            onset_to_peak_ratio,
            rise_slope,
            onset_zcr,
            onset_cent,
            peak_cent,
            cent_shift_onset_to_peak,
            float(np.mean(cent_env)),
            float(np.std(zcr_env)),
        ],
        dtype=np.float32,
    )


def extract_features(y_win: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Build full feature vector: start/mid/end pronunciation slices + onset/peak cues."""
    y_win = np.asarray(y_win, dtype=np.float32).flatten()
    y_emp = librosa.effects.preemphasis(y_win)

    n = len(y_emp)
    start_slice = _slice_stats(y_emp[: n // 3], sr)
    mid_slice = _slice_stats(y_emp[n // 3 : 2 * n // 3], sr)
    end_slice = _slice_stats(y_emp[2 * n // 3 :], sr)
    cues = extract_onset_and_peak_cues(y_emp, sr=sr)

    return np.concatenate([start_slice, mid_slice, end_slice, cues]).astype(np.float32)


def augment_clip(y: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    y = np.asarray(y, dtype=np.float32).flatten()
    variants = [y]

    for g in (0.75, 1.25):
        variants.append(np.clip(y * g, -1.0, 1.0).astype(np.float32))

    for steps in (-1.0, 1.0):
        shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        variants.append(np.clip(shifted, -1.0, 1.0).astype(np.float32))

    noise = np.random.normal(0.0, 0.003, size=len(y)).astype(np.float32)
    variants.append(np.clip(y + noise, -1.0, 1.0).astype(np.float32))

    return variants


def fit_window_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode="constant").astype(np.float32)
    if len(audio) > target_len:
        return audio[:target_len].astype(np.float32)
    return audio.astype(np.float32)


def _describe_feature_layout() -> list[tuple[str, int, int]]:
    """Return (group_name, start_idx, end_idx) for the 157-dim feature vector."""
    # Each _slice_stats slice: 13 mfcc_mean + 13 mfcc_std + 13 d1_mean + 9 extra = 48 dims
    # Three slices (start, mid, end) = 144 dims + 10 onset/peak cues = 154 dims
    # Let's compute exact lengths dynamically from a dummy signal so layout is never hardcoded wrongly.
    return []


def arrange_dataset(data_root: str = DATA_ROOT):
    np.random.seed(SEED)
    target_len = int(round(WIN_SEC * SAMPLE_RATE))

    dummy = np.zeros(target_len, dtype=np.float32)
    dummy[target_len // 3 : target_len // 3 + 1600] = 0.1
    slice_dim = len(_slice_stats(dummy[: target_len // 3], SAMPLE_RATE))
    cue_dim = len(extract_onset_and_peak_cues(dummy, SAMPLE_RATE))
    total_dim = 3 * slice_dim + cue_dim

    label_to_name = {idx: name for idx, name in enumerate(CLASSES)}
    X_rows = []
    y_labels = []
    label_names = []
    groups = []
    source_files = []

    print("=== Classification & Label Mapping ===")
    for idx, cname in label_to_name.items():
        print(f"  Target Class {idx}  -->  '{cname}'")

    print("\n=== Feature Group Layout ===")
    print(f"  [000 : {slice_dim:03d}]      Start segment stats (opening / starting sound)")
    print(f"  [{slice_dim:03d} : {2*slice_dim:03d}]      Middle segment stats (core vowel / body)")
    print(f"  [{2*slice_dim:03d} : {3*slice_dim:03d}]      End segment stats (closing / trailing sound)")
    print(f"  [{3*slice_dim:03d} : {total_dim:03d}]      Onset ZCR/centroid, onset-to-peak ratio, peak timing & slope")
    print(f"  Total feature dimension per window: {total_dim}\n")

    print("=== Data Preparation & Augmentation Log ===")
    for label_idx, cname in enumerate(CLASSES):
        class_dir = os.path.join(data_root, cname)
        wav_files = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        if not wav_files:
            raise FileNotFoundError(f"Missing .wav recordings in '{class_dir}'.")

        print(f"  Folder '{class_dir}' -> mapped to y={label_idx} ('{cname}') | {len(wav_files)} raw .wav files")

        for wav_path in wav_files:
            raw_audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            base_win = fit_window_length(raw_audio, target_len)
            group_id = f"{cname}_{os.path.basename(wav_path)}"

            for aug_idx, aug_win in enumerate(augment_clip(base_win, SAMPLE_RATE)):
                feat = extract_features(aug_win, SAMPLE_RATE)
                if feat.shape[0] != total_dim or np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                    raise ValueError(f"Invalid feature vector in {wav_path} (aug={aug_idx})")

                X_rows.append(feat)
                y_labels.append(label_idx)
                label_names.append(cname)
                groups.append(group_id)
                source_files.append(f"{wav_path}#aug{aug_idx}")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array(y_labels, dtype=np.int64)
    label_names_arr = np.array(label_names)
    groups_arr = np.array(groups)
    source_files_arr = np.array(source_files)
    classes_arr = np.array(CLASSES)

    assert X.shape[0] == y.shape[0] == len(label_names_arr) == len(groups_arr), (
        "Misaligned feature rows, labels, or group IDs!"
    )

    np.savez(
        OUT_NPZ,
        X=X,
        y=y,
        label_names=label_names_arr,
        groups=groups_arr,
        source_files=source_files_arr,
        classes=classes_arr,
        sample_rate=np.array([SAMPLE_RATE], dtype=np.int64),
        win_sec=np.array([WIN_SEC], dtype=np.float32),
    )

    print("\n=== Prepared Dataset & Class Mapping Summary ===")
    print(f"  Input matrix  X : {X.shape}  (dtype={X.dtype})")
    print(f"  Output vector y : {y.shape}  (dtype={y.dtype})")
    print(f"  Unique groups   : {len(np.unique(groups_arr))} (prevents train/test leakage across augmented variants)")

    for idx, cname in enumerate(CLASSES):
        n_samples = int(np.sum(y == idx))
        n_groups = len(np.unique(groups_arr[y == idx]))
        print(f"  Class {idx} ('{cname:5s}') : {n_samples:3d} samples across {n_groups:2d} source groups")

    # Show first mapped sample for each class so you can verify all three mappings directly
    print("\n=== Sample Input -> Class Mapping Verification ===")
    for idx, cname in enumerate(CLASSES):
        first_pos = int(np.flatnonzero(y == idx)[0])
        row_feat = X[first_pos]
        print(
            f"  Sample[{first_pos:03d}] -> y={y[first_pos]} ('{label_names_arr[first_pos]}') | "
            f"group={groups_arr[first_pos]} | feat_min={row_feat.min():.2f}, feat_max={row_feat.max():.2f}"
        )

    print(f"\nSaved prepared dataset to '{OUT_NPZ}'.")
    return X, y, groups_arr, classes_arr


if __name__ == "__main__":
    arrange_dataset(DATA_ROOT)