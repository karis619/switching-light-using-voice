import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preparation import SAMPLE_RATE, WIN_SEC, extract_features

NPZ_PATH = "dataset_prepared.npz"
MODEL_OUT = "on_off_model.joblib"
SEED = 42


def split_train_test(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_classes: int):
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    for train_idx, test_idx in sgkf.split(X, y, groups):
        if len(np.unique(y[test_idx])) == n_classes:
            return train_idx, test_idx

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    return next(gss.split(X, y, groups))


def main() -> None:
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    groups = data["groups"]
    classes = [str(c) for c in data["classes"]]
    sr = int(data["sample_rate"][0]) if "sample_rate" in data.files else SAMPLE_RATE
    win_sec = float(data["win_sec"][0]) if "win_sec" in data.files else WIN_SEC

    print(f"Loaded '{NPZ_PATH}': X={X.shape}, y={y.shape}, classes={classes}")
    print(f"Unique recording groups: {len(np.unique(groups))}")

    train_idx, test_idx = split_train_test(X, y, groups, len(classes))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(
        f"Split -> Train: {len(y_train)} samples ({len(np.unique(groups[train_idx]))} groups) | "
        f"Test: {len(y_test)} samples ({len(np.unique(groups[test_idx]))} groups)"
    )

    counts = np.bincount(y_train, minlength=len(classes))
    total = len(y_train)
    class_weight_map = {
        idx: total / (len(classes) * max(1, int(counts[idx])))
        for idx in range(len(classes))
    }
    train_weights = np.array([class_weight_map[lbl] for lbl in y_train], dtype=np.float32)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            max_depth=6,
            min_samples_leaf=6,
            random_state=SEED,
        )),
    ])

    pipe.fit(X_train, y_train, clf__sample_weight=train_weights)

    y_pred = pipe.predict(X_test)
    print("\n--- Evaluation on Unseen Recording Groups ---")
    print("Rows = True [other, on, off], Cols = Predicted")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=classes,
            zero_division=0,
        )
    )

    # Fit final deployment model across all prepared samples with balanced weights
    full_weights = np.array([class_weight_map[lbl] for lbl in y], dtype=np.float32)
    pipe.fit(X, y, clf__sample_weight=full_weights)

    joblib.dump(
        {
            "pipeline": pipe,
            "classes": classes,
            "sample_rate": sr,
            "win_sec": win_sec,
            "feature_dim": int(X.shape[1]),
        },
        MODEL_OUT,
    )
    print(f"\nSaved trained model to '{MODEL_OUT}'.")


if __name__ == "__main__":
    main()