import os, random, itertools, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from datetime import datetime
import matplotlib.pyplot as plt

from preprocessing.preprocess import preprocess_patient
from windowing.window_size import window_size


def build_model(timesteps, features, n_lstm_layers=3, lr=1e-3, batchnorm=True):
    """LSTM architecture identical to her version."""
    layers_list = [layers.Input(shape=(timesteps, features))]
    layers_list += [layers.LSTM(128, return_sequences=(n_lstm_layers > 1)), layers.Dropout(0.2)]
    if n_lstm_layers >= 2:
        layers_list += [layers.LSTM(64, return_sequences=(n_lstm_layers > 2)), layers.Dropout(0.3)]
    if n_lstm_layers >= 3:
        layers_list += [layers.LSTM(32, return_sequences=False), layers.Dropout(0.4)]
    layers_list += [layers.Dense(32)]
    if batchnorm:
        layers_list += [layers.BatchNormalization()]
    layers_list += [layers.Activation("relu"), layers.Dense(1, activation="sigmoid")]

    model = models.Sequential(layers_list)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def tune_threshold_by_f1(y_true, y_prob, grid=np.linspace(0.05, 0.95, 37)):
    best = {"t": 0.5, "f1": 0}
    for t in grid:
        pred = (y_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best.update({"t": float(t), "f1": f1})
    return best


def plot_curves(hist, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)


def run_personalized_lstm_search(patient_data, out_dir="./results_lstm_search", seed=42):
    """Reimplementation of her full search loop using your preprocess + window_size."""
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    results = []
    start_all = datetime.now()

    for pid, df in patient_data.items():
        print(f"\n--- Processing {pid} ---")

        # Step 1 — Preprocess and windowize
        segments, _ = preprocess_patient(df)
        if len(segments) == 0:
            print("Skipping (no valid segments).")
            continue
        combined = pd.concat(segments)
        X_all, y_all, meta_all = window_size([combined], window_minutes=150, stride_minutes=15)

        if len(X_all) < 80 or y_all.sum() == 0:
            print(f"Skipping {pid}: insufficient windows.")
            continue

        # Chronological split
        n = len(X_all)
        split_idx = int(n * 0.8)
        X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
        X_va, y_va = X_all[split_idx:], y_all[split_idx:]

        X_tr = X_tr[..., np.newaxis]
        X_va = X_va[..., np.newaxis]

        cw = {0: len(y_tr) / (2 * max(len(y_tr) - y_tr.sum(), 1)),
              1: len(y_tr) / (2 * max(y_tr.sum(), 1))}

        # Hyperparameter grid (same as hers)
        combos = list(itertools.product([2, 3], [1e-3, 5e-4, 3e-4], [128, 256], [20, 35], [True, False]))
        random.shuffle(combos)
        combos = combos[:10]

        best_f1, best_info = -1, None

        for n_layers, lr, bs, epochs_budget, bn in combos:
            tf.keras.backend.clear_session()
            model = build_model(X_tr.shape[1], X_tr.shape[2], n_lstm_layers=n_layers, lr=lr, batchnorm=bn)
            es = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0)
            rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0)
            hist = model.fit(
                X_tr, y_tr.astype(int),
                validation_data=(X_va, y_va.astype(int)),
                epochs=epochs_budget,
                batch_size=bs,
                class_weight=cw,
                callbacks=[es, rlrop],
                verbose=0
            )

            y_prob = model.predict(X_va, verbose=0).ravel()
            best = tune_threshold_by_f1(y_va, y_prob)
            if best["f1"] > best_f1:
                best_f1 = best["f1"]
                best_info = {"pid": pid, "layers": n_layers, "lr": lr, "bs": bs,
                             "epochs": epochs_budget, "batchnorm": bn, "best_t": best["t"], "best_f1": best["f1"]}
                plot_curves(hist, os.path.join(out_dir, f"{pid}_curves.png"))

        if best_info:
            results.append(best_info)
            print(f"→ Best F1 for {pid}: {best_info['best_f1']:.3f} @ t={best_info['best_t']:.2f}")

    df_summary = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\n✅ Saved summary to {csv_path}")
    print(f"Total runtime: {datetime.now() - start_all}")