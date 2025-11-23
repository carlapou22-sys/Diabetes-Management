import os
import random
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
)
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
SEED = 42
MAX_COMBOS = 8                    # keep smaller, multi-class is heavier
VAL_SPLIT = 0.2                   # 80% train / 20% val
OUT_DIR = "results/lstm_multilabel"

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- splitting / weighting guards ---
STRATIFIED_SPLIT_IF_MISSING = True   # if a class is missing in train, switch to stratified split
MIN_TRAIN_PER_CLASS = 3              # ensure at least a few samples of each class in train (when available)
WEIGHT_CLIP = (0.25, 8.0)            # clip per-class weights to a reasonable range
# ------------------------------------------------

def tolerant_accuracy(y_true, y_pred, tol=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))

def sparse_categorical_focal_loss(gamma=2.0, alpha=None, n_classes=None):
    """
    Focal loss for sparse integer targets (classes 0..K-1).
    alpha: optional per-class weights (length K).
    """
    alpha_const = None
    if alpha is not None:
        alpha_const = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        C = n_classes if n_classes is not None else tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=C)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        ce = -tf.math.log(p_t)
        mod = tf.pow(1.0 - p_t, gamma)
        if alpha_const is not None:
            a_t = tf.gather(alpha_const, y_true)
            loss = a_t * mod * ce
        else:
            loss = mod * ce
        return tf.reduce_mean(loss)
    return loss_fn

def build_multilabel_model(
    timesteps,
    features,
    n_lstm_layers=1,
    lr=1e-3,
    batchnorm=True,
    dense1_units=64,
    dense2_units=32,
    lstm_dropout=0.2,
    lstm_recurrent_dropout=0.0,
    dense_dropout=0.3,
    n_classes: int = 5,
):
    """Multi-class LSTM classifier (5 classes, 0–4) with softmax head."""
    inputs = layers.Input(shape=(timesteps, features))

    d1 = float(np.clip(lstm_dropout, 0.0, 0.5))
    d2 = float(np.clip(lstm_dropout + 0.1, 0.0, 0.5))
    d3 = float(np.clip(lstm_dropout + 0.2, 0.0, 0.5))

    x = layers.LSTM(
        128, return_sequences=(n_lstm_layers > 1),
        dropout=d1, recurrent_dropout=lstm_recurrent_dropout
    )(inputs)

    if n_lstm_layers >= 2:
        x = layers.LSTM(
            64, return_sequences=(n_lstm_layers > 2),
            dropout=d2, recurrent_dropout=lstm_recurrent_dropout
        )(x)

    if n_lstm_layers >= 3:
        x = layers.LSTM(
            32, return_sequences=False,
            dropout=d3, recurrent_dropout=lstm_recurrent_dropout
        )(x)

    if dense1_units and dense1_units > 0:
        x = layers.Dense(dense1_units)(x)
        if batchnorm: x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dense_dropout)(x)

    if dense2_units and dense2_units > 0:
        x = layers.Dense(dense2_units)(x)
        if batchnorm: x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def plot_curves(hist, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["loss"], label="train")
    ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    if "accuracy" in hist.history:
        ax[1].plot(hist.history["accuracy"], label="train")
    if "val_accuracy" in hist.history:
        ax[1].plot(hist.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close(fig)

def plot_multilabel_diagnostics(y_true, y_pred, n_classes, save_prefix):
    """Save confusion matrix + per-class PRF bar chart."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion.png", dpi=150); plt.close(fig)
    pd.DataFrame(cm, index=range(n_classes), columns=range(n_classes)).to_csv(
        f"{save_prefix}_confusion.csv"
    )

    prec_k, rec_k, f1_k, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=np.arange(n_classes)
    )
    idx = np.arange(n_classes); width = 0.27
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    ax2.bar(idx - width, prec_k, width, label="Precision")
    ax2.bar(idx,         rec_k,  width, label="Recall")
    ax2.bar(idx + width, f1_k,   width, label="F1")
    ax2.set_ylim(0, 1.0); ax2.set_xticks(idx); ax2.set_title("Per-class metrics")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(f"{save_prefix}_perclass.png", dpi=150); plt.close(fig2)

def class_weights_from_multilabel(y, n_classes: int = 5, zero_handling: str = "ignore"):
    """Inverse-frequency weights from *true* train counts (clipped)."""
    true_counts = np.bincount(y, minlength=n_classes).astype(np.int64)
    n_samples = int(len(y))
    weights = {}
    for k, c in enumerate(true_counts):
        if c == 0:
            if zero_handling == "ignore":
                weights[k] = 0.0
            else:
                w = float(n_samples) / (n_classes * 1.0)
                weights[k] = float(np.clip(w, WEIGHT_CLIP[0], WEIGHT_CLIP[1]))
        else:
            w = float(n_samples) / (n_classes * float(c))
            weights[k] = float(np.clip(w, WEIGHT_CLIP[0], WEIGHT_CLIP[1]))
    return weights, true_counts

def stratified_time_preserving_split(X, y, val_frac=VAL_SPLIT, min_train=MIN_TRAIN_PER_CLASS):
    """Stratified split preserving order within each class, ensuring ≥min_train per class."""
    y = np.asarray(y); idx = np.arange(len(y))
    train_idx_parts, val_idx_parts = [], []
    for k in np.unique(y):
        k_idx = idx[y == k]
        cutoff = max(min_train, int((1.0 - val_frac) * len(k_idx)))
        cutoff = min(cutoff, len(k_idx))
        train_idx_parts.append(k_idx[:cutoff])
        val_idx_parts.append(k_idx[cutoff:])
    train_idx = np.sort(np.concatenate(train_idx_parts)) if train_idx_parts else np.array([], dtype=int)
    val_idx   = np.sort(np.concatenate(val_idx_parts))   if val_idx_parts   else np.array([], dtype=int)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def downsample_class0(X, y, max_ratio=2.0, rng=None):
    """Cap class-0 count to (max_ratio × non-zero count) in TRAIN set."""
    if rng is None: rng = np.random
    y = np.asarray(y)
    idx0 = np.where(y == 0)[0]; idxN = np.where(y != 0)[0]
    if len(idxN) == 0: return X, y
    keep0 = min(len(idx0), int(max_ratio * len(idxN)))
    keep0_idx = rng.choice(idx0, size=keep0, replace=False) if keep0 < len(idx0) else idx0
    keep = np.sort(np.concatenate([keep0_idx, idxN]))
    return X[keep], y[keep]

def run_personalized_lstm_multilabel(model_input_data):
    """
    Personalized multi-class grid search: one 5-class LSTM per patient.

    Expects model_input_data[pid] = {"X": X, "y_multi": y_multi, "meta": meta_df}
    """
    print("=== Starting personalized MULTI-LABEL LSTM grid search ===")
    results = []; start_all = datetime.now()

    for pid, data_dict in model_input_data.items():
        print(f"\n--- Processing {pid} (multi-label) ---")
        X_all = data_dict["X"]; y_all = data_dict["y_multi"]
        if X_all is None or y_all is None:
            print(f"Skipping {pid}: missing X or y_multi."); continue
        if len(X_all) < 80:
            print(f"Skipping {pid}: insufficient windows ({len(X_all)})."); continue
        if len(np.unique(y_all)) < 2:
            print(f"Skipping {pid}: only one class present in labels {np.unique(y_all)}."); continue

        if X_all.ndim == 2: X_all = X_all[..., np.newaxis]

        # chronological split, then guard for missing train classes
        n = len(X_all); split_idx = int(n * (1 - VAL_SPLIT))
        X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
        X_va, y_va = X_all[split_idx:],  y_all[split_idx:]

        overall_counts = np.bincount(y_all, minlength=5)
        train_counts   = np.bincount(y_tr,  minlength=5)
        missing_in_train = np.where((overall_counts > 0) & (train_counts == 0))[0]
        if STRATIFIED_SPLIT_IF_MISSING and len(missing_in_train) > 0:
            print(f"[SPLIT] Class(es) {missing_in_train.tolist()} absent in train; using stratified split.")
            X_tr, X_va, y_tr, y_va = stratified_time_preserving_split(
                X_all, y_all, val_frac=VAL_SPLIT, min_train=MIN_TRAIN_PER_CLASS
            )
            train_counts = np.bincount(y_tr, minlength=5)
            val_counts   = np.bincount(y_va, minlength=5)
            print(f"[SPLIT] New train counts: {train_counts}")
            print(f"[SPLIT] New val counts:   {val_counts}")

        # downsample class 0 on TRAIN only
        X_tr, y_tr = downsample_class0(X_tr, y_tr, max_ratio=2.0)
        print(f"[DS] After downsampling class 0: {np.bincount(y_tr, minlength=5)}")

        # weights from true train counts
        class_weights, counts = class_weights_from_multilabel(y_tr, n_classes=5, zero_handling="ignore")
        print(f"Class counts for {pid} (train): {counts}")
        print(f"Class weights: {class_weights}")

        timesteps, features = X_tr.shape[1], X_tr.shape[2]
        alpha_vec = np.ones(5, dtype=np.float32)
        for k, w in class_weights.items(): alpha_vec[k] = float(w)
        alpha_vec = alpha_vec / alpha_vec.sum()

        # small randomized grid
        combos = list(itertools.product(
            [1, 2], [1e-3, 3e-4], [128, 256], [20, 35],
            [True, False], [32, 64], [0, 32], [0.1, 0.2], [0.0, 0.05]
        ))
        random.shuffle(combos); combos = combos[:MAX_COMBOS]

        best_f1, best_info = -1.0, None
        for (n_layers, lr, bs, epochs_budget, bn, d1_units, d2_units, lstm_do, lstm_rdo) in combos:
            tf.keras.backend.clear_session()
            model = build_multilabel_model(
                timesteps, features, n_lstm_layers=n_layers, lr=lr,
                batchnorm=bn, dense1_units=d1_units, dense2_units=d2_units,
                lstm_dropout=lstm_do, lstm_recurrent_dropout=lstm_rdo, n_classes=5
            )
            # focal loss with alpha weights
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=sparse_categorical_focal_loss(gamma=2.0, alpha=alpha_vec, n_classes=5),
                metrics=["accuracy"],
            )
            es = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
            rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0)

            hist = model.fit(
                X_tr, y_tr.astype("int32"),
                validation_data=(X_va, y_va.astype("int32")),
                epochs=epochs_budget, batch_size=bs,
                callbacks=[es, rlrop], verbose=0
            )

            # evaluate (argmax)
            y_prob = model.predict(X_va, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)

            prec, rec, f1, _ = precision_recall_fscore_support(
                y_va, y_pred, average="macro", zero_division=0
            )
            mae = mean_absolute_error(y_va, y_pred)
            mse = mean_squared_error(y_va, y_pred)
            tol_acc = tolerant_accuracy(y_va, y_pred, tol=1)

            if f1 > best_f1:
                best_f1 = f1
                cm = confusion_matrix(y_va, y_pred, labels=np.arange(5))
                off_by = np.abs(y_va - y_pred)
                exact_acc = float((off_by == 0).mean())
                pm1_acc   = float((off_by <= 1).mean())
                qwk       = cohen_kappa_score(y_va, y_pred, weights="quadratic")

                best_info = {
                    "pid": pid, "n_lstm_layers": n_layers, "lr": lr, "batch_size": bs,
                    "epochs_budget": epochs_budget, "batchnorm": bn,
                    "dense1_units": d1_units, "dense2_units": d2_units,
                    "lstm_dropout": lstm_do, "lstm_recurrent_dropout": lstm_rdo,
                    "macro_F1": f1, "macro_precision": prec, "macro_recall": rec,
                    "MAE_labels": mae, "MSE_labels": mse, "tolerant_acc_±1": tol_acc,
                    "exact_acc": exact_acc, "pm1_acc": pm1_acc, "QWK": qwk,
                }

                plot_curves(hist, os.path.join(OUT_DIR, f"{pid}_multilabel_curves.png"))
                plot_multilabel_diagnostics(y_va, y_pred, n_classes=5,
                                            save_prefix=os.path.join(OUT_DIR, f"{pid}_multilabel"))

        if best_info:
            results.append(best_info)
            print(
                f"→ Best macro-F1 for {pid}: {best_info['macro_F1']:.3f} "
                f"(P={best_info['macro_precision']:.3f}, "
                f"R={best_info['macro_recall']:.3f}, "
                f"MAE={best_info['MAE_labels']:.3f})"
            )
        else:
            print(f"No valid multi-label model found for {pid}.")

    if results:
        df_summary = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_summary_path = os.path.join(OUT_DIR, f"summary_multilabel_{ts}.csv")
        df_summary.to_csv(run_summary_path, index=False)
        print(f"\n Saved MULTI-LABEL RUN summary: {run_summary_path}")

        global_summary_path = os.path.join(OUT_DIR, "summary_multilabel_all_runs.csv")
        if os.path.exists(global_summary_path):
            old_df = pd.read_csv(global_summary_path)
            combined = pd.concat([old_df, df_summary], ignore_index=True)
        else:
            combined = df_summary.copy()
        combined.to_csv(global_summary_path, index=False)
        print(f" Updated GLOBAL MULTI-LABEL summary: {global_summary_path}")
    else:
        print("\n️ No multi-label results to save.")