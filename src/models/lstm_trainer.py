import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def train_lstm(X, y, seed=42, verbose=1):
    """
    Train an LSTM model on pre-windowed, normalized CGM data.
    X: numpy array of shape (samples, timesteps)
    y: binary labels
    """
    # --- Reproducibility
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # --- Chronological split
    n = len(X)
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # --- Reshape for LSTM
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    # --- Class weights
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    class_weight = {0: 1.0, 1: 1.0} if pos == 0 or neg == 0 else {
        0: len(y_train) / (2.0 * max(neg, 1)),
        1: len(y_train) / (2.0 * max(pos, 1)),
    }

    # --- Model
    timesteps = X_train.shape[1]
    model = models.Sequential([
        layers.Input(shape=(timesteps, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # --- Train
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=128,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=verbose
    )

    # --- Evaluation
    y_val_prob = model.predict(X_val, verbose=0).ravel()
    thresholds = np.linspace(0.05, 0.95, 37)
    f1_scores = []

    for t in thresholds:
        pred = (y_val_prob >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_val, pred, average="binary", zero_division=0)
        f1_scores.append(f1)

    best_threshold = float(thresholds[np.argmax(f1_scores)])
    print(f"✅ Best threshold: {best_threshold:.2f} (F1 = {max(f1_scores):.3f})")

    # --- Plot curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist.history["accuracy"], label="train"); ax[1].plot(hist.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model, hist, best_threshold