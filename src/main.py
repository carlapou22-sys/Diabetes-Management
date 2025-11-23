import numpy as np

from utils.load_data import load_patient_data
from utils.summary import build_summary, build_label_summary, plot_label_counts
from preprocessing.preprocess import preprocess_patient
from windowing.window_size import window_size, window_size_multilabel
from models.lstm_gridsearch import run_personalized_lstm_search
from models.lstm_generalized import run_global_lstm_search
from models.lstm_multilabel import run_personalized_lstm_multilabel
from models.lstm_regression import build_regression_lstm, evaluate_regression
from tensorflow.keras import callbacks


# ---------- PIPELINE 1: Single-patient pipeline ----------
def run_single_patient_pipeline(data, patient_key, mode="binary"):
    print(f"\n=== Running SINGLE-PATIENT pipeline for {patient_key} ({mode}) ===")

    df = data[patient_key]

    segments, summary = preprocess_patient(
        df,
        time_col="5minute_intervals_timestamp",
        cbg_col="cbg",
        normalize=True,
        long_gap_thresh=120,
    )

    if mode == "multi":
        # --- multi-label windows (0..4, with optional class-0 downsampling) ---
        X, y_multi, meta = window_size_multilabel(
            segments,
            window_minutes=120,
            stride_minutes=45,
            sample_every=5,
            downsample=True,
            max_class0_ratio=2.0,
        )
        # quick distro check
        try:
            import numpy as np  # ensure local scope
            print(f"Total: {len(X)} windows | class counts = {np.bincount(y_multi, minlength=5)}")
        except Exception:
            pass

        run_personalized_lstm_multilabel({patient_key: {"X": X, "y_multi": y_multi, "meta": meta}})

    elif mode == "binary":
        # --- binary windows (keep existing function) ---
        X, y_binary, y_multi_unused, meta = window_size(
            segments,
            window_minutes=120,
            stride_minutes=45,
            sample_every=5
        )
        print(f"Total: {len(X)} windows | positives={int(y_binary.sum())}")
        run_personalized_lstm_search({patient_key: {"X": X, "y": y_binary, "meta": meta}})

    elif mode == "regression":
        run_single_patient_regression(data, patient_key)

    else:
        raise ValueError("Invalid mode")


# ---------- PIPELINE 2: Loop over all patients (individual training) ----------
def run_all_patients_individual(data, mode="binary"):
    """
    Train a PERSONALIZED model for EVERY training patient.
    mode: "binary" or "multi"
    """
    print(f"\n=== Running ALL-PATIENTS pipeline ({mode}) ===")

    for patient_key, df in data.items():
        # Skip test sets
        if "training" not in patient_key.lower():
            continue

        try:
            print(f"\n=== Preprocessing {patient_key} ===")

            # --- Preprocessing ---
            segments, _ = preprocess_patient(
                df,
                time_col="5minute_intervals_timestamp",
                cbg_col="cbg",
                normalize=True,
                long_gap_thresh=120,
            )

            if mode == "multi":
                # --- Multi-label windowing ---
                X, y_multi, meta = window_size_multilabel(
                    segments,
                    window_minutes=120,
                    stride_minutes=45,
                    sample_every=5,
                    downsample=True,
                    max_class0_ratio=3,
                )
                model_input_data = {patient_key: {"X": X, "y_multi": y_multi, "meta": meta}}
                run_personalized_lstm_multilabel(model_input_data)

            elif mode == "binary":
                # --- Binary windowing ---
                X, y_binary, y_multi_unused, meta = window_size(
                    segments,
                    window_minutes=120,
                    stride_minutes=45,
                    sample_every=5
                )
                model_input_data = {patient_key: {"X": X, "y": y_binary, "meta": meta}}
                run_personalized_lstm_search(model_input_data)

            else:
                raise ValueError("mode must be 'binary' or 'multi'")

        except Exception as e:
            print(f"️ Skipping {patient_key} due to error: {e}")

# ---------- PIPELINE 3: Generalized model ----------
def run_generalized_model(data):
    print("\n=== Running GENERALIZED model across all patients ===")
    all_segments = []

    for patient_key, df in data.items():
        if "training" not in patient_key.lower():
            continue
        try:
            print(f"\n=== Preprocessing {patient_key} ===")
            segments, _ = preprocess_patient(
                df,
                time_col="5minute_intervals_timestamp",
                cbg_col="cbg",
                normalize=True,
                long_gap_thresh=120,
            )
            all_segments.extend(segments)
        except Exception as e:
            print(f" Skipping {patient_key}: {e}")

    if not all_segments:
        print(" No segments collected, cannot train global model.")
        return

    # Combine all patients into one big dataset
    X, y, meta = window_size(
        all_segments,
        window_minutes=120,
        stride_minutes=45,
        sample_every=5,
    )
    print(f" Combined dataset: {len(X)} windows ({y.sum()} positive).")

    run_global_lstm_search(X, y, meta)

# ---------- LABEL DISTRIBUTION ONLY ----------
def run_label_distribution_overview(data):
    """
    Preprocess + window ALL training patients,
    then build + plot label summary (0 vs 1) WITHOUT training any models.
    """
    print("\n=== Collecting label distributions for ALL training patients (no training) ===")

    model_input_data = {}

    for patient_key, df in data.items():
        if "training" not in patient_key.lower():
            continue  # skip test sets

        try:
            print(f"\n=== Preprocessing {patient_key} ===")

            # --- Preprocessing ---
            segments, summary = preprocess_patient(
                df,
                time_col="5minute_intervals_timestamp",
                cbg_col="cbg",
                normalize=True,
                long_gap_thresh=120
            )

            # --- Windowing ---
            X, y, meta = window_size(
                segments,
                window_minutes=120,
                stride_minutes=45,
                sample_every=5
            )

            model_input_data[patient_key] = {"X": X, "y": y, "meta": meta}

        except Exception as e:
            print(f" Skipping {patient_key} due to error: {e}")

    if not model_input_data:
        print("️ No patients were processed, cannot build label summary.")
        return

    # ===== Build table + plot =====
    label_summary = build_label_summary(model_input_data)
    label_summary.to_csv("results/lstm_gridsearch/label_summary_only.csv", index=False)

    print("\nLabel summary (per patient):")
    print(label_summary)

    plot_label_counts(label_summary)

def run_single_patient_regression(data, pid):
    print(f"\n=== Running SINGLE-PATIENT regression for {pid} ===")

    df = data[pid]

    # Preprocess and windowize
    segments, _ = preprocess_patient(df)
    X, y_bin, y_multi, y_reg, meta = window_size(
        segments,
        window_minutes=120,
        stride_minutes=45,
        sample_every=5
    )

    if len(X) == 0:
        print("No data, skipping.")
        return

    # reshape (samples, timesteps, features)
    X = X[..., np.newaxis]

    n = len(X)
    split_idx = int(0.8 * n)

    X_tr, X_va = X[:split_idx], X[split_idx:]
    y_tr, y_va = y_reg[:split_idx], y_reg[split_idx:]

    model = build_regression_lstm(
        timesteps=X.shape[1],
        features=1,
        lr=1e-3
    )

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=40,
        batch_size=128,
        callbacks=[es],
        verbose=1
    )

    # predictions
    y_pred = model.predict(X_va).ravel()

    results = evaluate_regression(y_va, y_pred)
    print("Regression results:", results)
    print("\n📊 Example predictions (first 20 windows):")
    for t, p in list(zip(y_va[:20], y_pred[:20])):
        print(f"true={t:.1f} g, pred={p:.1f} g")

# ---------- MAIN ----------
if __name__ == "__main__":
    base_dir = "/Users/isabellemueller/BME unibern/Diabetes Management/Ohio Data"
    data = load_patient_data(base_dir)
    summary_df = build_summary(data)

    # Choose which pipeline to run:
    mode = "single_multi"

    if mode == "single_binary":
        run_single_patient_pipeline(data, "588-ws-training_processed", mode="binary")

    elif mode == "single_multi":
        run_single_patient_pipeline(data, "588-ws-training_processed", mode="multi")

    elif mode == "single_regression":
        run_single_patient_pipeline(data, "588-ws-training_processed", mode="regression")

    elif mode == "all_binary":
        run_all_patients_individual(data, mode="binary")

    elif mode == "all_multi":
        run_all_patients_individual(data, mode="multi")

    #elif mode == "all_regression":
        # run_all_patients_regression(data)
