from utils.load_data import load_patient_data
from utils.summary import build_summary
from preprocessing.preprocess import preprocess_patient
from windowing.window_size import window_size
from models.lstm_gridsearch import run_personalized_lstm_search
from models.lstm_generalized import run_global_lstm_search  # NEW

from utils.summary import build_label_summary, plot_label_counts  # moved up
from models.lstm_generalized import run_global_lstm_search


# ---------- PIPELINE 1: Single-patient pipeline ----------
def run_single_patient_pipeline(data, patient_key):
    print(f"\n=== Running pipeline for patient {patient_key} ===")
    df = data[patient_key]
    segments, summary = preprocess_patient(
        df, time_col="5minute_intervals_timestamp", cbg_col="cbg",
        normalize=True, long_gap_thresh=120
    )
    X, y, meta = window_size(segments, window_minutes=120, stride_minutes=45, sample_every=5)
    print(f" Total windows: {len(X)}, Positive (meal) windows: {y.sum()}")
    model_input_data = {patient_key: {"X": X, "y": y, "meta": meta}}
    run_personalized_lstm_search(model_input_data)


# ---------- PIPELINE 2: Loop over all patients (individual training) ----------
def run_all_patients_individual(data):
    print("\n=== Running pipeline for ALL patients (individual models) ===")

    model_input_data = {}

    for patient_key in data.keys():
        if "training" not in patient_key.lower():
            continue  # skip test sets

        try:
            print(f"\n=== Preprocessing {patient_key} ===")
            df = data[patient_key]

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

            # store for later
            model_input_data[patient_key] = {"X": X, "y": y, "meta": meta}

            # --- Train personalized LSTM ---
            run_personalized_lstm_search({patient_key: model_input_data[patient_key]})

        except Exception as e:
            print(f"⚠️ Skipping {patient_key} due to error: {e}")

    # ===== After training all patients: Label summary =====
    if model_input_data:
        label_summary = build_label_summary(model_input_data)
        label_summary.to_csv("results/lstm_gridsearch/label_summary.csv", index=False)

        print("\nSaved label summary table!")
        print(label_summary)

        plot_label_counts(label_summary)


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
            print(f"⚠️ Skipping {patient_key}: {e}")

    if not all_segments:
        print("⚠️ No segments collected, cannot train global model.")
        return

    # Combine all patients into one big dataset
    X, y, meta = window_size(
        all_segments,
        window_minutes=120,
        stride_minutes=45,
        sample_every=5,
    )
    print(f"🧩 Combined dataset: {len(X)} windows ({y.sum()} positive).")

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
            print(f"⚠️ Skipping {patient_key} due to error: {e}")

    if not model_input_data:
        print("⚠️ No patients were processed, cannot build label summary.")
        return

    # ===== Build table + plot =====
    label_summary = build_label_summary(model_input_data)
    label_summary.to_csv("results/lstm_gridsearch/label_summary_only.csv", index=False)

    print("\nLabel summary (per patient):")
    print(label_summary)

    plot_label_counts(label_summary)


# ---------- MAIN ----------
if __name__ == "__main__":
    base_dir = r"C:\Users\carla\Documents\Master\Third Semester\Diabetes\Ohio Data\Ohio Data"
    data = load_patient_data(base_dir)
    summary_df = build_summary(data)

    # Choose which pipeline to run:
    mode = "generalized"  # options: "single", "all", "generalized", "labels_only"

    if mode == "single":
        run_single_patient_pipeline(data, "588-ws-training_processed")  # <--- select patient

    elif mode == "all":
        run_all_patients_individual(data)

    elif mode == "generalized":
        run_generalized_model(data)

    elif mode == "labels_only":
        run_label_distribution_overview(data)
