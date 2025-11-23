"""
Main pipeline for CGM meal detection project.
"""

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.load_data import load_patient_data
from utils.summary import build_summary
from utils.plotting import plot_meal_counts
from preprocessing.preprocess import preprocess_patient
from windowing.window_size import window_size
from preprocessing.preprocess_plotting import plot_preprocessed_segments
from windowing.window_plotting import visualize_windows  # adjust path if needed
#from models.lstm_trainer import train_lstm
from models.lstm_gridsearch import run_personalized_lstm_search


if __name__ == "__main__":
    # Step 1 — Load data
    base_dir = "/Users/isabellemueller/BME unibern/Diabetes Management/Ohio Data"
    data = load_patient_data(base_dir)

   # Step 2 — Summarize
    summary_df = build_summary(data)
    #print(summary_df.head())

    # Step 3 — Plot
    #plot_meal_counts(summary_df)

    # Step 4 — Preprocessing
    patient_key = "588-ws-training_processed" # <----- change patient name here
    df = data[patient_key]
    time_col = "5minute_intervals_timestamp"

    # Run preprocessing
    segments, summary = preprocess_patient(
        df,
        time_col=time_col,
        cbg_col="cbg",
        normalize=True,
        long_gap_thresh=120  # adjust if needed
    )
    for i, seg in enumerate(segments):
        nans = seg["cbg"].isna().sum()
        if nans > 0:
            print(f"Segment {i + 1} still has {nans} NaN values")
    segments, summary = preprocess_patient(df, time_col="5minute_intervals_timestamp", cbg_col="cbg")
    #plot_preprocessed_segments(df, segments)
    '''
    # Step 5 — Windowing
    X, y, meta = window_size(segments, window_minutes=120, stride_minutes=45, sample_every=5)
    print(f" Total windows: {len(X)}, Positive (meal) windows: {y.sum()}")

    Visualize windows
    visualize_windows(segments, window_minutes=120, stride_minutes=45, max_windows=10)
    '''

    # Step 5 — Windowing
    X, y, meta = window_size(segments, window_minutes=120, stride_minutes=45, sample_every=5)
    print(f" Total windows: {len(X)}, Positive (meal) windows: {y.sum()}")
    # Visualize windows (optional for visualizing)
    # visualize_windows(segments, window_minutes=120, stride_minutes=45, max_windows=10)

    # Step 6 Run LSTM grid search
    # Prepare input dictionary for model
    model_input_data = {
        patient_key: {"X": X, "y": y, "meta": meta}
    }
    #run
    run_personalized_lstm_search(model_input_data)

    '''
    # For looping through all patients, apply this:
    Step 4 — Preprocess all patients
    for key, df in patient_data.items():
        print(f"\n=== Processing {key} ===")
        segments, summary = preprocess_patient(df)
        print(f"  → Found {len(segments)} continuous CGM segments")

        # Step 5 — Apply windowing for each segment
        for i, seg in enumerate(segments):
            X, y, meta = window_size(seg)
            print(f"    Segment {i+1}: {len(X)} windows, {y.sum()} meals")
    '''