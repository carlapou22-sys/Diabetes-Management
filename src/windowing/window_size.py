import numpy as np
import pandas as pd

def downsample_class_zero_simple(X, y_multi, max_class0_ratio=3, seed=42, return_indices=False):
    """
    Simple downsampling:
      - keep all non-zero classes
      - reduce class 0 so that: n0 <= max_class0_ratio * n_nonzero

    Returns (X_ds, y_ds) by default.
    If return_indices=True, returns (X_ds, y_ds, keep_idx) to allow slicing meta.
    """
    idx0 = np.where(y_multi == 0)[0]
    idx_non0 = np.where(y_multi != 0)[0]

    # Nothing to balance
    if len(idx_non0) == 0 or len(idx0) == 0:
        if return_indices:
            keep_idx = np.arange(len(y_multi), dtype=int)
            return X, y_multi, keep_idx
        return X, y_multi

    # Max number of zeros to keep
    max_n0 = int(max_class0_ratio * len(idx_non0))

    rng = np.random.RandomState(seed)
    if len(idx0) > max_n0:
        keep0 = rng.choice(idx0, size=max_n0, replace=False)
    else:
        keep0 = idx0

    keep_idx = np.sort(np.concatenate([keep0, idx_non0]))
    X_ds = X[keep_idx]
    y_ds = y_multi[keep_idx]

    if return_indices:
        return X_ds, y_ds, keep_idx
    return X_ds, y_ds

def window_size_multilabel(segments, window_minutes=120, stride_minutes=45, sample_every=5,
                           downsample=False, max_class0_ratio=3):
    """
    Build sliding windows across multiple continuous CGM segments for MULTI-CLASS labels only.

    Returns
    -------
    X : np.ndarray               (n_windows, window_length)
    y_multi : np.ndarray[int32]  (n_windows,) values in {0,1,2,3,4}
    meta : pd.DataFrame          metadata incl. total_CHO per window
    """
    X_all, y_multi_all, meta_all = [], [], []

    # window/stride in samples
    W = int(window_minutes / sample_every)
    S = int(stride_minutes / sample_every)

    for seg_idx, df in enumerate(segments):
        # Require CGM column
        if "cbg" not in df.columns:
            continue

        # Ensure carbInput exists
        if "carbInput" not in df.columns:
            df = df.copy()
            df["carbInput"] = 0

        # Time axis in minutes (for meta)
        if isinstance(df.index, pd.DatetimeIndex):
            t = (df.index - df.index[0]).total_seconds() / 60.0
        else:
            t = np.arange(len(df)) * sample_every

        cbg = df["cbg"].astype("float32").values
        carbs = df["carbInput"].fillna(0).values

        if len(df) < W:
            # skip too-short segments
            continue

        starts = np.arange(0, len(df) - W + 1, S)

        for s in starts:
            e = s + W

            window_cbg = cbg[s:e]
            window_carbs = carbs[s:e]

            total_CHO = float(window_carbs.sum())
            label_multi = cho_to_meal_size_label(total_CHO)  # 0..4 (with classes 4&5 merged)

            X_all.append(window_cbg)
            y_multi_all.append(label_multi)

            meta_all.append({
                "segment": seg_idx,
                "start_idx": s,
                "end_idx": e,
                "start_time_min": t[s],
                "end_time_min": t[e - 1],
                "total_CHO": total_CHO,
                "label_multiclass": label_multi
            })

    # Stack
    X = np.asarray(X_all, dtype="float32")
    y_multi = np.asarray(y_multi_all, dtype="int32")
    meta = pd.DataFrame(meta_all)

    print(f" Created {len(X)} windows across {len(segments)} segments "
          f"(window={window_minutes}min, stride={stride_minutes}min).")

    # Optional simple downsampling of class 0 only (disabled by default to match earlier best runs)
    if downsample and len(X) and len(y_multi):
        X, y_multi, keep_idx = downsample_class_zero_simple(
            X, y_multi, max_class0_ratio=max_class0_ratio, return_indices=True
        )
        meta = meta.iloc[keep_idx].reset_index(drop=True)

    return X, y_multi, meta

def window_size(segments, window_minutes=120, stride_minutes=45, sample_every=5):
    """
    Build sliding windows across multiple continuous CGM segments.

    Parameters
    ----------
    segments : list[pd.DataFrame]
        Continuous, preprocessed DataFrames (each with 'cbg' and optionally 'carbInput')
    window_minutes : int
        Duration of each window in minutes (default 150 = 2.5 hours)
    stride_minutes : int
        Step size between consecutive windows (default 15)
    sample_every : int
        Sampling interval in minutes (default 5)

    Returns
    -------
    X : np.ndarray
        Array of CGM sequences, shape (n_windows, window_length)
    y : np.ndarray
        Binary labels (1 = meal in window, 0 = no meal)
    y_multiclass # 0…5 meal-size label
    meta: metadata incl. both labels
    """

    X_all, y_all, y_multi_all, y_reg, meta_all = [], [], [], [], []

    # number of samples per window and stride
    W = int(window_minutes / sample_every)
    S = int(stride_minutes / sample_every)

    for seg_idx, df in enumerate(segments):
        if "cbg" not in df.columns:
            continue

        # Fill missing carbInput column with zeros (if not present)
        if "carbInput" not in df.columns:
            df["carbInput"] = 0

        # Convert index to numeric time (minutes) if datetime
        if isinstance(df.index, pd.DatetimeIndex):
            t = (df.index - df.index[0]).total_seconds() / 60.0
        else:
            t = np.arange(len(df)) * sample_every

        cbg = df["cbg"].astype("float32").values
        carbs = df["carbInput"].fillna(0).values

        if len(df) < W:
            # Skip too-short segments
            continue

        starts = np.arange(0, len(df) - W + 1, S)

        for s in starts:
            e = s + W

            window_cbg = cbg[s:e]
            window_carbs = carbs[s:e]

            # --- total CHO in this window ---
            total_CHO = float(window_carbs.sum())

            # --- labels ---
            window_binary_label = 1 if total_CHO > 0 else 0
            window_multiclass_label = cho_to_meal_size_label(total_CHO)
            window_reg_label = total_CHO  # <---- THIS is the regression target

            # --- append outputs ---
            X_all.append(window_cbg)
            y_all.append(window_binary_label)
            y_multi_all.append(window_multiclass_label)
            y_reg.append(window_reg_label)  # <----- ADD THIS

            meta_all.append({
                "segment": seg_idx,
                "start_idx": s,
                "end_idx": e,
                "start_time_min": t[s],
                "end_time_min": t[e - 1],
                "label_binary": window_binary_label,
                "label_multiclass": window_multiclass_label,
                "total_CHO": total_CHO,
            })

    # --- Convert outputs ---
    X = np.asarray(X_all, dtype="float32")
    y = np.asarray(y_all, dtype="int32")  # binary labels stay untouched
    y_multi = np.asarray(y_multi_all, dtype="int32")
    meta = pd.DataFrame(meta_all)

    print(f" Created {len(X)} windows across {len(segments)} segments "
          f"(window={window_minutes}min, stride={stride_minutes}min).")

    # Note: no downsampling here — this function returns the full set for binary and multi.
    return X, y, y_multi, meta



def cho_to_meal_size_label(cho):
    # 0 = no CHO (treat tiny values as 0)
    if cho < 1:
        return 0
    # 1 = CHO < 10 g
    if cho < 10:
        return 1
    # 2 = 10–39 g
    if cho < 40:
        return 2
    # 3 = 40–69 g
    if cho < 70:
        return 3
    # 4 = ≥70 g (merged former classes 4 and 5)
    return 4