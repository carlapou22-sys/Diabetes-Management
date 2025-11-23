import pandas as pd
import numpy as np

def normalize_cgm(df, col="cbg"):
    """Normalize CGM values to [0,1] range per segment."""
    x_min, x_max = df[col].min(), df[col].max()
    if pd.notna(x_min) and pd.notna(x_max) and x_max > x_min:
        df[col] = (df[col] - x_min) / (x_max - x_min)
    else:
        df[col] = 0.0
    return df, x_min, x_max


def find_missing_segments(df, time_col="5minute_intervals_timestamp", cbg_col="cbg",
                          short_gap_thresh=15, medium_gap_thresh=60, long_gap_thresh=120):
    """Detect contiguous NaN stretches in CGM signal."""
    df = df.copy()

    # Ensure datetime index for correct interpolation timing
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col] * 5, unit="m", origin="unix", errors="coerce")

    df = df.sort_values(by=time_col).set_index(time_col)

    missing_segments = []
    in_gap, gap_start = False, None

    for t, val in df[cbg_col].items():
        if pd.isna(val) and not in_gap:
            gap_start = t
            in_gap = True
        elif not pd.isna(val) and in_gap:
            gap_end = t
            L = (gap_end - gap_start).total_seconds() / 60.0
            missing_segments.append((gap_start, gap_end, L))
            in_gap = False

    if in_gap:
        gap_end = df.index[-1]
        L = (gap_end - gap_start).total_seconds() / 60.0
        missing_segments.append((gap_start, gap_end, L))

    return missing_segments, df


def handle_missing_data(df, missing_segments, cbg_col="cbg",
                        short_gap_thresh=15, medium_gap_thresh=60, long_gap_thresh=120):
    """Split into continuous segments and interpolate small and medium gaps."""
    df = df.copy()
    df = df.sort_index()

    segments = []
    segment_start = df.index.min()

    for start, end, L in missing_segments:
        if L > long_gap_thresh:
            # Long gap → cut segment before it
            segment = df.loc[segment_start:start].copy()
            if not segment.empty:
                segments.append(segment)
            segment_start = end
        else:
            # For short and medium gaps, just mark NaNs for interpolation later
            df.loc[start:end, cbg_col] = np.nan

    # Add last segment after final gap
    tail = df.loc[segment_start:].copy()
    if not tail.empty:
        segments.append(tail)

    # Interpolate short gaps with linear method, medium gaps with nearest method
    interpolated_segments = []
    for segment in segments:
        segment = segment.copy()
        # Identify gaps within this segment
        gaps = []
        in_gap = False
        gap_start = None
        for t, val in segment[cbg_col].items():
            if pd.isna(val) and not in_gap:
                gap_start = t
                in_gap = True
            elif not pd.isna(val) and in_gap:
                gap_end = t
                L = (gap_end - gap_start).total_seconds() / 60.0
                gaps.append((gap_start, gap_end, L))
                in_gap = False
        if in_gap:
            gap_end = segment.index[-1]
            L = (gap_end - gap_start).total_seconds() / 60.0
            gaps.append((gap_start, gap_end, L))

        # --- Interpolate small and medium gaps properly ---
        for g_start, g_end, length in gaps:
            if length <= short_gap_thresh:
                # Use time-based linear interpolation over slightly extended window
                idx_start = segment.index.get_loc(g_start)
                idx_end = segment.index.get_loc(g_end)
                # extend window by 1 on both sides if possible
                left = max(0, idx_start - 1)
                right = min(len(segment) - 1, idx_end + 1)
                segment.iloc[left:right + 1, segment.columns.get_loc(cbg_col)] = (
                    segment.iloc[left:right + 1][cbg_col]
                    .interpolate(method="time", limit_direction="both")
                )

            elif short_gap_thresh < length <= medium_gap_thresh:
                # Use nearest interpolation for medium gaps
                idx_start = segment.index.get_loc(g_start)
                idx_end = segment.index.get_loc(g_end)
                left = max(0, idx_start - 1)
                right = min(len(segment) - 1, idx_end + 1)
                segment.iloc[left:right + 1, segment.columns.get_loc(cbg_col)] = (
                    segment.iloc[left:right + 1][cbg_col]
                    .interpolate(method="nearest", limit_direction="both")
                )

        # --- Final cleanup for remaining stray NaNs ---
        segment[cbg_col] = segment[cbg_col].interpolate(limit_direction="both")
        segment[cbg_col] = segment[cbg_col].ffill().bfill()
        '''
        # Interpolate short gaps linearly
        for g_start, g_end, length in gaps:
            if length <= short_gap_thresh:
                segment.loc[g_start:g_end, cbg_col] = segment.loc[g_start:g_end, cbg_col].interpolate(
                    method="time", limit_direction="both"
                )
        # Interpolate medium gaps with nearest
        for g_start, g_end, length in gaps:
            if short_gap_thresh < length <= medium_gap_thresh:
                segment.loc[g_start:g_end, cbg_col] = segment.loc[g_start:g_end, cbg_col].interpolate(
                    method="nearest", limit_direction="both"
                )
        '''
        interpolated_segments.append(segment)

    return interpolated_segments


def preprocess_patient(df, cbg_col="cbg", time_col="5minute_intervals_timestamp",
                       normalize=True, short_gap_thresh=15, medium_gap_thresh=60, long_gap_thresh=120):
    """Return list of cleaned, normalized CGM segments."""
    df = df.copy()
    if "missing_cbg" in df.columns:
        df.loc[df["missing_cbg"] > 0.5, cbg_col] = np.nan

    # Detect and split by NaN-based gaps
    missing_segments, df_reindexed = find_missing_segments(df, time_col, cbg_col,
                                                          short_gap_thresh=short_gap_thresh,
                                                          medium_gap_thresh=medium_gap_thresh,
                                                          long_gap_thresh=long_gap_thresh)
    segments = handle_missing_data(df_reindexed, missing_segments, cbg_col,
                                   short_gap_thresh=short_gap_thresh,
                                   medium_gap_thresh=medium_gap_thresh,
                                   long_gap_thresh=long_gap_thresh)

    #  Patient-level normalization
    # collect all cbg values from all segments
    if normalize:
        all_cbg = pd.concat([seg[cbg_col] for seg in segments])
        patient_min = all_cbg.min()
        patient_max = all_cbg.max()

        # avoid division by zero
        if patient_max == patient_min:
            patient_max = patient_min + 1e-6
    else:
        patient_min, patient_max = None, None

    # Normalize & add relative time based on patient-wide min/max
    norm_segments = []
    for seg in segments:
        seg = seg.copy()

        # relative time index
        seg["time_minutes"] = np.arange(len(seg)) * 5

        if normalize:
            seg[cbg_col] = (seg[cbg_col] - patient_min) / (patient_max - patient_min)

        norm_segments.append(seg)

    summary = {
        "num_segments": len(norm_segments),
        "avg_length": np.mean([len(s) for s in norm_segments]) if norm_segments else 0,
    }

    return norm_segments, summary