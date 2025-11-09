
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_preprocessed_segments(df, segments, time_col="5minute_intervals_timestamp"):
    """
    Plot raw CGM data with gaps (gray) and preprocessed continuous segments.
    Top subplot = raw + true CGM values per segment
    Bottom subplot = normalized segments (0–1)
    """

    #  Ensure proper time base
    raw_x = df[time_col]
    if not np.issubdtype(raw_x.dtype, np.datetime64):
        raw_x = pd.to_datetime(raw_x * 5, unit="m", origin="unix", errors="coerce")

    raw_x_minutes = (raw_x - raw_x.min()).dt.total_seconds() / 60.0

    #  Setup subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(segments)))

    #  Plot 1: Raw CGM data
    axes[0].plot(raw_x_minutes, df["cbg"], color="gray", alpha=0.5, linewidth=1.2, label="Raw CGM (with gaps)")
    axes[0].set_title("Raw Continuous Glucose Monitor (CGM) Data")
    axes[0].set_ylabel("CBG (mg/dL)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Cleaned & normalized segments
    for i, (seg, c) in enumerate(zip(segments, colors)):
        seg = seg.copy()
        if not np.issubdtype(seg.index.dtype, np.datetime64):
            seg.index = pd.to_datetime(seg.index * 5, unit="m", origin="unix", errors="coerce")

        seg_x = (seg.index - raw_x.min()).total_seconds() / 60.0

        cbg_min, cbg_max = seg["cbg"].min(), seg["cbg"].max()
        seg_norm = (seg["cbg"] - cbg_min) / (cbg_max - cbg_min) if cbg_max > cbg_min else seg["cbg"]

        axes[1].plot(seg_x, seg_norm, color=c, linewidth=1.5, label=f"Segment {i + 1}")
        axes[0].plot(seg_x, seg["cbg"], color=c, linewidth=1.0, alpha=0.6)

    axes[1].set_title("Preprocessed CGM (Continuous Segments after Gap Removal)")
    axes[1].set_xlabel("Time (minutes since start)")
    axes[1].set_ylabel("CBG (normalized 0–1 per segment)")
    axes[1].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()