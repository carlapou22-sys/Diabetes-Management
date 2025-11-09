import matplotlib.pyplot as plt
import numpy as np

def visualize_windows(segments, window_minutes=120, stride_minutes=45, max_windows=10):
    """
    Visualize the sliding window approach on one example CGM segment.
    """
    if not segments:
        print("⚠️ No segments provided for visualization.")
        return

    seg_idx = 0  # visualize first segment
    seg = segments[seg_idx]
    cbg = seg["cbg"].values
    t = np.arange(len(seg)) * 5  # 5-min sampling in minutes

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, cbg, color="blue", label=f"Segment {seg_idx + 1} CGM trace")

    # --- Overlay windows ---
    W = int(window_minutes / 5)
    S = int(stride_minutes / 5)
    starts = np.arange(0, len(seg) - W + 1, S)

    for i, s in enumerate(starts[:max_windows]):
        e = s + W
        ax.axvspan(t[s], t[e - 1], color="orange", alpha=0.2)
        if i % 2 == 0:
            ax.text(t[s], 1.02, f"W{i + 1}", rotation=0, fontsize=7, color="darkorange")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Normalized CBG")
    ax.set_title(f"Sliding Window Visualization (first {max_windows} windows)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()