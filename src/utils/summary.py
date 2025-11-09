import pandas as pd

def meal_summary(df, key):
    meals = df['carbInput'].dropna()
    positive_meals = meals[meals > 0]

    # Robust split detection
    if "train" in key.lower():
        split = "train"
    elif "test" in key.lower():
        split = "test"
    else:
        split = "unknown"

    # Extract dataset and patient more safely
    parts = key.split("_")
    dataset = parts[0] if len(parts) > 0 else "unknown"
    patient = parts[1] if len(parts) > 1 else "unknown"

    return {
        "dataset": dataset,
        "patient": patient,
        "split": split,
        "total_rows": len(df),
        "total_meal_annotations": len(meals),
        "positive_meals": len(positive_meals),
        "total_CHO_grams": positive_meals.sum(),
    }


def build_summary(patient_data):
    """Builds a summary DataFrame from all patients."""
    summaries = [meal_summary(df, k) for k, df in patient_data.items()]
    summary_df = pd.DataFrame(summaries).sort_values(
        ["dataset", "positive_meals"], ascending=[True, False]
    )
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df