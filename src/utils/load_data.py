import os
import glob
import pandas as pd

def load_patient_data(base_dir="../data/Ohio Data"):
    """
    Loads all patient CSVs from structured folders:
    Ohio2018/train, Ohio2018/test, Ohio2020/train, Ohio2020/test
    Returns dict {dataset_patient_split: DataFrame}
    """
    datasets = ["Ohio2018_processed", "Ohio2020_processed"]
    splits = ["train", "test"]
    patient_data = {}

    for dataset in datasets:
        for split in splits:
            folder = os.path.join(base_dir, dataset, split)
            files = glob.glob(os.path.join(folder, "*.csv"))
            print(f" {dataset}/{split}: found {len(files)} files")
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]

                # Detect train/test
                if "training" in name.lower():
                    split = "train"
                elif "testing" in name.lower():
                    split = "test"
                else:
                    split = "unknown"

                df = pd.read_csv(f)
                df["dataset"] = dataset
                df["split"] = split

                # Use short, clean key
                patient_data[name] = df

    print(f"\n Total datasets loaded: {len(patient_data)}")
    print("Example keys:", list(patient_data.keys())[:6])
    return patient_data