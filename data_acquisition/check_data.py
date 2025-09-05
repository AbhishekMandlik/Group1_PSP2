"""
Quick script to check AG News train/test CSVs.
"""

import pandas as pd
from collections import Counter

def inspect_csv(path: str, n_samples: int = 5):
    df = pd.read_csv(path)
    print(f"\n=== Inspecting {path} ===")
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))

    # Class distribution
    if "Class Index" in df.columns:
        counts = Counter(df["Class Index"])
        print("Class distribution:", counts)

    # Show first few rows
    print("\nSample rows:")
    print(df.head(n_samples))


if __name__ == "__main__":
    inspect_csv("/Users/abhmandl/Codes/root/data_acquisition/train.csv")
    inspect_csv("/Users/abhmandl/Codes/root/data_acquisition/test.csv")
