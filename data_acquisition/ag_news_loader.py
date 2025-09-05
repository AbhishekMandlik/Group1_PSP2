"""
Custom AG News loader for local CSV files (train.csv, test.csv).
"""

import pandas as pd

def load_ag_news_csv(path: str):
    """
    Load AG News from CSV file with columns: Class Index, Title, Description.

    Args:
        path (str): Path to train.csv or test.csv
    Returns:
        dict with keys: "text", "y"
    """
    df = pd.read_csv(path)

    # Combine Title + Description
    df["text"] = df["Title"].astype(str) + " " + df["Description"].astype(str)

    # Rename Class Index -> y
    df = df.rename(columns={"Class Index": "y"})

    return {"text": df["text"].tolist(), "y": df["y"].tolist()}
