from pathlib import Path
from typing import List, Tuple, Literal
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_text
Split = Tuple[pd.Series, pd.Series, pd.Series, pd.Series]

def load_uci_sentiment_labelled_txt(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep="\t", header=None, names=["text", "label"], quoting=3, encoding="utf-8")
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["sentiment"] = data["label"].map({0: "negative", 1: "positive"})
    data = data.drop(columns=["label"]).dropna(subset=["text", "sentiment"]).copy()
    data["text"] = data["text"].apply(preprocess_text)
    data = data[["text", "sentiment"]]
    return data

def train_test_split_stratified(df,test_size=0.2,random_state=42):
    X = df["text"].values
    y = df["sentiment"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_tr, X_te, y_tr, y_te
