import argparse
from pathlib import Path
import pandas as pd

from data_loader import load_uci_sentiment_labelled_txt, train_test_split_stratified
from model import build_pipeline
from utils import (
    evaluate_and_report, format_human_report,
    save_model, save_text, save_json
)

def parse_args():
    p = argparse.ArgumentParser(description="Train baseline sentiment model")
    p.add_argument("--data_dir", type=Path, default=Path("data"),
                   help="Папка с *labelled.txt файлами UCI")
    p.add_argument("--vectorizer", type=str, default="bow", choices=["bow", "tfidf"],
                   help="Векторизация: bow|tfidf")
    p.add_argument("--out_dir", type=Path, default=Path("artifacts"),
                   help="Куда класть модель и отчёты")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--class_weight", type=str, default=None, choices=[None, "balanced"],
                   help="balanced — если классы несбалансированы")
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        args.data_dir / "amazon_cells_labelled.txt",
        args.data_dir / "imdb_labelled.txt",
        args.data_dir / "yelp_labelled.txt",
    ]
    df = load_uci_sentiment_labelled_txt(files)
    class_balance = dict(df["sentiment"].value_counts())
    total_size = len(df)


    X_train, X_test, y_train, y_test = train_test_split_stratified(
        df, test_size=args.test_size, random_state=args.random_state
    )

    pipe = build_pipeline(
        vectorizer=args.vectorizer,
        ngram_range=(1, 2),
        max_iter=1000,
        random_state=args.random_state,
        class_weight=args.class_weight
    )


    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = evaluate_and_report(y_test, y_pred)

    human_report = format_human_report(
        metrics=metrics,
        train_size=len(X_train),
        test_size=len(X_test),
        total_size=total_size,
        class_balance=class_balance,
        vectorizer_name=args.vectorizer
    )
    print(human_report)

    save_text(out_dir / "report.txt", human_report)
    save_json(out_dir / "metrics.json", metrics)
    save_model(pipe, out_dir / "sentiment_benchmark.joblib")

if __name__ == "__main__":
    main()
