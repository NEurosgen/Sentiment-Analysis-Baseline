from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def save_model(model, path):
    ensure_dir(path.parent)
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def evaluate_and_report(y_true,y_pred,labels=("negative", "positive"),digits = 3):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    cls_rep = classification_report(y_true, y_pred, labels=list(labels), digits=digits, output_dict=True)
    return {"accuracy": acc, "confusion_matrix": cm.tolist(), "classification_report": cls_rep}

def format_human_report(metrics,train_size,test_size,total_size,class_balance= None,vectorizer_name= "bow"):
    acc = metrics["accuracy"]
    cm = np.array(metrics["confusion_matrix"])
    cls_rep = metrics["classification_report"]

    lines = []
    lines.append("Отчёт по анализу тональности (baseline)\n")
    lines.append("1) Данные:")
    lines.append(f"   - Всего объектов: {total_size}")
    lines.append(f"   - Train: {train_size}, Test: {test_size}")
    if class_balance:
        lines.append(f"   - Баланс классов: {class_balance}")
    lines.append("")
    lines.append("2) Модель:")
    lines.append(f"   - Векторизация: {'Bag-of-Words' if vectorizer_name=='bow' else 'TF-IDF'} (n-gram 1–2)")
    lines.append("   - Классификатор: Logistic Regression (max_iter=1000)")
    lines.append("")
    lines.append("3) Результаты на тесте:")
    lines.append(f"   - Accuracy: {acc:.4f}")
    lines.append(f"   - Confusion matrix [rows=true, cols=pred] labels=['negative','positive']:\n{cm}")
    lines.append("")
    lines.append("=== Classification report ===")
    for label in ["negative", "positive", "macro avg", "weighted avg"]:
        if label in cls_rep:
            d = cls_rep[label]
            lines.append(f"{label:13s}  prec={d['precision']:.3f}  rec={d['recall']:.3f}  f1={d['f1-score']:.3f}  support={int(d['support'])}")
    return "\n".join(lines)

def save_text(path,content):
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

def save_json(path, obj):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
