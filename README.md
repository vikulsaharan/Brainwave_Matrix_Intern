#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)           # URLs
    text = re.sub(r"<[^>]+>", " ", text)                           # HTML
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)                    # mentions
    text = re.sub(r"#\w+", " ", text)                              # hashtags
    text = re.sub(r"[^a-z0-9'\s]", " ", text)                      # punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()                       # extra spaces
    return text


def load_dataset(path: str, text_cols=None, label_col="label"):
    df = pd.read_csv(path)
    # Infer sensible defaults if not provided
    if text_cols is None:
        if set(["title", "text"]).issubset(df.columns):
            text_cols = ["title", "text"]
        elif "text" in df.columns:
            text_cols = ["text"]
        elif "content" in df.columns:
            text_cols = ["content"]
        else:
            raise ValueError("Could not infer text columns. Pass --text-cols with one or more columns.")

    if label_col not in df.columns:
        # Try common alternatives
        for alt in ["label", "target", "y", "class"]:
            if alt in df.columns:
                label_col = alt
                break
        else:
            raise ValueError("Could not find label column. Use --label-col to specify.")

    # Combine text columns
    df["__content__"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    X_raw = df["__content__"].fillna("")
    y_raw = df[label_col].astype(str).str.strip()

    # Drop empties
    mask = X_raw.str.len() > 0
    X_raw = X_raw[mask]
    y_raw = y_raw[mask]

    return X_raw, y_raw


def main(args):
    X_raw, y_raw = load_dataset(
        args.csv_path,
        text_cols=args.text_cols,
        label_col=args.label_col
    )

    # Encode labels to 0/1 (or multi-class if present)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    # Train/validation split
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X_raw, y, y_raw, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Compute class weights for imbalance (used to weight the loss)
    cw = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {cls: w for cls, w in zip(np.unique(y_train), cw)}

    # Pipeline: TF-IDF + LinearSVC (fast, strong baseline for text)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(class_weight=class_weight_dict, random_state=args.seed))
    ])

    # Hyperparameter grid (kept small for speed)
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 3, 5],
        "clf__C": [0.5, 1.0, 2.0]
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=args.cv,
        n_jobs=-1,
        scoring="f1_weighted",
        verbose=1
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)

    # For ROC-AUC with LinearSVC, use decision_function
    try:
        decision_scores = best_model.decision_function(X_test)
        # Binary or one-vs-rest
        if decision_scores.ndim == 1:
            auc = roc_auc_score(y_test, decision_scores)
        else:
            # macro-average AUC for multi-class
            auc = roc_auc_score(y_test, decision_scores, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, support = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print("\n=== Best params ===")
    print(gs.best_params_)
    print("\n=== Metrics (Test) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {pr:.4f}")
    print(f"Recall (weighted): {rc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    # Human-readable report with original labels
    y_pred_labels = le.inverse_transform(y_pred)
    print("\n=== Classification report ===")
    print(classification_report(y_test_raw, y_pred_labels, labels=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test_raw, y_pred_labels, labels=class_names)
    print("=== Confusion matrix ===")
    print(pd.DataFrame(cm, index=[f"true_{c}" for c in class_names], columns=[f"pred_{c}" for c in class_names]))

    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "fake_news_model.joblib")
    le_path = os.path.join(args.out_dir, "label_encoder.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(le, le_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved label encoder to: {le_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Fake News Detection model (TF-IDF + LinearSVC).")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to CSV with text and labels.")
    parser.add_argument("--text-cols", type=str, nargs="*", default=None,
                        help="One or more text columns to concatenate (e.g., --text-cols title text).")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name (default: label).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (default: 0.2).")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Where to save model artifacts.")
    args = parser.parse_args()
    main(args)
