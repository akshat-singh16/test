import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z\s]", " ", regex=True)
    )


def train_model(input_csv, model_out, vectorizer_out):
    df = pd.read_csv(input_csv)

    if "description" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must contain 'description' and 'category' columns.")

    X = clean_text(df["description"])
    y = df["category"]

    # remove rare categories
    valid_cats = y.value_counts()[lambda x: x >= 2].index
    df = df[df["category"].isin(valid_cats)]

    X = clean_text(df["description"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, model_out)
    joblib.dump(vectorizer, vectorizer_out)

    print(f"Saved model as {model_out}")
    print(f"Saved vectorizer as {vectorizer_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_out", default="model.pkl")
    parser.add_argument("--vectorizer_out", default="vectorizer.pkl")

    args = parser.parse_args()

    train_model(args.input, args.model_out, args.vectorizer_out)
