import numpy as np
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# from data_acquisition.ag_news_loader import load_ag_news_csv
from models.deep_learning.ann_classifier import ANNClassifier
from models.deep_learning.cnn_classifier import CNNClassifier
from models.deep_learning.rnn_classifier import RNNClassifier
from models.baseline.mlflow_utils import log_confusion_matrix, log_classification_report

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def compute_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}


# ---------- Data prep for ANN (TF-IDF) ----------
def prepare_tfidf_data(train_texts, test_texts, y_train, y_test, max_features=20000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train.shape[1]


# ---------- Data prep for CNN/RNN (token indices) ----------
def prepare_sequence_data(train_texts, test_texts, y_train, y_test, max_features=20000, max_len=100):
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    vectorizer.fit(train_texts)
    vocab = vectorizer.vocabulary_
    vocab_size = len(vocab) + 2  # + padding, + unk

    def encode(texts):
        encoded = []
        for t in texts:
            tokens = [vocab.get(tok, len(vocab) + 1) for tok in t.lower().split()]
            tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
            encoded.append(tokens)
        return torch.tensor(encoded, dtype=torch.long)

    X_train = encode(train_texts)
    X_test = encode(test_texts)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test, vocab_size


# ---------- Training helper ----------
def train_and_eval(model, train_loader, test_loader, num_epochs=5, lr=1e-3, use_float=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)

    for epoch in range(num_epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            if use_float:
                Xb = Xb.float()
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            if use_float:
                Xb = Xb.float()
            out = model(Xb)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())
    return np.array(all_true), np.array(all_preds)


# ---------- Main ----------
def main():
    train_ds = load_ag_news_csv("/Users/abhmandl/Codes/root/data_acquisition/train.csv")
    test_ds = load_ag_news_csv("/Users/abhmandl/Codes/root/data_acquisition/test.csv")

    X_train_texts = train_ds["text"]
    y_train = np.array(train_ds["y"]) - 1   # FIX: make 0-based
    X_test_texts = test_ds["text"]
    y_test = np.array(test_ds["y"]) - 1

    labels = sorted(list(set(y_train)))

    mlflow.set_experiment("AGNews_DL_Models")

    # --- ANN on TF-IDF ---
    with mlflow.start_run(run_name="ANN_on_TFIDF"):
        X_train, X_test, y_train_tensor, y_test_tensor, input_dim = prepare_tfidf_data(
            X_train_texts, X_test_texts, y_train, y_test
        )
        train_loader = DataLoader(TensorDataset(X_train, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test_tensor), batch_size=64)

        ann = ANNClassifier(input_dim=input_dim, num_classes=len(labels))
        y_true, y_pred = train_and_eval(ann, train_loader, test_loader, use_float=True)

        metrics = compute_metrics(y_true, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        log_confusion_matrix(y_true, y_pred, labels, "cm_ann.png")
        log_classification_report(y_true, y_pred, labels, "report_ann.json")
        mlflow.pytorch.log_model(ann, name="ann")

    # --- CNN on sequences ---
    with mlflow.start_run(run_name="CNN_on_Text"):
        X_train, X_test, y_train_tensor, y_test_tensor, vocab_size = prepare_sequence_data(
            X_train_texts, X_test_texts, y_train, y_test
        )
        train_loader = DataLoader(TensorDataset(X_train, y_train_tensor), batch_size=64, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test_tensor), batch_size=64)

        cnn = CNNClassifier(vocab_size=vocab_size, num_classes=len(labels))
        y_true, y_pred = train_and_eval(cnn, train_loader, test_loader)

        metrics = compute_metrics(y_true, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        log_confusion_matrix(y_true, y_pred, labels, "cm_cnn.png")
        log_classification_report(y_true, y_pred, labels, "report_cnn.json")
        mlflow.pytorch.log_model(cnn, name="cnn")

    # --- LSTM on sequences ---
    with mlflow.start_run(run_name="LSTM_on_Text"):
        rnn = RNNClassifier(vocab_size=vocab_size, num_classes=len(labels))
        y_true, y_pred = train_and_eval(rnn, train_loader, test_loader)

        metrics = compute_metrics(y_true, y_pred)
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        log_confusion_matrix(y_true, y_pred, labels, "cm_lstm.png")
        log_classification_report(y_true, y_pred, labels, "report_lstm.json")
        mlflow.pytorch.log_model(rnn, name="lstm")


if __name__ == "__main__":
    main()
