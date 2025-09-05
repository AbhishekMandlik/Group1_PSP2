import numpy as np
import mlflow
import mlflow.sklearn
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_acquisition.ag_news_loader import load_ag_news_csv
from models.baseline.bert_embedder import BertSentenceEmbedder
from models.baseline.mlflow_utils import log_confusion_matrix, log_classification_report

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def compute_basic_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}

def main():
    train_ds = load_ag_news_csv("data_acquisition/train.csv")
    test_ds = load_ag_news_csv("data_acquisition/test.csv")

    X_train_texts = train_ds["text"]
    y_train = train_ds["y"]
    X_test_texts = test_ds["text"]
    y_test = test_ds["y"]

    labels = sorted(list(set(y_train)))

    # === Set experiment ===
    mlflow.set_experiment("AGNews_BERT_and_Baselines")

    # === Parent run ===
    with mlflow.start_run(run_name="AGNews_Training"):

        # ---------- TF-IDF + MultinomialNB ----------
        with mlflow.start_run(run_name="NB_on_TFIDF", nested=True):
            nb_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000, min_df=2)),
                ("nb", MultinomialNB())
            ])
            nb_pipeline.fit(X_train_texts, y_train)
            y_pred = nb_pipeline.predict(X_test_texts)

            metrics = compute_basic_metrics(y_test, y_pred)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            log_confusion_matrix(y_test, y_pred, labels, "cm_nb.png")
            log_classification_report(y_test, y_pred, labels, "report_nb.json")
            mlflow.sklearn.log_model(nb_pipeline, name="naive_bayes")

        # ---------- BERT embeddings ----------
        embedder = BertSentenceEmbedder()
        X_train_emb = embedder.encode(X_train_texts, batch_size=16)
        X_test_emb = embedder.encode(X_test_texts, batch_size=16)

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_emb)
        X_test_std = scaler.transform(X_test_emb)

        # Logistic Regression
        with mlflow.start_run(run_name="LogReg_on_BERT", nested=True):
            logreg = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)
            logreg.fit(X_train_std, y_train)
            y_pred = logreg.predict(X_test_std)

            metrics = compute_basic_metrics(y_test, y_pred)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            log_confusion_matrix(y_test, y_pred, labels, "cm_logreg.png")
            log_classification_report(y_test, y_pred, labels, "report_logreg.json")
            mlflow.sklearn.log_model(logreg, name="logreg")

        # Linear SVM
        with mlflow.start_run(run_name="SVM_on_BERT", nested=True):
            svm = LinearSVC(random_state=SEED)
            svm.fit(X_train_std, y_train)
            y_pred = svm.predict(X_test_std)

            metrics = compute_basic_metrics(y_test, y_pred)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            log_confusion_matrix(y_test, y_pred, labels, "cm_svm.png")
            log_classification_report(y_test, y_pred, labels, "report_svm.json")
            mlflow.sklearn.log_model(svm, name="svm")

if __name__ == "__main__":
    main()
