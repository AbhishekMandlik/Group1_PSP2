"""
MLflow helper utilities for logging metrics and artifacts.
"""
import io
import json
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

def log_confusion_matrix(y_true, y_pred, labels, artifact_name="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    fig.savefig(artifact_name, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(artifact_name)

def log_classification_report(y_true, y_pred, labels, artifact_name="classification_report.json"):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    with open(artifact_name, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(artifact_name)

    if "macro avg" in report:
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])
    if "weighted avg" in report:
        mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])
