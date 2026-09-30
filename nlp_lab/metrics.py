import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss


def evaluate_predictions(rows, probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(rows), len(labels)) or not np.isfinite(p).all():
        raise ValueError("预测概率形状错误或含非有限值")
    if (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("预测概率必须非负且每行和为 1")
    # FP32 softmax introduces harmless rounding; normalize after validation so
    # log_loss's FP64 tolerance does not misclassify it as unnormalized scores.
    p = p / p.sum(axis=1, keepdims=True)
    truth = [labels.index(x["label"]) for x in rows]
    pred = p.argmax(axis=1)
    indices = list(range(len(labels)))
    results = {
        "samples": len(rows), "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=indices, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=indices, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(truth, p, labels=indices)),
        "labels": labels, "confusion_matrix": confusion_matrix(truth, pred, labels=indices).tolist(),
        "per_class": classification_report(truth, pred, labels=indices, target_names=labels, output_dict=True, zero_division=0),
        "unsupported_labels": [label for i, label in enumerate(labels) if i not in truth],
    }
    details = [{"text": row["text"], "source_line": row["source_line"], "expected": row["label"],
                "predicted": labels[int(pred[i])], "confidence": float(p[i, pred[i]]),
                "correct": row["label"] == labels[int(pred[i])]} for i, row in enumerate(rows)]
    return results, details
