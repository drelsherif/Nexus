"""metrics.py -- standard binary classification metrics, ADE = positive class."""


def calc_metrics(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 'ADE' and l == 'ADE')
    fp = sum(1 for p, l in zip(preds, labels) if p == 'ADE' and l == 'NOT_ADE')
    fn = sum(1 for p, l in zip(preds, labels) if p == 'NOT_ADE' and l == 'ADE')
    tn = sum(1 for p, l in zip(preds, labels) if p == 'NOT_ADE' and l == 'NOT_ADE')
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    return {
        "f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
