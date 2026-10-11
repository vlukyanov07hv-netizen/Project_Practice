"""Common scorer extracted unchanged from the verified test evaluator."""

import numpy as np

from src.constants import CLASSES

def pairwise_iou(predictions, targets):
    p = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    t = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    low = np.maximum(p[:, None, :2], t[None, :, :2])
    high = np.minimum(p[:, None, 2:], t[None, :, 2:])
    sizes = np.maximum(high - low, 0)
    intersection = sizes[..., 0] * sizes[..., 1]
    pa = np.maximum(p[:, 2] - p[:, 0], 0) * np.maximum(p[:, 3] - p[:, 1], 0)
    ta = np.maximum(t[:, 2] - t[:, 0], 0) * np.maximum(t[:, 3] - t[:, 1], 0)
    union = pa[:, None] + ta[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

def ap_101(scores, matched, total_gt):
    if total_gt == 0:
        return None
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(matched, dtype=np.float64)[order]
    tp_sum = tp.cumsum()
    precision = tp_sum / np.arange(1, len(tp) + 1)
    recall = tp_sum / total_gt
    # Monotonic precision envelope at 101 evenly spaced recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
    sampled = np.zeros(101, dtype=np.float64)
    ok = indices < len(precision)
    sampled[ok] = precision[indices[ok]]
    return float(sampled.mean())

def evaluate(images, predictions, conf=0.25):
    """All-area bbox AP, 101 recall points, one-to-one score-ordered matching.

    No crowd/ignored targets are present in this YOLO dataset. The same function
    can later evaluate exports from Faster R-CNN, SSD and a repaired DETR.
    P/R/F1 are micro-averaged at fixed confidence, never tuned on test.
    """
    thresholds = np.linspace(0.5, 0.95, 10)
    predictions_by_file = {p["file"]: p["detections"] for p in predictions}
    expected_files = {im["file"] for im in images}
    if set(predictions_by_file) != expected_files or len(predictions) != len(images):
        raise ValueError("Список предсказаний не совпадает с test.")
    per_class, all_ap = [], []
    total_tp = total_fp = total_fn = 0
    for cid, name in enumerate(CLASSES):
        scores, matches = [], []
        total_gt = image_support = 0
        for image in images:
            gt = [r["box"] for r in image["ground_truth"] if r["class_id"] == cid]
            dt = sorted((r for r in predictions_by_file[image["file"]] if r["class_id"] == cid), key=lambda r: -r["score"])[:100]
            total_gt += len(gt)
            image_support += bool(gt)
            ious = pairwise_iou([r["box"] for r in dt], gt)
            matched = np.zeros((len(dt), len(thresholds)), dtype=bool)
            for column, threshold in enumerate(thresholds):
                used = set()
                for row in range(len(dt)):
                    # Choose the best remaining eligible target, not an already
                    # matched target that would unnecessarily turn this into FP.
                    candidates = [j for j in range(len(gt)) if j not in used and ious[row, j] >= threshold]
                    if candidates:
                        target = max(candidates, key=lambda j: (ious[row, j], j))
                        used.add(target)
                        matched[row, column] = True
            scores.extend(r["score"] for r in dt)
            matches.extend(matched.tolist())
        matched = np.asarray(matches, dtype=bool).reshape(-1, len(thresholds))
        aps = [ap_101(scores, matched[:, column].tolist(), total_gt) for column in range(len(thresholds))]
        include = np.asarray(scores) >= conf
        tp = int(matched[include, 0].sum())
        fp = int(include.sum()) - tp
        fn = total_gt - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class.append({"class_id": cid, "class": name, "images": image_support, "objects": total_gt,
                          "AP50": aps[0], "AP50_95": float(np.mean(aps)) if total_gt else None,
                          "precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn})
        if total_gt:
            all_ap.append(aps)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    p = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    return {"mAP50": float(np.mean([a[0] for a in all_ap])) if all_ap else None,
            "mAP50_95": float(np.mean(all_ap)) if all_ap else None,
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "precision_macro": float(np.mean([v["precision"] for v in per_class if v["objects"]])) if all_ap else None,
            "recall_macro": float(np.mean([v["recall"] for v in per_class if v["objects"]])) if all_ap else None,
            "f1_macro": float(np.mean([v["f1"] for v in per_class if v["objects"]])) if all_ap else None,
            "tp": total_tp, "fp": total_fp, "fn": total_fn, "per_class": per_class}
