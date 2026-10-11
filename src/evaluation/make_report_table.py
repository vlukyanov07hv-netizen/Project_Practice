import json
import os

METRICS_DIR = r"G:\Project\cv-project\results\metrics"

models = ["yolov8n_baseline", "yolov8s_small", "yolov8n_augmented"]

print(f"{'Модель':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14}")
print("=" * 85)
for name in models:
    path = os.path.join(METRICS_DIR, f"{name}.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        m = json.load(f)
    print(f"{m['model']:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {m['mAP50']:>10.4f} {m['mAP50_95']:>14.4f}")
print("=" * 85)