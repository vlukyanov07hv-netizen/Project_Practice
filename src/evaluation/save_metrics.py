import json
import os
import pandas as pd

# Список всех обученных моделей
MODELS = [
    {
        "name": "yolov8n_baseline",
        "results_dir": r"G:\Project\cv-project\results\yolo_baseline-3",
        "model_arch": "yolov8n",
    },
    {
        "name": "yolov8s_small",
        "results_dir": r"G:\Project\cv-project\results\yolo_small",
        "model_arch": "yolov8s",
    },
    {
        "name": "yolov8n_augmented",
        "results_dir": r"G:\Project\cv-project\results\yolo_augmented",
        "model_arch": "yolov8n",
    },
]

OUT_DIR = r"G:\Project\cv-project\results\metrics"
os.makedirs(OUT_DIR, exist_ok=True)

def get_value(row, keys):
    for k in keys:
        for col in row.index:
            if k.lower() in col.lower():
                return float(row[col])
    return 0.0

all_metrics = []

for m in MODELS:
    csv_path = os.path.join(m["results_dir"], "results.csv")
    if not os.path.exists(csv_path):
        print(f"[!] Не найден: {csv_path}, пропускаем")
        continue

    df = pd.read_csv(csv_path)
    last = df.iloc[-1]

    precision = get_value(last, ["precision"])
    recall = get_value(last, ["recall"])
    map50 = get_value(last, ["mAP50(B)", "mAP_0.5"])
    map5095 = get_value(last, ["mAP50-95(B)", "mAP_0.5:0.95"])
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "model": m["name"],
        "architecture": m["model_arch"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map5095, 4),
    }
    all_metrics.append(metrics)

    # Отдельный json на модель
    with open(os.path.join(OUT_DIR, f"{m['name']}.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

# Общий json со всеми моделями
with open(os.path.join(OUT_DIR, "all_models.json"), "w") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

# Красивый вывод
print("\n" + "=" * 70)
print(f"{'Модель':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP@0.5':>10}")
print("=" * 70)
for m in all_metrics:
    print(f"{m['model']:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['mAP50']:>10.4f}")
print("=" * 70)
print(f"\nВсе метрики сохранены в: {OUT_DIR}")