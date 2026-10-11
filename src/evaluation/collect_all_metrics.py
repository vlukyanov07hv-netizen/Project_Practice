import json
import os
import pandas as pd

RESULTS_ROOT = r"G:\Project\cv-project\results"
METRICS_DIR = os.path.join(RESULTS_ROOT, "metrics")
OUT_PATH = os.path.join(RESULTS_ROOT, "final_comparison.csv")

rows = []

# 1. YOLO — из metrics/*.json
if os.path.isdir(METRICS_DIR):
    for name in ["yolov8n_baseline", "yolov8s_small", "yolov8n_augmented"]:
        path = os.path.join(METRICS_DIR, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "model": m["model"],
            "type": "YOLO (one-stage)",
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "mAP50": m["mAP50"],
            "mAP50_95": m["mAP50_95"],
            "final_loss": "—",
        })

# 2. Colab-модели — рекурсивно ищем все history.json
print("Поиск history.json во всех подпапках results...")
found = []
for dirpath, _, files in os.walk(RESULTS_ROOT):
    for f in files:
        if f == "history.json":
            full = os.path.join(dirpath, f)
            # Пропускаем YOLO-папки
            if "yolo" in full.lower():
                continue
            found.append(full)
            print(f"  найден: {full}")

for hist_path in found:
    # Имя модели = имя родительской папки
    model_name = os.path.basename(os.path.dirname(hist_path))

    try:
        with open(hist_path) as f:
            h = json.load(f)
    except Exception as e:
        print(f"  [!] Ошибка чтения {hist_path}: {e}")
        continue

    losses = h.get("loss", [])
    final = losses[-1] if losses else "—"

    rows.append({
        "model": model_name,
        "type": "Colab",
        "precision": "—",
        "recall": "—",
        "f1": "—",
        "mAP50": "—",
        "mAP50_95": "—",
        "final_loss": f"{final:.4f}" if isinstance(final, (int, float)) else final,
    })

# Собираем в DataFrame
df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n" + "=" * 90)
print(df.to_string(index=False))
print("=" * 90)
print(f"\nСохранено: {OUT_PATH}")