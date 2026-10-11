"""Plots use measured metrics/history and deterministic example selection."""
import csv
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
from src.constants import CLASSES, MODEL_TITLES
from src.utils.files import read_json


def learning_plot(root, out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = out / "history.json"
    reference = root / "results/reference" / name
    if not history.is_file():
        history = reference / "history.json"
    csv_file = out / "training_results.csv"
    if not csv_file.is_file():
        csv_file = reference / "results.csv"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    available = False
    if history.is_file():
        value = read_json(history)
        shutil.copy2(history, out / "training_history.json")
        if isinstance(value, dict) and "loss" in value:
            values = value["loss"]
            ax.plot(range(1, len(values) + 1), values, marker="o", label="Train loss")
            available = True
        elif isinstance(value, list) and value:
            epochs = [row["epoch"] for row in value]
            values = [row.get("train_loss", row.get("train", {}).get("loss")) for row in value]
            ax.plot(epochs, values, marker="o", label="Train loss")
            if name == "efficientdet" and all("loss_components" in row.get("val", {}) for row in value):
                ax.plot(epochs, [row["val"]["loss_components"]["loss"] for row in value],
                        marker="o", label="Validation loss")
            if all("loss" in row.get("val", {}) for row in value):
                ax.plot(epochs, [row["val"]["loss"] for row in value], marker="o", label="Validation loss")
            available = True
    elif csv_file.is_file():
        shutil.copy2(csv_file, out / "training_results_used.csv")
        with csv_file.open(encoding="utf-8-sig", newline="") as stream:
            rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
        if rows:
            epochs = [float(row.get("epoch", i + 1)) for i, row in enumerate(rows)]
            for key in rows[0]:
                if "loss" in key and key.startswith(("train/", "val/")):
                    ax.plot(epochs, [float(row[key]) for row in rows], label=key)
                    available = True
    if not available:
        plt.close(fig)
        return False
    ax.set(title=MODEL_TITLES[name] + ": measured learning history", xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plots/learning.png", dpi=160)
    plt.close(fig)
    return True


def make_outputs(root, out, name, metrics, images, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = out / "plots"
    target.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    bars = ax.bar(keys, [metrics[key] for key in keys], color="#0072B2")
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set(ylim=(0, 1.05), title=MODEL_TITLES[name] + ": " + metrics["split"], ylabel="Metric")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(target / "metrics.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rows = metrics["per_class"]
    ax.bar(range(5), [row["AP50"] or 0. for row in rows], color="#009E73")
    ax.set_xticks(range(5), CLASSES, rotation=15)
    ax.set(ylim=(0, 1), ylabel="AP50", title="Per-class AP50; support is saved in metrics.json")
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(target / "per_class.png", dpi=160)
    plt.close(fig)
    history_available = learning_plot(root, out, name)
    examples = out / "examples"
    examples.mkdir(exist_ok=True)
    by_file = {row["file"]: row["detections"] for row in predictions}
    seen = set()
    for cid in range(5):
        sample = next((s for s in images if s["file"] not in seen and
                       any(b["class_id"] == cid for b in s["ground_truth"])), None)
        if not sample:
            continue
        seen.add(sample["file"])
        with Image.open(sample["path"]) as im:
            rgb = im.convert("RGB")
        scale = min(1., 1000 / rgb.width, 650 / rgb.height)
        rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))))
        draw = ImageDraw.Draw(rgb)
        for gt in sample["ground_truth"]:
            draw.rectangle([v * scale for v in gt["box"]], outline="lime", width=2)
        dt = sorted((d for d in by_file[sample["file"]] if d["score"] >= .25), key=lambda d: -d["score"])
        for row in dt[:20]:
            draw.rectangle([v * scale for v in row["box"]], outline="red", width=2)
        canvas = Image.new("RGB", (max(700, rgb.width), rgb.height + 40), "white")
        canvas.paste(rgb, ((canvas.width - rgb.width) // 2, 40))
        ImageDraw.Draw(canvas).text((5, 5), "{}; GT green; predictions red; displayed {}/{}".format(
            sample["file"], min(20, len(dt)), len(dt)), fill="black")
        canvas.save(examples / ("example_{:02d}.jpg".format(cid + 1)), quality=91)
    return history_available
