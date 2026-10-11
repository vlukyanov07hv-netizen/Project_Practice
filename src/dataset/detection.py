"""YOLO labels -> original pixel XYXY, shared by every architecture."""
import collections
import math
from pathlib import Path
from PIL import Image
from src.constants import CLASSES
from src.utils.files import digest, read_json

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-5


def read_boxes(path, width, height):
    boxes = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("нужно 5 полей")
            cid = int(fields[0])
            x, y, w, h = map(float, fields[1:])
            if cid not in range(5) or not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
                raise ValueError("неверный класс или координаты")
            if min(x - w / 2, y - h / 2) < -EPS or max(x + w / 2, y + h / 2) > 1 + EPS:
                raise ValueError("рамка за границами изображения")
            box = [max(0., (x - w / 2) * width), max(0., (y - h / 2) * height),
                   min(float(width), (x + w / 2) * width), min(float(height), (y + h / 2) * height)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("нулевая площадь рамки")
        except (ValueError, OverflowError) as exc:
            raise ValueError("{}:{}: {}".format(path, number, exc)) from exc
        boxes.append({"class_id": cid, "box": box})
    return boxes


def load_split(data, split):
    image_dir, label_dir = data / "images" / split, data / "labels" / split
    files = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError("Нет изображений: " + str(image_dir))
    samples, manifest, seen = [], [], set()
    for index, path in enumerate(files):
        file = path.relative_to(image_dir).as_posix()
        label = (label_dir / file).with_suffix(".txt")
        if not label.is_file() or label in seen:
            raise ValueError("Отсутствует или повторно используется разметка: " + str(label))
        seen.add(label)
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
        boxes = read_boxes(label, width, height)
        samples.append({"file": file, "path": str(path), "label": str(label), "image_id": index,
                        "width": width, "height": height, "ground_truth": boxes})
        manifest.append({"file": file, "image_sha256": digest(path), "label_sha256": digest(label),
                         "width": width, "height": height, "objects": len(boxes)})
    orphans = [p for p in label_dir.rglob("*.txt") if p.is_file() and p not in seen]
    if orphans:
        raise ValueError("Разметка без изображения: " + str(orphans[:3]))
    return samples, manifest


def verify_reference(manifest, root, split):
    expected = read_json(root / "results/reference/eda" / (split + "_manifest.json"))
    # Earlier test manifest intentionally lacks width/height; compare its exact fields.
    projected = [{key: row[key] for key in template}
                 for row, template in zip(manifest, expected)]
    if len(manifest) != len(expected) or projected != expected:
        raise ValueError("{} отличается от данных проверенного эксперимента; оценка остановлена.".format(split))


def summary(samples):
    image_counts, box_counts = collections.Counter(), collections.Counter()
    for sample in samples:
        image_counts.update({b["class_id"] for b in sample["ground_truth"]})
        box_counts.update(b["class_id"] for b in sample["ground_truth"])
    return {"images": len(samples), "objects": sum(box_counts.values()),
            "classes": [{"class_id": i, "class": name, "images": image_counts[i], "objects": box_counts[i]}
                        for i, name in enumerate(CLASSES)]}


def torchvision_sample(sample, device):
    import numpy as np
    import torch
    with Image.open(sample["path"]) as im:
        pixels = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255
    target = {"boxes": torch.tensor([b["box"] for b in sample["ground_truth"]],
                                   dtype=torch.float32, device=device).reshape(-1, 4),
              "labels": torch.tensor([b["class_id"] + 1 for b in sample["ground_truth"]],
                                    dtype=torch.int64, device=device)}
    return tensor, target
