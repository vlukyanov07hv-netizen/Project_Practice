"""DETR geometry/loading guards from the successful setup check."""

from src.constants import CLASSES

EPS = 1e-5

COCO_BATCHNORM_COUNTERS = {"model.backbone.conv_encoder.model.layer{}.0.downsample.1.num_batches_tracked".format(i) for i in range(1,5)}

def coco_target(sample):
    objects = []
    for gt in sample["ground_truth"]:
        x1, y1, x2, y2 = gt["box"]
        objects.append({"category_id": gt["class_id"], "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
    return {"image_id": sample["image_id"], "annotations": objects}

def expected_normalized(sample):
    width, height = sample["width"], sample["height"]
    return [[(b["box"][0] + b["box"][2]) / (2 * width),
             (b["box"][1] + b["box"][3]) / (2 * height),
             (b["box"][2] - b["box"][0]) / width,
             (b["box"][3] - b["box"][1]) / height] for b in sample["ground_truth"]]

def verify_encoding(encoded, sample, torch):
    labels = encoded["labels"][0]
    expected = torch.tensor(expected_normalized(sample), dtype=torch.float32).reshape(-1, 4)
    actual = labels["boxes"].cpu()
    ids = labels["class_labels"].cpu().tolist()
    if ids != [b["class_id"] for b in sample["ground_truth"]]:
        raise ValueError("Processor изменил число или классы объектов: {}".format(sample["file"]))
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=2e-5, rtol=2e-5):
        raise ValueError("Ошибка преобразования рамок: {}".format(sample["file"]))
    if not torch.isfinite(actual).all().item() or not torch.isfinite(encoded["pixel_values"]).all().item():
        raise ValueError("NaN/Inf после обработки: {}".format(sample["file"]))
    if actual.numel() and (actual.min().item() < -EPS or actual.max().item() > 1 + EPS):
        raise ValueError("Нормализованные рамки вышли за [0,1]")
    if labels["orig_size"].cpu().tolist() != [sample["height"], sample["width"]]:
        raise ValueError("Processor потерял исходный размер изображения")
    if "pixel_mask" not in encoded or not encoded["pixel_mask"].all().item():
        raise ValueError("Некорректная маска пикселей для batch=1")
    return {"file": sample["file"], "original_size_hw": [sample["height"], sample["width"]],
            "processed_shape": list(encoded["pixel_values"].shape), "class_ids": ids,
            "expected_cxcywh": expected.tolist(), "processor_cxcywh": actual.tolist(),
            "geometry_ok": True}

def prepare_sample(sample, processor, Image, torch):
    with Image.open(sample["path"]) as image:
        rgb = image.convert("RGB")
        encoded = processor(images=rgb, annotations=coco_target(sample), return_tensors="pt")
    geometry = verify_encoding(encoded, sample, torch)
    return encoded, geometry

def validate_loading(info):
    allowed = {"class_labels_classifier.weight", "class_labels_classifier.bias"}
    missing = [k for k in info.get("missing_keys", []) if k not in allowed]
    unexpected = [k for k in info.get("unexpected_keys", []) if k not in COCO_BATCHNORM_COUNTERS]
    mismatched = [item for item in info.get("mismatched_keys", []) if item[0] not in allowed]
    errors = info.get("error_msgs", [])
    if missing or unexpected or mismatched or errors:
        raise ValueError("Не удалось корректно загрузить COCO checkpoint: {}".format(
            {"missing": missing, "unexpected": unexpected, "mismatched": mismatched, "errors": errors}))
    return [k for k in info.get("unexpected_keys", []) if k in COCO_BATCHNORM_COUNTERS]
