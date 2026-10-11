"""Geometry copied from the successful version 1.1 EfficientDet experiment."""
import math
from src.utils.files import write_json

def geometry_plan(sample, size_hw=(512, 512)):
    """Original XYXY/IDs 0..4 -> resized pixel YXYX/IDs 1..5; never normalize boxes."""
    height, width = sample["height"], sample["width"]
    if min(height, width, *size_hw) <= 0:
        raise ValueError("Неверный размер изображения")
    scale = min(size_hw[0] / height, size_hw[1] / width)
    scaled_h, scaled_w = int(height * scale), int(width * scale)
    boxes, labels = [], []
    for row in sample["ground_truth"]:
        cid = row["class_id"]
        x1, y1, x2, y2 = row["box"]
        if cid not in range(5) or not all(math.isfinite(v) for v in row["box"]):
            raise ValueError("Неверный класс или координаты: " + sample["file"])
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Неверная исходная рамка: " + sample["file"])
        converted = [min(y1 * scale, scaled_h), min(x1 * scale, scaled_w),
                     min(y2 * scale, scaled_h), min(x2 * scale, scaled_w)]
        if converted[0] >= converted[2] or converted[1] >= converted[3]:
            raise ValueError("Рамка исчезает при ResizePad: " + sample["file"])
        boxes.append(converted)
        labels.append(cid + 1)
    return {"file": sample["file"], "original_size_hw": [height, width],
            "model_img_size_wh": [width, height], "content_size_hw": [scaled_h, scaled_w],
            "input_size_hw": list(size_hw), "scale_to_input": scale, "img_scale_to_original": 1 / scale,
            "expected_bbox_yxyx_pixels": boxes, "class_ids_one_based": labels}

def normalize_pixels(pixels, input_config):
    import numpy as np
    if pixels.dtype != np.uint8 or pixels.shape != tuple(input_config["input_size"]):
        raise ValueError("Неверный размер/тип RGB после ResizePad")
    # Same formula and model mean/std as the native effdet prefetcher; computed on CPU.
    mean = np.array(input_config["mean"], dtype=np.float32).reshape(3, 1, 1) * 255
    std = np.array(input_config["std"], dtype=np.float32).reshape(3, 1, 1) * 255
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Неверные параметры нормализации")
    normalized = (pixels.astype(np.float32) - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("NaN/Inf в изображении")
    return np.ascontiguousarray(normalized)

def transform_sample(sample, input_config, transform=None):
    import numpy as np
    from PIL import Image
    if transform is None:
        from effdet.data.transforms import transforms_coco_eval
        transform = transforms_coco_eval(img_size=input_config["input_size"][1:],
            interpolation=input_config["interpolation"], use_prefetcher=True,
            fill_color=input_config["fill_color"], mean=input_config["mean"], std=input_config["std"])
    plan = geometry_plan(sample, input_config["input_size"][1:])
    original = np.array([row["box"] for row in sample["ground_truth"]], dtype=np.float32).reshape(-1, 4)
    annotations = {"bbox": original[:, [1, 0, 3, 2]].copy(),
                   "cls": np.array(plan["class_ids_one_based"], dtype=np.int64),
                   "img_size": tuple(plan["model_img_size_wh"])}
    with Image.open(sample["path"]) as image:
        if image.size != (sample["width"], sample["height"]):
            raise ValueError("Исходный размер изображения изменился")
        pixels, annotations = transform(image.convert("RGB"), annotations)
    expected = np.array(plan["expected_bbox_yxyx_pixels"], dtype=np.float32).reshape(-1, 4)
    if annotations["bbox"].shape != expected.shape or not np.allclose(
            annotations["bbox"], expected, atol=1e-3, rtol=1e-5):
        raise ValueError("ResizePad неверно преобразовал рамки: " + sample["file"])
    if annotations["cls"].tolist() != plan["class_ids_one_based"]:
        raise ValueError("ResizePad изменил классы или число объектов")
    if tuple(annotations["img_size"]) != tuple(plan["model_img_size_wh"]):
        raise ValueError("Перепутаны ширина и высота в img_size")
    if not math.isclose(float(annotations["img_scale"]), plan["img_scale_to_original"], rel_tol=1e-6):
        raise ValueError("Некорректный масштаб возвращения предсказаний")
    normalized = normalize_pixels(pixels, input_config)
    plan.update(geometry_ok=True, normalization_ok=True, bbox_yxyx_pixels=annotations["bbox"].tolist())
    return normalized, annotations, plan

def device_batch(pixels, annotations, torch, device):
    images = torch.from_numpy(pixels).unsqueeze(0).to(device=device, dtype=torch.float32)
    target = {"bbox": torch.tensor(annotations["bbox"], device=device, dtype=torch.float32).unsqueeze(0),
              "cls": torch.tensor(annotations["cls"], device=device, dtype=torch.int64).unsqueeze(0),
              "img_scale": torch.tensor([annotations["img_scale"]], device=device, dtype=torch.float32),
              "img_size": torch.tensor([annotations["img_size"]], device=device, dtype=torch.float32)}
    return images, target

def finite_losses(output):
    values = {key: float(output[key].detach().cpu().item()) for key in ["loss", "class_loss", "box_loss"]}
    if not all(math.isfinite(value) and value >= 0 for value in values.values()) or values["loss"] <= 0:
        raise ValueError("Некорректный loss: " + str(values))
    return values

def detection_rows(sample, rows, converter, audit=None):
    boxes, scores, labels = [], [], []
    counters = {"file": sample["file"], "input_rows": len(rows), "native_padding_rows": 0,
                "clipped_rows": 0, "degenerate_rows_removed": 0, "boundary_examples": []}
    for row in rows:
        if len(row) != 6 or not all(math.isfinite(float(value)) for value in row):
            raise ValueError("Неверный формат/NaN в предсказании EfficientDet")
        x1, y1, x2, y2, score, label = map(float, row)
        if row == [0] * 6:
            counters["native_padding_rows"] += 1
            continue  # Native bench pads output with all-zero rows.
        if label != int(label) or int(label) not in range(1, 6) or not 0 <= score <= 1:
            raise ValueError("Неверный класс/confidence в предсказании")
        if x2 < x1 or y2 < y1 or min(x1, y1) < -1e-3 or x2 > sample["width"] + 1e-3 or y2 > sample["height"] + 1e-3:
            raise ValueError("Предсказание не восстановлено в исходных XYXY координатах")
        # Native FP32 clips before inverse scaling: a boundary may return just
        # above W/H (e.g. 234.000015 for H=234). Clamp EVERY coordinate on both
        # sides, then remove boxes with no remaining area. Partial clamping can
        # otherwise create x1>x2 or y1>y2 in a box already collapsed at the edge.
        clipped = [min(float(sample["width"]), max(0., x1)), min(float(sample["height"]), max(0., y1)),
                   min(float(sample["width"]), max(0., x2)), min(float(sample["height"]), max(0., y2))]
        changed = clipped != [x1, y1, x2, y2]
        degenerate = clipped[2] <= clipped[0] or clipped[3] <= clipped[1]
        counters["clipped_rows"] += int(changed)
        counters["degenerate_rows_removed"] += int(degenerate)
        if (changed or degenerate) and len(counters["boundary_examples"]) < 3:
            counters["boundary_examples"].append({"original": [x1, y1, x2, y2], "clipped": clipped,
                "score": score, "class_id": int(label) - 1, "removed": degenerate})
        if degenerate:
            continue
        boxes.append(clipped)
        scores.append(score)
        labels.append(int(label))
    result = converter(sample["file"], labels, scores, boxes, offset=1)
    counters["exported_rows"] = len(result["detections"])
    if audit is not None:
        audit.update(counters)
    return result

def verify_class_head(base):
    anchors_per_location = len(base.config.aspect_ratios) * int(base.config.num_scales)
    output_channels = int(base.class_net.predict.conv_pw.out_channels)
    if int(base.config.num_classes) != 5 or output_channels != 5 * anchors_per_location:
        raise ValueError("Не удалось создать голову на пять классов")
    return output_channels
