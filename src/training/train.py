"""Explicit new training; original checkpoints are never overwritten."""
import copy
import math
import random
import shutil
import time
from pathlib import Path
from PIL import Image
from src.dataset.detection import torchvision_sample
from src.models.detectors import fresh_detector
from src.utils.files import save_torch, write_json
from src.utils.runtime import ensure_finite
from src.evaluation.metrics import evaluate


def train_yolo(cfg, root, out, device):
    import yaml
    from src.constants import CLASSES
    detector = fresh_detector(cfg, root, device)
    data_file = out / "data_runtime.yaml"
    # The absolute runtime path is generated from the portable relative config.
    dataset = {"path": str((root / cfg["data"]).resolve()),
               "train": "images/train", "val": "images/val", "test": "images/test",
               "names": dict(enumerate(CLASSES))}
    data_file.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    train = cfg["training"]
    kwargs = copy.deepcopy(train.get("ultralytics_args", {}))
    detector.model.train(data=str(data_file), epochs=train["epochs"], imgsz=640,
        batch=train["batch"], seed=cfg["seed"], workers=0, deterministic=True,
        device=str(device), project=str(out), name="native", exist_ok=False, **kwargs)
    native = Path(detector.model.trainer.save_dir)
    best = native / "weights/best.pt"
    if not best.is_file():
        raise ValueError("Ultralytics не сохранил best.pt")
    ensure_finite(detector.model.model)
    if (native / "results.csv").is_file():
        shutil.copy2(native / "results.csv", out / "training_results.csv")
    return best


def train_torch(cfg, root, out, device, train_images, val_images):
    import torch
    from src.models.detr_utils import prepare_sample
    detector = fresh_detector(cfg, root, device)
    model, params = detector.model, cfg["training"]
    if cfg["model"] == "detr":
        backbone, other = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (backbone if "backbone" in name else other).append(parameter)
        optimizer = torch.optim.AdamW([{"params": other, "lr": params["lr"]},
            {"params": backbone, "lr": params["lr_backbone"]}], weight_decay=params["weight_decay"])
    elif cfg["model"] == "ssd":
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=params["lr"], weight_decay=params["weight_decay"])
    else:
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
            lr=params["lr"], momentum=params["momentum"], weight_decay=params["weight_decay"])
    history, best_state, best_epoch, best_score = [], None, 0, -1.
    accumulation = params["gradient_accumulation"]
    for epoch in range(1, params["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = list(range(len(train_images)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        total, updates, started = 0., 0, time.perf_counter()
        for index, position in enumerate(order):
            sample = train_images[position]
            if cfg["model"] == "detr":
                encoded, _ = prepare_sample(sample, detector.processor, Image, torch)
                batch = {key: encoded[key].to(device) for key in ["pixel_values", "pixel_mask"]}
                batch["labels"] = [{key: value.to(device) if torch.is_tensor(value) else value
                                    for key, value in label.items()} for label in encoded["labels"]]
                loss = model(**batch).loss
            else:
                tensor, target = torchvision_sample(sample, device)
                losses = model([tensor], [target])
                loss = sum(losses.values())
            if not torch.isfinite(loss).item():
                raise ValueError("NaN/Inf в loss: эпоха {}, файл {}".format(epoch, sample["file"]))
            value = float(loss.detach().cpu())
            total += value
            group_start = (index // accumulation) * accumulation
            divisor = min(accumulation, len(train_images) - group_start)
            (loss / divisor).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(train_images):
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.get("clip_grad_norm", float("inf")),
                                              error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(train_images):
                print("Эпоха {}/{}: {}/{}, loss={:.4f}".format(
                    epoch, params["epochs"], index + 1, len(train_images), total / (index + 1)), flush=True)
            del loss
        ensure_finite(model)
        predictions, _ = detector.predict(val_images)
        val_metrics = evaluate(val_images, predictions, conf=.25)
        score = val_metrics["mAP50_95"]
        if not math.isfinite(score):
            raise ValueError("Неверная validation mAP")
        if score > best_score:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch, best_score = epoch, score
        record = {"epoch": epoch, "train_loss": total / len(train_images), "val": val_metrics,
                  "train_images": len(train_images), "optimizer_updates": updates,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        # One replaceable progress file; all weights are local to this new run.
        save_torch(out / "last_training.pt", {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "best_model": best_state, "best_epoch": best_epoch, "best_score": best_score,
            "history": history, "config": cfg}, torch)
        write_json(out / "history.json", history)
        print("Сохранена эпоха {}. Val mAP50-95={:.4f}; лучшая эпоха {}.".format(epoch, score, best_epoch), flush=True)
    model.load_state_dict(best_state, strict=True)
    ensure_finite(model)
    best_dir = out / "weights"
    best_dir.mkdir()
    if cfg["model"] == "detr":
        model.save_pretrained(best_dir, safe_serialization=True)
        detector.processor.save_pretrained(best_dir)
        best = best_dir
    else:
        best = best_dir / "best.pth"
        save_torch(best, best_state, torch)
    write_json(out / "selection.json", {"best_epoch": best_epoch, "validation_mAP50_95": best_score,
        "criterion": "maximum val mAP50_95; earlier epoch on exact ties", "test_used": False})
    return best


def train_model(cfg, root, out, device, train_images, val_images):
    if cfg["model"] == "efficientdet":
        from src.training.efficientdet import train_efficientdet
        return train_efficientdet(cfg, root, out, device, train_images, val_images)
    if cfg["model"].startswith("yolo"):
        return train_yolo(cfg, root, out, device)
    return train_torch(cfg, root, out, device, train_images, val_images)
