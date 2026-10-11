"""Training loop copied from the successful version 1.1 experiment."""
import copy
import math
import random
import time
from src.models import efficientdet_geometry as support
VERSION = "1.1"

def accumulation_divisor(index, count, accumulation):
    return min(accumulation, count - (index // accumulation) * accumulation)

def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order

def augmented_sample(sample, flip):
    result = dict(sample)
    result["ground_truth"] = copy.deepcopy(sample["ground_truth"])
    if flip:
        for row in result["ground_truth"]:
            x1, y1, x2, y2 = row["box"]
            row["box"] = [sample["width"] - x2, y1, sample["width"] - x1, y2]
    return result

def prepare_train(sample, input_config, transform, cfg):
    from PIL import ImageEnhance, ImageOps
    params = cfg["training"]
    flip = random.random() < params["horizontal_flip_probability"]
    brightness = random.uniform(*params["brightness_range"])
    contrast = random.uniform(*params["contrast_range"])
    row = augmented_sample(sample, flip)
    def augment(image, annotations):
        # Boxes have already been mirrored in row, before analytical validation.
        if flip:
            image = ImageOps.mirror(image)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        return transform(image, annotations)
    return support.transform_sample(row, input_config, augment)

def rng_state(np, torch):
    n = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"kind": n[0], "keys": n[1].tolist(), "position": n[2],
                      "has_gauss": n[3], "cached_gaussian": n[4]},
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

def restore_rng(state, np, torch):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["kind"], np.asarray(n["keys"], dtype=np.uint32),
                        n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])

def cpu_weights(base):
    return {name: value.detach().cpu().clone() for name, value in base.state_dict().items()}

def create_optimizer(base, cfg, torch):
    params = cfg["training"]
    backbone, other = [], []
    for name, value in base.named_parameters():
        if value.requires_grad:
            (backbone if name.startswith("backbone.") else other).append(value)
    if not backbone or not other:
        raise ValueError("Не удалось разделить параметры backbone и головы")
    return torch.optim.AdamW([{"params": other, "lr": params["lr"]},
        {"params": backbone, "lr": params["lr_backbone"]}], weight_decay=params["weight_decay"])

def set_epoch_lr(optimizer, cfg, epoch):
    params = cfg["training"]
    factor = params["lr_decay_factor"] if epoch >= params["lr_decay_start_epoch"] else 1.
    for group, rate in zip(optimizer.param_groups, [params["lr"], params["lr_backbone"]]):
        group["lr"] = rate * factor

def train_epoch(bench, optimizer, samples, epoch, cfg, input_config, transform, device, torch, report=None):
    from src.utils.runtime import ensure_finite
    params, count = cfg["training"], len(samples)
    bench.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    totals = {key: 0. for key in ["loss", "class_loss", "box_loss"]}
    updates, first_update_changed_head = 0, False
    before = bench.model.class_net.predict.conv_pw.weight.detach().clone()
    for index, position in enumerate(epoch_order(count, cfg["seed"], epoch)):
        sample = samples[position]
        pixels, annotations, _ = prepare_train(sample, input_config, transform, cfg)
        images, target = support.device_batch(pixels, annotations, torch, device)
        result = bench(images, target)
        values = support.finite_losses(result)
        for key in totals:
            totals[key] += values[key]
        (result["loss"] / accumulation_divisor(index, count, params["gradient_accumulation"])).backward()
        if (index + 1) % params["gradient_accumulation"] == 0 or index + 1 == count:
            norm = float(torch.nn.utils.clip_grad_norm_(bench.parameters(), params["clip_grad_norm"],
                                                        error_if_nonfinite=True).item())
            if not math.isfinite(norm):
                raise ValueError("NaN/Inf в градиентах: " + sample["file"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if report is not None:
                report["optimizer_updates_this_invocation"] += 1
            if updates == 1:
                first_update_changed_head = not torch.equal(before, bench.model.class_net.predict.conv_pw.weight)
                if not first_update_changed_head:
                    raise ValueError("Первый шаг optimizer не изменил классификатор")
                del before
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == count:
            print("Эпоха {}/{}: train {}/{}, средний loss={:.4f}".format(
                epoch, params["epochs"], index + 1, count, totals["loss"] / (index + 1)), flush=True)
        del result, images, target, pixels, annotations
    torch.cuda.synchronize()
    ensure_finite(bench.model)
    return {**{key: value / count for key, value in totals.items()}, "images": count,
            "optimizer_updates": updates, "first_update_changed_head": first_update_changed_head,
            "seconds": time.perf_counter() - started,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}

def ground_truth(samples):
    return [{"file": row["file"], "ground_truth": row["ground_truth"]} for row in samples]

def evaluate_split(bench, samples, split, input_config, transform, device, torch, cfg, validation, diagnostic_dir=None):
    from src.evaluation.metrics import evaluate
    from src.models.detectors import detection_rows as converter
    bench.eval()
    predictions, losses = [], {key: 0. for key in ["loss", "class_loss", "box_loss"]}
    geometry_audit = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(samples, 1):
            pixels, annotations, _ = support.transform_sample(row, input_config, transform)
            images, target = support.device_batch(pixels, annotations, torch, device)
            if validation:
                result = bench(images, target)
                values = support.finite_losses(result)
                for key in losses:
                    losses[key] += values[key]
                detections = result["detections"]
            else:
                # No boxes or classes are passed to the prediction bench on test.
                detections = bench(images, {key: target[key] for key in ["img_size", "img_scale"]})
            if detections.shape != (1, cfg["inference"]["max_det_per_image"], 6):
                raise ValueError("Неверная форма предсказаний: " + row["file"])
            native_rows = detections[0].cpu().tolist()
            audit = {}
            try:
                predictions.append(support.detection_rows(row, native_rows, converter, audit))
            except Exception:
                if diagnostic_dir is not None:
                    support.write_json(diagnostic_dir / (split + "_failed_prediction.json"),
                        {"file": row["file"], "width": row["width"], "height": row["height"], "rows": native_rows})
                raise
            geometry_audit.append(audit)
            if index == 1 or index % 50 == 0 or index == len(samples):
                print("{}: {}/{}".format(split, index, len(samples)), flush=True)
            del pixels, annotations, images, target, detections
            if validation:
                del result
    metrics = evaluate(ground_truth(samples), predictions, conf=cfg["inference"]["precision_recall_confidence"])
    metrics.update(model="efficientdet", mAP=metrics["mAP50"], mAP_definition="mAP@IoU=0.50",
                   split=split, images=len(samples), objects=sum(len(row["ground_truth"]) for row in samples),
                   confidence=cfg["inference"]["precision_recall_confidence"],
                   seconds=time.perf_counter() - started)
    if validation:
        metrics["loss_components"] = {key: value / len(samples) for key, value in losses.items()}
    metrics["geometry_cleanup"] = {key: sum(row[key] for row in geometry_audit)
        for key in ["native_padding_rows", "clipped_rows", "degenerate_rows_removed", "exported_rows"]}
    if diagnostic_dir is not None:
        support.write_json(diagnostic_dir / (split + "_geometry_audit.json"), geometry_audit)
    for key in ["mAP50", "mAP50_95", "precision", "recall", "f1"]:
        if not math.isfinite(metrics[key]):
            raise ValueError("Некорректная метрика " + key)
    return metrics, predictions

def checkpoint_progress(saved, cfg, binding):
    if saved["binding"] != binding or saved["config"] != cfg:
        raise ValueError("Данные, код, окружение или конфигурация отличаются от сохранённого эксперимента")
    completed, history = saved["completed_epochs"], saved["history"]
    pending = saved.get("phase") == "awaiting_validation"
    minimum = 0 if pending else 1
    if not isinstance(completed, int) or not minimum <= completed <= cfg["training"]["epochs"]:
        raise ValueError("Неверное число завершённых эпох")
    if [row["epoch"] for row in history] != list(range(1, completed + 1)):
        raise ValueError("История не соответствует сохранённому прогрессу")
    scores = [row["val"]["mAP50_95"] for row in history]
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
        raise ValueError("Неверные validation метрики в checkpoint")
    if scores:
        chosen = max(range(len(scores)), key=lambda i: scores[i])
        if saved["best_epoch"] != chosen + 1 or saved["best_score"] != scores[chosen]:
            raise ValueError("Лучшая эпоха выбрана не по validation mAP50_95")
    elif saved["best_epoch"] != 0 or saved["best_score"] != -1. or saved["best_model"] is not None:
        raise ValueError("Best checkpoint создан до первой завершённой валидации")
    if pending:
        if saved["current_epoch"] != completed + 1 or saved["current_epoch"] > cfg["training"]["epochs"]:
            raise ValueError("Неверная эпоха в checkpoint перед валидацией")
        if saved["train"]["optimizer_updates"] <= 0 or not math.isfinite(saved["train"]["loss"]):
            raise ValueError("Неверное сохранённое состояние обученной эпохи")
    return completed, history, saved["best_model"], saved["best_epoch"], saved["best_score"]

def choose_resume_checkpoint(validated, pending, cfg, binding):
    """A validated checkpoint wins if both files describe the same epoch."""
    for value in [validated, pending]:
        if value is not None:
            checkpoint_progress(value, cfg, binding)
    if pending is None:
        if validated is None:
            raise ValueError("Нет сохранённой эпохи")
        return validated
    if validated is None:
        if pending["completed_epochs"] != 0:
            raise ValueError("Пропущена история предыдущих завершённых эпох")
        return pending
    if validated["completed_epochs"] >= pending["current_epoch"]:
        return validated
    if pending["completed_epochs"] != validated["completed_epochs"] or pending["history"] != validated["history"]:
        raise ValueError("История checkpoint до/после валидации не совпадает")
    return pending

def train_and_select(base, bench, optimizer, train, val, cfg, binding, saved, output,
                     report, input_config, transform, device, np, torch, finalize_only):
    from src.utils.files import save_torch, write_json
    history, best_state, best_epoch, best_score, completed = [], None, 0, -1., 0
    pending = None
    if saved is not None:
        completed, history, best_state, best_epoch, best_score = checkpoint_progress(saved, cfg, binding)
        base.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], np, torch)
        if saved.get("phase") == "awaiting_validation":
            pending = saved
    if finalize_only and completed != cfg["training"]["epochs"]:
        raise ValueError("--finalize-only допустим только после всех десяти завершённых эпох")
    report.update(completed_epochs=completed, best_epoch=best_epoch, best_validation_mAP50_95=best_score,
                  optimizer_steps=sum(row["train"]["optimizer_updates"] for row in history),
                  pending_validation_reused=False, training_epochs_saved=completed)
    for epoch in range(completed + 1, cfg["training"]["epochs"] + 1):
        report["stage"] = "train_epoch_{}".format(epoch)
        started = time.perf_counter()
        if pending is not None:
            trained = pending["train"]
            report["pending_validation_reused"] = True
            print("Эпоха {} уже обучена и сохранена; повторяю только её валидацию.".format(epoch), flush=True)
        else:
            set_epoch_lr(optimizer, cfg, epoch)
            report["training_performed"] = True
            trained = train_epoch(bench, optimizer, train, epoch, cfg, input_config, transform, device, torch, report)
            report["stage"] = "save_before_val_epoch_{}".format(epoch)
            pending = {"script_version": VERSION, "phase": "awaiting_validation", "current_epoch": epoch,
                "completed_epochs": epoch - 1, "model": cpu_weights(base), "optimizer": optimizer.state_dict(),
                "best_model": best_state, "best_epoch": best_epoch, "best_score": best_score, "history": history,
                "train": trained, "config": cfg, "binding": binding, "rng": rng_state(np, torch)}
            save_torch(output / "pending_validation.pt", pending, torch)
            print("Обучение эпохи {} сохранено до валидации.".format(epoch), flush=True)
        report.update(training_epochs_saved=epoch, pending_validation_epoch=epoch,
                      optimizer_steps=sum(row["train"]["optimizer_updates"] for row in history) + trained["optimizer_updates"])
        report["stage"] = "val_epoch_{}".format(epoch)
        validated, _ = evaluate_split(bench, val, "val", input_config, transform, device, torch, cfg, True,
                                      output / "validation" / ("epoch_{}".format(epoch)))
        score = validated["mAP50_95"]
        improved = score > best_score
        if improved:
            best_state, best_score, best_epoch = cpu_weights(base), score, epoch
        elapsed = time.perf_counter() - started
        if saved is not None and saved.get("phase") == "awaiting_validation" and saved["current_epoch"] == epoch:
            elapsed += trained.get("seconds", 0.)
        history.append({"epoch": epoch, "train": trained, "val": validated,
                        "best_so_far": improved, "seconds": elapsed})
        # All coupled states in one atomic file: failures preserve the previous full epoch.
        payload = {"script_version": VERSION, "phase": "validated", "completed_epochs": epoch, "model": cpu_weights(base),
            "optimizer": optimizer.state_dict(), "best_model": best_state, "best_epoch": best_epoch,
            "best_score": best_score, "history": history, "config": cfg, "binding": binding,
            "rng": rng_state(np, torch)}
        save_torch(output / "last_training.pt", payload, torch)
        # Only after the complete checkpoint replaces the old one. If cleanup
        # fails, choose_resume_checkpoint still prefers the validated epoch.
        (output / "pending_validation.pt").unlink(missing_ok=True)
        pending = None
        report.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_mAP50_95=best_score,
                      pending_validation_epoch=None,
                      optimizer_steps=sum(row["train"]["optimizer_updates"] for row in history))
        write_json(output / "history.json", history)
        write_json(output / "progress.json", {"completed_epochs": epoch, "best_epoch": best_epoch,
                    "best_validation_mAP50_95": best_score})
        remaining = (cfg["training"]["epochs"] - epoch) * sum(row["seconds"] for row in history) / len(history) / 60
        print("Сохранена эпоха {}. Val mAP50={:.4f}, mAP50-95={:.4f}; лучшая эпоха {}. Осталось ~{:.0f} мин.".format(
            epoch, validated["mAP50"], score, best_epoch, remaining), flush=True)
        del payload
    if completed != cfg["training"]["epochs"] and report["completed_epochs"] != cfg["training"]["epochs"]:
        raise ValueError("Обучение не завершено; test запрещён")
    if best_state is None:
        raise ValueError("Нет лучшего checkpoint")
    base.load_state_dict(best_state, strict=True)
    return history, best_epoch, best_score

def make_plot(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], "o-", label="train (augmented)")
    axes[0].plot(epochs, [row["val"]["loss_components"]["loss"] for row in history], "o-", label="val")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, [row["val"]["mAP50"] for row in history], "o-", label="val mAP50")
    axes[1].plot(epochs, [row["val"]["mAP50_95"] for row in history], "o-", label="val mAP50-95")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("EfficientDet-D0: checkpoint selected on validation")
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
