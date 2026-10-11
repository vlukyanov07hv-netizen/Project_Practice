"""Training/evaluation orchestration with unchanged test protocol."""
import copy
import gc
from src.constants import CLASSES, MODEL_NAMES, inference_protocol
from src.dataset.detection import load_split, verify_reference
from src.evaluation.metrics import evaluate
from src.evaluation.visualize import make_outputs
from src.models.detectors import load_detector
from src.utils.files import digest, project_path, read_json, relative, write_json, write_text
from src.utils.runtime import backend_state, environment, prepare_device, reference_backend


def load_config(root, name, custom=None):
    import yaml
    path = project_path(root, custom) if custom else root / "configs/pipeline" / (name + ".yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if cfg["model"] != name or cfg["classes"] != CLASSES:
        raise ValueError("Модель или порядок классов в конфигурации не совпадает")
    if name == "efficientdet" and (cfg["architecture"] != "tf_efficientdet_d0" or cfg["input_size"] != [3, 512, 512]):
        raise ValueError("Проверенный EfficientDet использует tf_efficientdet_d0 и ResizePad 512x512")
    if cfg["inference"] != inference_protocol(name):
        raise ValueError("Протокол сравнения изменился. Для текущего сравнения используй сохранённые параметры inference.")
    if cfg.get("runtime_backend", reference_backend()) != reference_backend():
        raise ValueError("GPU-протокол отличается от проверенной конфигурации")
    cfg["runtime_backend"] = reference_backend()
    return cfg


def resolve_checkpoint(root, cfg, supplied=None):
    path = project_path(root, supplied if supplied else cfg["checkpoint"])
    weights = path / "model.safetensors" if cfg["model"] == "detr" else path
    if not weights.is_file():
        raise FileNotFoundError("Не найдены обученные веса: " + str(weights))
    actual_hash = digest(weights)
    if supplied is None and actual_hash != cfg["checkpoint_sha256"]:
        raise ValueError("Веса {} отличаются от проверенного checkpoint.".format(cfg["model"]))
    if supplied is None:
        for file, expected in cfg.get("metadata_sha256", {}).items():
            if digest(path / file) != expected:
                raise ValueError("Метаданные DETR изменились: " + file)
    return path, actual_hash


def publish(root, metric):
    write_json(root / "results/metrics.json", metric)
    path = root / "results/all_metrics.json"
    values = read_json(path) if path.is_file() else []
    by_model = {m["model"]: m for m in values}
    by_model[metric["model"]] = metric
    write_json(path, [by_model[name] for name in MODEL_NAMES if name in by_model])


def run_one(root, out, name, args):
    cfg = load_config(root, name, args.config)
    cfg = copy.deepcopy(cfg)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.mode == "train":
        if not 1 <= cfg["training"]["epochs"] <= 100:
            raise ValueError("Число эпох должно быть от 1 до 100")
        if cfg["model"] in ["faster_rcnn", "ssd", "detr", "efficientdet"] and cfg["training"]["batch"] != 1:
            raise ValueError("Текущий локальный цикл использует batch=1 и gradient_accumulation.")
        if cfg["model"] in ["faster_rcnn", "ssd", "detr", "efficientdet"]:
            accumulation = cfg["training"]["gradient_accumulation"]
            if not isinstance(accumulation, int) or not 1 <= accumulation <= 128:
                raise ValueError("gradient_accumulation должно быть целым числом от 1 до 128")
    write_json(out / "config_used.json", cfg)
    device = prepare_device(args.device, cfg["seed"], training=args.mode == "train")
    run_environment = environment()
    run_environment.update(device=str(device), backend_after_setup=backend_state())
    write_json(out / "environment.json", run_environment)
    data = project_path(root, cfg["data"])
    images, manifests = {}, {}
    required_splits = ["train", "val", args.split] if args.mode == "train" else [args.split]
    for split in dict.fromkeys(required_splits):
        images[split], manifests[split] = load_split(data, split)
        if cfg.get("verify_reference_dataset", True):
            verify_reference(manifests[split], root, split)
        write_json(out / (split + "_manifest.json"), manifests[split])
    supplied = args.checkpoint
    if args.mode == "train":
        from src.training.train import train_model
        supplied = str(train_model(cfg, root, out, device, images["train"], images["val"]))
        # Do not evaluate a dataset that changed while training.
        current, manifest = load_split(data, args.split)
        if manifest != manifests[args.split]:
            raise ValueError("Данные изменились во время обучения")
        images[args.split] = current
    checkpoint, weights_hash = resolve_checkpoint(root, cfg, supplied)
    detector = load_detector(name, checkpoint, device)
    backend_before = backend_state()
    predictions, elapsed = detector.predict(images[args.split])
    backend_after = backend_state()
    run_environment.update(backend_before_inference=backend_before, backend_after_inference=backend_after)
    write_json(out / "environment.json", run_environment)
    if not name.startswith("yolo"):
        for state in [backend_before, backend_after]:
            if {key: state[key] for key in reference_backend()} != reference_backend():
                raise ValueError("Настройки GPU изменились для " + name)
    metrics = evaluate(images[args.split], predictions, conf=.25)
    metrics.update(model=name, mAP=metrics["mAP50"], mAP_definition="mAP@IoU=0.50",
        split=args.split, images=len(images[args.split]), weights_sha256=weights_hash,
        checkpoint=relative(checkpoint, root), precision_recall_f1_confidence=.25,
        precision_recall_f1_iou=.5, precision_recall_f1_average="micro", ap_recall_points=101,
        max_detections_per_class_per_image=100, inference=cfg["inference"],
        runtime_protocol_version="gpu-v1.1", runtime_backend_before=backend_before,
        runtime_backend_after=backend_after, weights_finite=True, elapsed_seconds=elapsed, test_used_for_training=False,
        test_used_for_checkpoint_selection=False)
    for key in ["mAP", "mAP50", "mAP50_95", "precision", "recall", "f1"]:
        if metrics[key] is None or not 0 <= metrics[key] <= 1:
            raise ValueError("Неверная метрика " + key)
    if name == "efficientdet":
        write_json(out / "geometry_audit.json", detector.geometry_audit)
        metrics["geometry_cleanup"] = {key: sum(row[key] for row in detector.geometry_audit)
            for key in ["native_padding_rows", "clipped_rows", "degenerate_rows_removed", "exported_rows"]}
    write_json(out / "metrics.json", metrics)
    write_json(out / "predictions.json", predictions)
    portable_truth = [{key: value for key, value in sample.items() if key not in ["path", "label"]}
                      for sample in images[args.split]]
    write_json(out / "ground_truth.json", portable_truth)
    history_available = make_outputs(root, out, name, metrics, images[args.split], predictions)
    if not history_available:
        raise FileNotFoundError("Нет реального лога обучения {}; проверь папку results/reference/{}.".format(name, name))
    write_text(out / "log.txt", "Mode: {}\nModel: {}\nSplit: {}\nCheckpoint SHA256: {}\n"
        "mAP50: {:.8f}\nmAP50-95: {:.8f}\nPrecision: {:.8f}\nRecall: {:.8f}\nF1: {:.8f}\n".format(
            args.mode, name, args.split, weights_hash, metrics["mAP50"], metrics["mAP50_95"],
            metrics["precision"], metrics["recall"], metrics["f1"]))
    if args.split == "test":
        publish(root, metrics)
    del detector
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("{}: mAP50={:.4f}, mAP50-95={:.4f}, P={:.4f}, R={:.4f}, F1={:.4f}".format(
        name, metrics["mAP50"], metrics["mAP50_95"], metrics["precision"], metrics["recall"], metrics["f1"]), flush=True)
    return metrics
