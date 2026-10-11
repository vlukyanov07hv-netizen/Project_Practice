"""Common prediction interface; native resize/postprocessing remains architecture-specific."""
import math
import tempfile
import time
from pathlib import Path
from PIL import Image
from src.constants import CLASSES, MODEL_ID, REVISION
from src.utils.runtime import ensure_finite


def build_torchvision(name, pretrained=False):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.ops.misc import FrozenBatchNorm2d
    if name == "ssd":
        return ssd300_vgg16(weights=None, weights_backbone=None, num_classes=6,
                           score_thresh=.001, nms_thresh=.7, detections_per_img=300)
    if pretrained:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT", box_score_thresh=.001,
                                       box_nms_thresh=.7, box_detections_per_img=300)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 6)
        return model
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=6,
                                  box_score_thresh=.001, box_nms_thresh=.7, box_detections_per_img=300)
    def replace_bn(module):
        for key, child in list(module.named_children()):
            if isinstance(child, torch.nn.BatchNorm2d):
                module.add_module(key, FrozenBatchNorm2d(child.num_features, eps=0.))
            else:
                replace_bn(child)
    replace_bn(model)
    return model


def detection_rows(file, labels, scores, boxes, offset=0):
    detections = []
    if not (len(labels) == len(scores) == len(boxes)):
        raise ValueError("Несовпадающие массивы предсказания")
    for label, score, box in zip(labels, scores, boxes):
        cid, score = int(label) - offset, float(score)
        box = [float(value) for value in box]
        if float(label) != int(label) or cid not in range(5) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Неверный класс или confidence: " + file)
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] < box[0] or box[3] < box[1]:
            raise ValueError("Неверная геометрия предсказания: " + file)
        if score >= .001:
            detections.append({"class_id": cid, "score": score, "box": box})
    return {"file": file, "detections": detections}


class Detector:
    def __init__(self, name, model, device, processor=None):
        self.name, self.model, self.device, self.processor = name, model, device, processor

    def predict(self, images):
        import torch
        started = time.perf_counter()
        predictions = []
        if self.name.startswith("yolo"):
            # A text file uses the same image loader / batches as the verified run.
            # A Python list may be treated as one in-memory batch by Ultralytics.
            with tempfile.TemporaryDirectory(prefix="cv_pipeline_") as folder:
                source = Path(folder) / "images.txt"
                source.write_text("\n".join(s["path"] for s in images) + "\n", encoding="utf-8")
                results = self.model.predict(source=str(source), stream=True,
                    imgsz=640, conf=.001, iou=.7, max_det=300, batch=4, rect=False,
                    device=str(self.device), half=False, verbose=False, save=False)
                for index, output in enumerate(results, 1):
                    if index > len(images):
                        raise ValueError("YOLO вернул лишнее изображение")
                    sample = images[index - 1]
                    if Path(output.path).resolve() != Path(sample["path"]).resolve():
                        raise ValueError("YOLO вернул другое изображение")
                    if tuple(output.orig_shape) != (sample["height"], sample["width"]):
                        raise ValueError("YOLO изменил исходные размеры")
                    boxes = output.boxes
                    predictions.append(detection_rows(sample["file"], boxes.cls.cpu().tolist(),
                        boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist()))
                    if index == 1 or index % 50 == 0 or index == len(images):
                        print("{}: {}/{}".format(self.name, index, len(images)), flush=True)
            if len(predictions) != len(images):
                raise ValueError("YOLO вернул неполный список изображений")
        else:
            from src.dataset.detection import torchvision_sample
            self.model.eval()
            with torch.inference_mode():
                for index, sample in enumerate(images, 1):
                    if self.name == "detr":
                        with Image.open(sample["path"]) as im:
                            encoded = self.processor(images=im.convert("RGB"), return_tensors="pt")
                        batch = {key: encoded[key].to(self.device) for key in ["pixel_values", "pixel_mask"]}
                        result = self.model(**batch)
                        if not torch.isfinite(result.logits).all().item() or not torch.isfinite(result.pred_boxes).all().item():
                            raise ValueError("NaN/Inf в DETR")
                        target_sizes = torch.tensor([[sample["height"], sample["width"]]], device=self.device)
                        result = self.processor.post_process_object_detection(
                            result, threshold=.001, target_sizes=target_sizes)[0]
                        offset = 0
                    else:
                        tensor, _ = torchvision_sample(sample, self.device)
                        result = self.model([tensor])[0]
                        offset = 1
                    predictions.append(detection_rows(sample["file"], result["labels"].detach().cpu().tolist(),
                        result["scores"].detach().cpu().tolist(), result["boxes"].detach().cpu().tolist(), offset))
                    if index == 1 or index % 50 == 0 or index == len(images):
                        print("{}: {}/{}".format(self.name, index, len(images)), flush=True)
        return predictions, time.perf_counter() - started


def load_detector(name, checkpoint, device):
    if name == "efficientdet":
        from src.models.efficientdet import load_efficientdet
        return load_efficientdet(checkpoint, device)
    import torch
    if name.startswith("yolo"):
        from ultralytics import YOLO
        model = YOLO(str(checkpoint))
        names = model.names
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
        if names != CLASSES:
            raise ValueError("Порядок классов YOLO отличается")
        ensure_finite(model.model)
        return Detector(name, model, device)
    if name == "detr":
        from transformers import DetrForObjectDetection, DetrImageProcessor
        processor = DetrImageProcessor.from_pretrained(checkpoint, local_files_only=True)
        if processor.size != {"shortest_edge": 512, "longest_edge": 1024}:
            raise ValueError("Размеры processor DETR отличаются от проверенного протокола")
        model = DetrForObjectDetection.from_pretrained(checkpoint, local_files_only=True,
            use_safetensors=True, use_pretrained_backbone=False)
        if model.config.num_labels != 5 or model.class_labels_classifier.out_features != 6:
            raise ValueError("Неверное число классов DETR")
        if [model.config.id2label[i] for i in range(5)] != CLASSES:
            raise ValueError("Порядок классов DETR отличается")
    else:
        model = build_torchvision(name)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        processor = None
    ensure_finite(model)
    model.to(device).eval()
    return Detector(name, model, device, processor)


def fresh_detector(cfg, root, device):
    name = cfg["model"]
    if name.startswith("yolo"):
        from ultralytics import YOLO
        return Detector(name, YOLO(cfg["architecture"]), device)
    if name == "detr":
        from transformers import DetrForObjectDetection, DetrImageProcessor
        from src.models.detr_utils import validate_loading
        cache = str(root / "results/cache/huggingface")
        processor = DetrImageProcessor.from_pretrained(MODEL_ID, revision=REVISION,
            size=cfg["training"]["size"], cache_dir=cache)
        model, info = DetrForObjectDetection.from_pretrained(MODEL_ID, revision=REVISION, num_labels=5,
            id2label=dict(enumerate(CLASSES)), label2id={name: i for i, name in enumerate(CLASSES)},
            ignore_mismatched_sizes=True, use_safetensors=True, use_pretrained_backbone=False,
            output_loading_info=True, cache_dir=cache)
        validate_loading(info)
    else:
        model = build_torchvision(name, pretrained=name == "faster_rcnn")
        processor = None
    ensure_finite(model)
    model.to(device)
    return Detector(name, model, device, processor)
