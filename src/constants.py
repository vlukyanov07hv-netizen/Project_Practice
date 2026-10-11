CLASSES = ["deburring", "inclusion", "rolled_pit", "silk_spot", "waist_folding"]
MODEL_NAMES = ["yolo_n", "yolo_s", "yolo_aug", "faster_rcnn", "ssd", "detr", "efficientdet"]
MODEL_ALIASES = {"yolo": "yolo_n", "yolov8n": "yolo_n", "yolov8s": "yolo_s", "detr_fixed": "detr", "efficientdet_d0": "efficientdet"}
MODEL_TITLES = {"yolo_n": "YOLOv8n", "yolo_s": "YOLOv8s", "yolo_aug": "YOLOv8n aug",
                "faster_rcnn": "Faster R-CNN", "ssd": "SSD scratch", "detr": "DETR", "efficientdet": "EfficientDet-D0"}
REVISION = "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"
MODEL_ID = "facebook/detr-resnet-50"


def inference_protocol(name):
    if name == "efficientdet":
        return {"min_confidence": .001, "nms_iou": .5, "max_detections": 300,
                "batch": 1, "resize": "512x512 ResizePad; bicubic; top-left padding",
                "precision": "float32"}
    return {"min_confidence": .001, "nms_iou": None if name == "detr" else .7,
            "max_detections": 100 if name == "detr" else 300,
            "batch": 4 if name.startswith("yolo") else 1,
            "resize": "640x640 letterbox; rect=False" if name.startswith("yolo") else
                      "shortest_edge=512,longest_edge=1024" if name == "detr" else
                      "native 300x300" if name == "ssd" else "native min=800,max=1333",
            "precision": "float32"}
