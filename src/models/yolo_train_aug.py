from ultralytics import YOLO
import random, numpy as np, torch
import multiprocessing


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="G:/Project/cv-project/configs/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        seed=42,
        project="G:/Project/cv-project/results",
        name="yolo_augmented",
        workers=0,
        # --- Аугментация ---
        mosaic=1.0,
        mixup=0.1,
        degrees=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )

    print("Обучение завершено!")
    print("Метрики:", results.results_dict)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()