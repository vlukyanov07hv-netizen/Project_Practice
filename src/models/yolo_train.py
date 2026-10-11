from ultralytics import YOLO
import random, numpy as np, torch
import multiprocessing

def main():
    # Фиксация seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = YOLO(r"G:\Project\cv-project\yolov8s.pt")

    results = model.train(
        data="G:/Project/cv-project/configs/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,          # если GPU не хватает — поставь 8
        seed=42,
        project="G:/Project/cv-project/results",
        name="yolo_small",
        workers=4          # Количество потоков для загрузки данных
    )

    print("Обучение завершено!")
    print("Метрики:", results.results_dict)

if __name__ == '__main__':
    multiprocessing.freeze_support() # Обязательно для Windows
    main()