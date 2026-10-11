import os
import shutil
import xml.etree.ElementTree as ET
import random
from collections import Counter

# --- НАСТРОЙКИ ---
DATASET_ROOT = r"G:\Project\Dataset\metal_defects"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "label")
PROJECT_DATA_PATH = r"G:\Project\cv-project\data\raw"

# --- КЛАССЫ: имя папки -> (имя класса, id) ---
FOLDER_TO_CLASS = {
    "deburring": ("deburring", 0),
    "inclusion": ("inclusion", 1),
    "rolled_pit": ("rolled_pit", 2),
    "silk_spot": ("silk_spot", 3),
    "waist folding": ("waist_folding", 4),
}

CLASSES = ["deburring", "inclusion", "rolled_pit", "silk_spot", "waist_folding"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
random.seed(42)


def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def parse_xml(xml_path, class_id):
    """Читает Pascal VOC XML. Возвращает список объектов (cid, xc, yc, w, h)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return []

    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
        except (AttributeError, ValueError):
            continue
        objects.append((class_id, *convert_bbox((img_w, img_h), (xmin, xmax, ymin, ymax))))

    return objects


def find_pairs():
    """
    Идём по подпапкам images/. Для каждой картинки:
      - класс = из имени папки
      - XML ищем по base-имени в label/
    """
    pairs = []
    missing_xml = []
    unknown_folders = []

    for folder_name in os.listdir(IMAGES_DIR):
        folder_path = os.path.join(IMAGES_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Определяем класс по имени папки
        key = folder_name.lower().strip()
        # Нормализуем: заменяем пробелы/дефисы на _
        key_norm = key.replace(" ", "_").replace("-", "_")

        # Ищем в словаре
        class_info = None
        for k, v in FOLDER_TO_CLASS.items():
            if k.lower().replace(" ", "_").replace("-", "_") == key_norm:
                class_info = v
                break

        if class_info is None:
            unknown_folders.append(folder_name)
            continue

        class_name, class_id = class_info
        print(f"Папка '{folder_name}' -> класс '{class_name}' (id={class_id})")

        for f in os.listdir(folder_path):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            base = os.path.splitext(f)[0]
            xml_path = os.path.join(LABELS_DIR, base + ".xml")

            if not os.path.exists(xml_path):
                missing_xml.append(f"{folder_name}/{f}")
                continue

            pairs.append((os.path.join(folder_path, f), xml_path, class_id))

    return pairs, missing_xml, unknown_folders


if __name__ == "__main__":
    pairs, missing, unknown = find_pairs()
    print(f"\nНайдено пар (картинка, XML, класс): {len(pairs)}")
    if unknown:
        print(f"Неизвестные папки: {unknown}")
    if missing:
        print(f"Картинок без XML: {len(missing)}. Примеры: {missing[:5]}")

    if not pairs:
        print("Нечего конвертировать.")
        exit(1)

    # Диагностика: сколько объектов и картинок по классам
    cnt_objects = Counter()
    cnt_images = Counter()
    for img, xml, cid in pairs:
        try:
            objs = parse_xml(xml, cid)
            cnt_objects[CLASSES[cid]] += len(objs)
            cnt_images[CLASSES[cid]] += 1
        except Exception as e:
            print(f"  [!] Ошибка в {os.path.basename(xml)}: {e}")

    print("\nСтатистика по классам:")
    print(f"  {'Класс':<15} {'Картинок':>10} {'Объектов':>10}")
    for name in CLASSES:
        print(f"  {name:<15} {cnt_images[name]:>10} {cnt_objects[name]:>10}")

    ans = input("\nПродолжить конвертацию? (y/n): ").strip().lower()
    if ans != "y":
        exit(0)

    random.shuffle(pairs)
    total = len(pairs)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)

    splits = {
        "train": pairs[:train_size],
        "val": pairs[train_size:train_size + val_size],
        "test": pairs[train_size + val_size:],
    }

    for split in splits:
        os.makedirs(os.path.join(PROJECT_DATA_PATH, "images", split), exist_ok=True)
        os.makedirs(os.path.join(PROJECT_DATA_PATH, "labels", split), exist_ok=True)

    for split, split_pairs in splits.items():
        print(f"\n--- {split}: {len(split_pairs)} ---")
        saved = 0
        for img_path, xml_path, cid in split_pairs:
            objs = parse_xml(xml_path, cid)
            if not objs:
                continue

            filename = os.path.basename(img_path)
            base = os.path.splitext(filename)[0]

            shutil.copy2(img_path, os.path.join(PROJECT_DATA_PATH, "images", split, filename))
            with open(os.path.join(PROJECT_DATA_PATH, "labels", split, base + ".txt"), "w") as f:
                for obj in objs:
                    f.write(f"{obj[0]} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f}\n")
            saved += 1
        print(f"  Сохранено: {saved}")

    print("\n✅ Готово!")