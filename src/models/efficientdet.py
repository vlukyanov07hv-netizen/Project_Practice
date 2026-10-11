"""Native EfficientDet-D0 inference with the verified preprocessing and box cleanup."""
import importlib.metadata
import time
from pathlib import Path

from src.models import efficientdet_geometry as support
from src.utils.files import digest, read_json
from src.utils.runtime import ensure_finite


def implementation():
    return read_json(Path(__file__).resolve().parents[2] / 'configs/efficientdet_implementation.json')


def verified_library():
    expected = implementation()
    for name, version in [('effdet', expected['effdet_version']), ('timm', expected['timm_version'])]:
        actual = importlib.metadata.version(name)
        if actual != version:
            raise ValueError('Версия {} отличается: ожидается {}, установлена {}'.format(name, version, actual))
    import effdet
    folder = Path(effdet.__file__).resolve().parent
    for name, sha256 in expected['implementation_sha256'].items():
        if digest(folder / name) != sha256:
            raise ValueError('Исходный код effdet изменился: ' + name)
    return effdet, expected


def preprocessing(base, expected):
    from effdet.data import resolve_input_config
    from effdet.data.transforms import transforms_coco_eval
    input_config = resolve_input_config({}, model_config=base.config)
    for key in ['input_size', 'mean', 'std']:
        input_config[key] = list(input_config[key])
    if input_config != expected['input_config']:
        raise ValueError('Preprocessing EfficientDet отличается от проверенного эксперимента')
    transform = transforms_coco_eval(img_size=input_config['input_size'][1:],
        interpolation=input_config['interpolation'], use_prefetcher=True,
        fill_color=input_config['fill_color'], mean=input_config['mean'], std=input_config['std'])
    return input_config, transform


class EfficientDetDetector:
    def __init__(self, bench, device, input_config, transform):
        self.name, self.model, self.device = 'efficientdet', bench, device
        self.input_config, self.transform = input_config, transform
        self.geometry_audit = []

    def predict(self, images):
        import torch
        from src.models.detectors import detection_rows as converter
        self.model.eval()
        predictions, self.geometry_audit = [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for index, sample in enumerate(images, 1):
                pixels, annotations, _ = support.transform_sample(sample, self.input_config, self.transform)
                tensor, target = support.device_batch(pixels, annotations, torch, self.device)
                # Only original size and scale are passed to the prediction bench.
                detections = self.model(tensor, {key: target[key] for key in ['img_size', 'img_scale']})
                if tuple(detections.shape) != (1, 300, 6):
                    raise ValueError('Неверная форма предсказаний EfficientDet: ' + sample['file'])
                audit = {}
                predictions.append(support.detection_rows(sample, detections[0].cpu().tolist(), converter, audit))
                self.geometry_audit.append(audit)
                if index == 1 or index % 50 == 0 or index == len(images):
                    print('efficientdet: {}/{}'.format(index, len(images)), flush=True)
                del pixels, annotations, tensor, target, detections
        return predictions, time.perf_counter() - started


def load_efficientdet(checkpoint, device):
    import torch
    effdet, expected = verified_library()
    base = effdet.create_model(expected['architecture'], pretrained=False, pretrained_backbone=False,
                              max_det_per_image=300)
    base.reset_head(num_classes=5)
    support.verify_class_head(base)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    loaded = base.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError('Обученные веса EfficientDet загружены не полностью')
    del state
    input_config, transform = preprocessing(base, expected)
    base.to(device)
    ensure_finite(base)
    bench = effdet.DetBenchPredict(base).to(device).eval()
    return EfficientDetDetector(bench, device, input_config, transform)
