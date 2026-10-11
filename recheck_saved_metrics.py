#!/usr/bin/env python3
"""Recalculate AP and P/R/F1 from saved predictions. NumPy only; no Torch."""
import json
import math
from pathlib import Path


def main():
    try:
        from src.constants import MODEL_NAMES
        from src.evaluation.metrics import evaluate
    except ModuleNotFoundError as exc:
        if exc.name == 'numpy':
            print('Для проверки нужен NumPy: python -m pip install numpy==2.0.2')
            return 1
        raise
    root = Path(__file__).resolve().parent
    checked = ('mAP50', 'mAP50_95', 'precision', 'recall', 'f1',
               'precision_macro', 'recall_macro', 'f1_macro', 'tp', 'fp', 'fn', 'per_class')
    def compare(actual, expected, path=''):
        if isinstance(expected, dict):
            for key, value in expected.items():
                compare(actual[key], value, path + '/' + key)
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise ValueError('Число элементов отличается: ' + path)
            for index, value in enumerate(expected):
                compare(actual[index], value, path + '/' + str(index))
        elif isinstance(expected, float):
            if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
                raise ValueError('Метрика отличается: ' + path)
        elif actual != expected:
            raise ValueError('Значение отличается: ' + path)
    failures = []
    for model in MODEL_NAMES:
        try:
            folder = root / 'results/final' / model
            read = lambda name: json.loads((folder / name).read_text(encoding='utf-8-sig'))
            expected = read('metrics.json')
            actual = evaluate(read('ground_truth.json'), read('predictions.json'), conf=0.25)
            compare({key: actual[key] for key in checked}, {key: expected[key] for key in checked})
            print('[OK] {}: mAP50={:.8f}, mAP50–95={:.8f}'.format(model, actual['mAP50'], actual['mAP50_95']), flush=True)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append(model)
            print('[ОШИБКА] {}: {}'.format(model, exc), flush=True)
    print('Проверено: {} из {}. Только сохранённые предсказания; нейросети и обучение не запускались.'.format(
        len(MODEL_NAMES) - len(failures), len(MODEL_NAMES)))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
