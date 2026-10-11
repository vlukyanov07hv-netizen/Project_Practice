#!/usr/bin/env python3
"""Show the measured final results using only the Python standard library."""
import csv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    with (root / 'comparison_table.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    print('Детекция дефектов металла: общий test, 162 изображения / 226 объектов.\n')
    print('{:<27} {:>9} {:>10} {:>10} {:>10} {:>10}'.format('Модель', 'mAP50', 'mAP50–95', 'Precision', 'Recall', 'F1'))
    for row in rows:
        print('{:<27} {:>9.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}'.format(
            row['title'], *(float(row[key]) for key in ('mAP50', 'mAP50_95', 'precision', 'recall', 'f1'))))
    print('\nГотовые графики и примеры: results/final/<model>/.')
    print('Это просмотр сохранённых результатов. Обучение и оценка нейросетей не запускались.')


if __name__ == '__main__':
    main()
