#!/usr/bin/env python3
"""Restore verified release assets without overwriting differing existing files."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import tempfile
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def target_path(root, name):
    relative = PurePosixPath(name)
    if ('\\' in name or relative.is_absolute() or PureWindowsPath(name).drive
            or '..' in relative.parts or not relative.parts or any(':' in part for part in relative.parts)
            or relative.parts[0] not in ('data', 'results')):
        raise ValueError('Недопустимый путь в архиве: ' + name)
    target = root.joinpath(*relative.parts)
    for part in (target, *target.parents):
        if part == root:
            break
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('В пути есть ссылка или junction: ' + name)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Путь выходит за пределы проекта: ' + name)
    return target


def restore(root, assets, manifest):
    if manifest.get('status') != 'ready' or not manifest.get('assets'):
        raise ValueError('Комплект Releases не подтверждён. Сначала исправь CHECK_ASSEMBLY.txt и собери его заново.')
    plans = []
    seen = set()
    for row in manifest['assets']:
        filename = row['name']
        if Path(filename).name != filename or '\\' in filename:
            raise ValueError('Неверное имя архива: ' + filename)
        path = assets / filename
        if not path.is_file():
            if row.get('optional_for_evaluation'):
                print('[ПРОПУСК] {}: нужен только для нового обучения EfficientDet.'.format(filename), flush=True)
                continue
            raise FileNotFoundError('Нет архива ' + str(path))
        if digest(path) != row['sha256']:
            raise ValueError('Контрольная сумма архива отличается: ' + filename)
        with zipfile.ZipFile(path) as archive:
            headers = [item for item in archive.infolist() if not item.is_dir()]
            expected_files = row['files']
            if len(headers) != len(expected_files) or {item.filename for item in headers} != set(expected_files):
                raise ValueError('Состав архива отличается: ' + filename)
            for item in headers:
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError('Ссылка в архиве: ' + item.filename)
                key = item.filename.casefold()
                if key in seen:
                    raise ValueError('Повторяющийся путь: ' + item.filename)
                seen.add(key)
                target = target_path(root, item.filename)
                spec = expected_files[item.filename]
                if item.file_size != spec['size_bytes']:
                    raise ValueError('Размер файла отличается: ' + item.filename)
                if target.exists() and (not target.is_file() or digest(target) != spec['sha256']):
                    raise FileExistsError('Отказ от перезаписи отличающегося файла: ' + str(target))
        plans.append((path, row))
    for path, row in plans:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                target = target_path(root, item.filename)
                if target.is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                                                     suffix='.tmp', delete=False) as stream:
                        pending = Path(stream.name)
                        with archive.open(item) as source:
                            shutil.copyfileobj(source, stream, length=1024 * 1024)
                    if digest(pending) != row['files'][item.filename]['sha256']:
                        raise ValueError('Контрольная сумма распакованного файла отличается: ' + item.filename)
                    if target.exists():
                        raise FileExistsError('Файл появился во время восстановления: ' + str(target))
                    os.replace(pending, target)
                finally:
                    if pending and pending.exists():
                        pending.unlink()
        print('[OK] ' + path.name, flush=True)
    print('Данные и веса восстановлены. Можно запускать main.py в режиме eval.')


def main():
    parser = argparse.ArgumentParser(description='Восстановление датасета и весов из проверенных архивов Releases.')
    parser.add_argument('--assets', default='release_assets', help='Папка со скачанными ZIP; относительно корня проекта.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assets = Path(args.assets).expanduser()
    if not assets.is_absolute():
        assets = root / assets
    try:
        path = root / 'release_manifest.json'
        if not path.is_file():
            raise FileNotFoundError('Нет release_manifest.json. Его создаёт сборщик Windows вместе с архивами весов и данных.')
        restore(root, assets, json.loads(path.read_text(encoding='utf-8-sig')))
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print('[ОШИБКА] ' + str(exc), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
