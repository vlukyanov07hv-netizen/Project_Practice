#!/usr/bin/env python3
"""One entry point; default eval uses existing trained weights."""
import argparse
import multiprocessing
import sys
import traceback
from pathlib import Path
from src.constants import MODEL_ALIASES, MODEL_NAMES
from src.utils.files import new_run, write_json, write_text


def main():
    parser = argparse.ArgumentParser(description="Детекция дефектов металла. По умолчанию оценивает сохранённые веса.")
    parser.add_argument("--model", default="yolo_n", help="yolo_n, yolo_s, yolo_aug, faster_rcnn, ssd, detr, efficientdet, all")
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--config", default=None, help="Относительный путь к конфигурации YAML")
    parser.add_argument("--checkpoint", default=None, help="Другие обученные веса; путь относительно корня проекта")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto", help="auto, cuda:0 или cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Число эпох нового обучения")
    args = parser.parse_args()
    args.model = MODEL_ALIASES.get(args.model, args.model)
    if args.model not in MODEL_NAMES + ["all"]:
        parser.error("Неизвестная модель: " + args.model)
    if args.model == "all" and (args.config or args.checkpoint):
        parser.error("--config и --checkpoint требуют выбора одной модели")
    if args.epochs is not None and (args.mode != "train" or not 1 <= args.epochs <= 100):
        parser.error("--epochs от 1 до 100 используется только с --mode train")
    root = Path(__file__).resolve().parent
    out = new_run(root, args.model, args.mode)
    print("Результат: " + str(out), flush=True)
    report = {"mode": args.mode, "split": args.split, "status": "running", "models": [], "errors": [],
              "training_performed": args.mode == "train", "inputs_modified": False}
    from src.pipeline import run_one
    for name in MODEL_NAMES if args.model == "all" else [args.model]:
        target = out / name
        target.mkdir()
        try:
            report["models"].append(run_one(root, target, name, args))
        except (Exception, KeyboardInterrupt) as exc:
            report["errors"].append({"model": name, "type": type(exc).__name__, "message": str(exc)})
            detail = traceback.format_exc()
            write_text(target / "error.txt", detail)
            print(detail, flush=True)
            if isinstance(exc, KeyboardInterrupt):
                break
        write_json(out / "report.json", report)
    report["status"] = "completed" if not report["errors"] else "failed"
    if args.model == "all" and args.split == "test" and report["status"] == "completed":
        # The leaderboard uses the best measured model; checkpoint selection stays on val.
        write_json(root / "results/metrics.json", max(report["models"], key=lambda row: row["mAP50"]))
    write_json(out / "report.json", report)
    write_json(root / "results/latest_pipeline.json", {"folder": out.relative_to(root).as_posix(),
                                                      "status": report["status"]})
    print("Готово. Моделей: {}; ошибок: {}.".format(len(report["models"]), len(report["errors"])), flush=True)
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
