"""Explicit new EfficientDet training using the verified loop and cached COCO weights."""
import copy
import importlib.metadata

from src.models.efficientdet import preprocessing, verified_library
from src.models import efficientdet_geometry as support
from src.training.efficientdet_loop import cpu_weights, create_optimizer, make_plot, train_and_select
from src.utils.files import digest, project_path, read_json, save_torch, write_json
from src.utils.runtime import backend_state, ensure_finite


def train_efficientdet(cfg, root, out, device, train_images, val_images):
    import numpy as np
    import torch
    if device.type != 'cuda':
        raise ValueError('Проверенный локальный цикл EfficientDet требует CUDA; используй --device cuda:0')
    if cfg['training']['optimizer'] != 'AdamW' or cfg['training']['batch'] != 1 or cfg['training']['amp']:
        raise ValueError('Проверенный цикл EfficientDet использует AdamW, batch=1 и float32 без AMP')
    effdet, expected = verified_library()
    pretrained = project_path(root, cfg['pretrained_checkpoint'])
    if digest(pretrained) != cfg['pretrained_checkpoint_sha256']:
        raise ValueError('Кэш начальных COCO весов EfficientDet изменился')
    base = effdet.create_model(cfg['architecture'], pretrained=False, pretrained_backbone=False,
                              max_det_per_image=300)
    state = torch.load(pretrained, map_location='cpu', weights_only=True)
    base.load_state_dict(state, strict=True)
    base.reset_head(num_classes=5)
    for name, value in state.items():
        if not name.startswith('class_net.') and not torch.equal(value, base.state_dict()[name]):
            raise ValueError('Замена головы изменяет backbone/box-head: ' + name)
    del state
    support.verify_class_head(base)
    input_config, transform = preprocessing(base, expected)
    base.to(device)
    ensure_finite(base)
    bench = effdet.DetBenchTrain(base, create_labeler=True).to(device)
    loop_cfg = copy.deepcopy(cfg)
    loop_cfg['inference'] = {
        'minimum_export_confidence': .001, 'native_nms_iou': .5, 'max_det_per_image': 300,
        'precision_recall_confidence': .25, 'precision_recall_iou': .5,
        'ap': 'shared_101_point_macro_iou_0.50_to_0.95'}
    binding = {
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'effdet', 'timm']},
        'runtime': backend_state(), 'implementation': expected,
        'data': {split: read_json(out / (split + '_manifest.json')) for split in ['train', 'val']}}
    report = {'optimizer_updates_this_invocation': 0, 'training_performed': False}
    optimizer = create_optimizer(base, loop_cfg, torch)
    history, epoch, score = train_and_select(base, bench, optimizer, train_images, val_images,
        loop_cfg, binding, None, out, report, input_config, transform, device, np, torch, False)
    ensure_finite(base)
    checkpoint = out / 'weights/best.pth'
    save_torch(checkpoint, cpu_weights(base), torch)
    write_json(out / 'selection.json', {'best_epoch': epoch, 'validation_mAP50_95': score,
        'criterion': 'maximum validation mAP50_95; earlier epoch on exact tie',
        'test_used': False, 'checkpoint_sha256': digest(checkpoint)})
    write_json(out / 'efficientdet_training_report.json', report)
    make_plot(out, history)
    return checkpoint
