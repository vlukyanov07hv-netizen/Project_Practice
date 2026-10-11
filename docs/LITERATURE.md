## Что изучалось?

В отчёте рассмотрено 16 научных источников. Их роль в выборе методов:

| № | Источник | Для чего использован |
|---|----------|---------------------|
| 1 | [Zhao et al. Object Detection with Deep Learning: A Review](https://arxiv.org/abs/1807.05511) | Обзор семейств детекторов. |
| 2 | [Girshick. Fast R-CNN](https://arxiv.org/abs/1504.08083) | Развитие двухэтапной детекции. |
| 3 | [Ren et al. Faster R-CNN](https://arxiv.org/abs/1506.01497) | Region Proposal Network и архитектура Faster R-CNN. |
| 4 | [Redmon et al. You Only Look Once](https://arxiv.org/abs/1506.02640) | Одноэтапная детекция и общий подход YOLO. |
| 5 | [Redmon, Farhadi. YOLO9000](https://arxiv.org/abs/1612.08242) | Развитие YOLO; статья не является описанием реализации YOLOv8. |
| 6 | [Liu et al. SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | MultiBox и детекция на нескольких масштабах. |
| 7 | [Lin et al. Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) | Многомасштабные признаки FPN. |
| 8 | [Lin et al. Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | Дисбаланс и Focal Loss. |
| 9 | [He et al. Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | Residual backbone ResNet. |
| 10 | [Tan, Le. EfficientNet](https://arxiv.org/abs/1905.11946) | Backbone семейства EfficientDet. |
| 11 | [Tan, Pang, Le. EfficientDet](https://arxiv.org/abs/1911.09070) | BiFPN и масштабирование детектора. |
| 12 | [Vaswani et al. Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Механизм attention и трансформеры. |
| 13 | [Carion et al. End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | DETR, object queries и set prediction. |
| 14 | [Lin et al. Microsoft COCO](https://arxiv.org/abs/1405.0312) | Предобучение и общие принципы оценки детекции. |
| 15 | [Shorten, Khoshgoftaar. A survey on Image Data Augmentation for Deep Learning](https://doi.org/10.1186/s40537-019-0197-0) | Обзор аугментаций изображений. |
| 16 | [Tabernik et al. Segmentation-Based Deep-Learning Approach for Surface-Defect Detection](https://arxiv.org/abs/1903.08536) | Промышленные дефекты и ограниченные данные; DOI: 10.1007/s10845-019-01476-x. |

## Какая документация?

- [c5data2 на Kaggle](https://www.kaggle.com/datasets/niclinn/c5data2) — источник датасета.
- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) — фактическая реализация YOLOv8.
- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) — рассмотренный альтернативный промышленный датасет.
- [Hugging Face Transformers v4.44.2 / DETR](https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/detr) — processor, форматы target и модель DETR.
- [EfficientDet-PyTorch, авторская реализация](https://github.com/rwightman/efficientdet-pytorch) — библиотека effdet и модель tf_efficientdet_d0.
- [Версии PyTorch](https://pytorch.org/get-started/previous-versions/) — установка согласованных torch 2.5.1 / torchvision 0.20.1.

Полные библиографические записи и обсуждение приведены в `Report_metal_surface_defects.docx`.
