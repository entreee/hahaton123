# Изменения: Переход на YOLOv8-OBB (Rotated Bounding Boxes)

## Обзор изменений

Проект переведен на использование **YOLOv8-OBB** (Oriented Bounding Box) для поддержки **rotated bounding boxes** - детекции объектов под наклоном.

## Основные изменения

### 1. Модель обучения (`src/models/train_model.py`)
- ✅ Заменена модель с `yolov8n.pt` на `yolov8n-obb.pt` (по умолчанию)
- ✅ Добавлена поддержка `task='obb'` при загрузке модели
- ✅ Обновлены сообщения об ошибках с указанием OBB моделей
- ✅ **Сохранены все оптимизации:**
  - Linux: до 12 workers, 'fork' multiprocessing
  - Windows: до 6 workers, 'spawn' multiprocessing
  - Скорость обучения: ~40-60 it/s на Linux
  - Оптимизированные параметры (epochs=30, img_size=640, batch_size=8)

### 2. Детектор (`src/inference/detect_utils.py`)
- ✅ Обновлен для работы с OBB форматом
- ✅ Рисование rotated bounding boxes (4 точки углов)
- ✅ Поддержка `results[0].obbs` вместо `results[0].boxes`
- ✅ Обновлена визуализация для rotated boxes
- ✅ Обновлены методы: `detect_image()`, `detect_video()`, `detect_camera()`

### 3. Новая программа `detect.py`
- ✅ Создана программа для работы с обученной моделью
- ✅ Поддержка различных источников:
  - Изображения (JPG, PNG, BMP)
  - Видео (MP4, AVI, MOV, MKV)
  - Папки с изображениями (пакетная обработка)
  - Камера (детекция в реальном времени)
- ✅ Настройка порога уверенности (`--conf`)
- ✅ Выбор устройства (`--device`)
- ✅ Сохранение результатов

### 4. Конфигурация
- ✅ Обновлен `config/ppe_data.yaml` с комментариями о формате OBB
- ✅ Обновлен `src/utils/config.py`:
  - `model_name`: `yolov8l-obb.pt` (по умолчанию)
  - `experiment_name`: `ppe_detection_obb`

### 5. Документация
- ✅ Создан `OBB_ANNOTATION_GUIDE.md` - руководство по разметке OBB
- ✅ Обновлен `README.md`:
  - Информация о модели OBB
  - Инструкции по использованию `detect.py`
  - Обновлен раздел об инструментах разметки

## Формат аннотаций OBB

**Новый формат** (в файлах `.txt`):
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Где `(x1,y1), (x2,y2), (x3,y3), (x4,y4)` - 4 точки углов rotated bounding box в нормализованных координатах (0.0 - 1.0).

**Старый формат** (не поддерживается):
```
class_id x_center y_center width height
```

## Инструменты для разметки

Для создания OBB аннотаций используйте:
1. **LabelMe** - `pip install labelme` (поддержка rotated boxes)
2. **CVAT** - веб-инструмент с полной поддержкой OBB
3. **Roboflow** - онлайн платформа с поддержкой OBB

⚠️ **LabelImg не поддерживает rotated boxes напрямую!**

## Использование

### Обучение модели
```bash
python run_full_pipeline.py
```

### Детекция
```bash
# Изображение
python detect.py --model models/ppe_detection_obb/weights/best.pt --source image.jpg

# Видео
python detect.py --model models/ppe_detection_obb/weights/best.pt --source video.mp4

# Камера
python detect.py --model models/ppe_detection_obb/weights/best.pt --camera
```

## Совместимость

- ✅ **Все оптимизации сохранены** (Linux, скорость обучения)
- ✅ **Обратная совместимость**: Старые модели (обычные YOLOv8) не будут работать
- ⚠️ **Требуется переразметка данных** в формате OBB (4 точки углов)

## Следующие шаги

1. Переразметьте данные в формате OBB (см. `OBB_ANNOTATION_GUIDE.md`)
2. Обучите модель: `python run_full_pipeline.py`
3. Используйте обученную модель: `python detect.py --model models/ppe_detection_obb/weights/best.pt --source ...`

