"""
Скрипт для детекции СИЗ на видео с использованием обученной модели YOLOv8.

Использование:
    python detect_video.py --video input_video.mp4 --model models/ppe_model/weights/best.pt
"""

import cv2
import argparse
import os
from pathlib import Path
from ultralytics import YOLO


# Цвета для каждого класса СИЗ
CLASS_COLORS = {
    0: (0, 255, 0),      # helmet - зеленый (BGR)
    1: (0, 165, 255),   # vest - оранжевый (BGR)
    2: (255, 0, 0),     # gloves - синий (BGR)
}

# Имена классов
CLASS_NAMES = {
    0: "helmet",
    1: "vest",
    2: "gloves"
}

# Порог уверенности
CONFIDENCE_THRESHOLD = 0.5


def draw_detections(frame, results, conf_threshold=0.5):
    """
    Рисует детекции на кадре с кастомными цветами.
    
    Args:
        frame: Кадр изображения (numpy array)
        results: Результаты детекции от YOLOv8
        conf_threshold: Порог уверенности
    
    Returns:
        Кадр с нарисованными детекциями
    """
    # Получаем результаты детекции
    boxes = results[0].boxes
    
    # Проходим по всем детекциям
    for box in boxes:
        # Получаем координаты бокса
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Получаем класс и уверенность
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        
        # Пропускаем детекции с низкой уверенностью
        if conf < conf_threshold:
            continue
        
        # Получаем цвет для класса
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        
        # Получаем имя класса
        class_name = CLASS_NAMES.get(cls, f"class_{cls}")
        
        # Рисуем прямоугольник
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Формируем текст с именем класса и уверенностью
        label = f"{class_name}: {conf:.2f}"
        
        # Размер текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Получаем размер текста для фона
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Рисуем фон для текста
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Рисуем текст
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # Белый текст
            thickness
        )
    
    return frame


def detect_video(video_path, model_path, output_path, conf_threshold=0.5):
    """
    Обрабатывает видео и детектирует СИЗ.
    
    Args:
        video_path: Путь к входному видео
        model_path: Путь к обученной модели
        output_path: Путь для сохранения выходного видео
        conf_threshold: Порог уверенности
    """
    # Проверяем существование файлов
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео файл не найден: {video_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    # Создаем директорию для выходного файла
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    # Открываем видео
    print(f"Открытие видео: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Параметры видео:")
    print(f"  - Разрешение: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Всего кадров: {total_frames}")
    print(f"  - Порог уверенности: {conf_threshold}")
    print()
    
    # Создаем видеописатель
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("Начинаю обработку видео...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Детектируем объекты на кадре
        results = model(frame, conf=conf_threshold, device="cpu")
        
        # Рисуем детекции
        frame = draw_detections(frame, results, conf_threshold)
        
        # Записываем кадр в выходное видео
        out.write(frame)
        
        frame_count += 1
        
        # Прогресс
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Обработано кадров: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    # Освобождаем ресурсы
    cap.release()
    out.release()
    
    print(f"\nОбработка завершена!")
    print(f"Выходное видео сохранено: {output_path}")


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Детекция СИЗ на видео с использованием YOLOv8"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Путь к входному видео файлу"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppe_model/weights/best.pt",
        help="Путь к обученной модели (по умолчанию: models/ppe_model/weights/best.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/output_video.mp4",
        help="Путь для сохранения выходного видео (по умолчанию: output/output_video.mp4)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Порог уверенности (по умолчанию: 0.5)"
    )
    
    args = parser.parse_args()
    
    try:
        detect_video(args.video, args.model, args.output, args.conf)
    except Exception as e:
        print(f"Ошибка при обработке видео: {e}")
        raise


if __name__ == "__main__":
    main()

