"""
Скрипт для детекции СИЗ в реальном времени с веб-камеры или IP-камеры.

Использование:
    python detect_camera.py --model models/ppe_model/weights/best.pt --source 0
    python detect_camera.py --model models/ppe_model/weights/best.pt --source rtsp://192.168.1.100:554/stream
"""

import cv2
import argparse
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
    boxes = results[0].boxes
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        
        if conf < conf_threshold:
            continue
        
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        class_name = CLASS_NAMES.get(cls, f"class_{cls}")
        
        # Рисуем прямоугольник
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Формируем текст
        label = f"{class_name}: {conf:.2f}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
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
            (255, 255, 255),
            thickness
        )
    
    return frame


def detect_camera(model_path, source, conf_threshold=0.5):
    """
    Обрабатывает видео с камеры в реальном времени.
    
    Args:
        model_path: Путь к обученной модели
        source: Источник видео (0 для веб-камеры, URL для IP-камеры)
        conf_threshold: Порог уверенности
    """
    import os
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    # Загружаем модель
    print(f"Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    # Определяем источник видео
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    
    print(f"Подключение к источнику: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть источник видео: {source}")
    
    print("Детекция запущена. Нажмите 'q' для выхода.")
    print(f"Порог уверенности: {conf_threshold}")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Не удалось получить кадр. Проверьте подключение.")
            break
        
        # Детектируем объекты
        results = model(frame, conf=conf_threshold, device="cpu")
        
        # Рисуем детекции
        frame = draw_detections(frame, results, conf_threshold)
        
        # Добавляем информацию о FPS
        frame_count += 1
        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Показываем кадр
        cv2.imshow("PPE Detection", frame)
        
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nДетекция остановлена.")


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Детекция СИЗ в реальном времени с камеры"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppe_model/weights/best.pt",
        help="Путь к обученной модели (по умолчанию: models/ppe_model/weights/best.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Источник видео: 0 для веб-камеры, URL для IP-камеры (по умолчанию: 0)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Порог уверенности (по умолчанию: 0.5)"
    )
    
    args = parser.parse_args()
    
    try:
        detect_camera(args.model, args.source, args.conf)
    except KeyboardInterrupt:
        print("\n\nДетекция прервана пользователем.")
    except Exception as e:
        print(f"Ошибка при детекции: {e}")
        raise


if __name__ == "__main__":
    main()

