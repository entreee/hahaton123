"""
Скрипт для обучения модели YOLOv8 на датасете СИЗ.

Использование:
    python train.py
"""

from ultralytics import YOLO
import os
from pathlib import Path


def train_model():
    """
    Обучает модель YOLOv8 на датасете СИЗ.
    
    Модель будет обучена на CPU и сохранена в models/ppe_model/
    """
    # Проверяем наличие конфигурационного файла
    config_path = "config/ppe_data.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Конфигурационный файл не найден: {config_path}\n"
            "Убедитесь, что файл config/ppe_data.yaml существует."
        )
    
    # Создаем директорию для моделей, если её нет
    Path("models").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Начало обучения модели YOLOv8 для детекции СИЗ")
    print("=" * 60)
    print(f"Конфигурация: {config_path}")
    print("Устройство: CPU")
    print("=" * 60)
    
    # Загружаем предобученную модель YOLOv8n (nano - самая легкая)
    print("\nЗагрузка предобученной модели yolov8n.pt...")
    model = YOLO("yolov8n.pt")
    
    # Параметры обучения
    print("\nПараметры обучения:")
    print("  - Эпохи: 30")
    print("  - Размер изображения: 640x640")
    print("  - Устройство: CPU")
    print("  - Проект: models")
    print("  - Имя модели: ppe_model")
    print("\nНачинаю обучение...\n")
    
    # Обучаем модель
    results = model.train(
        data=config_path,          # Путь к конфигурационному файлу
        epochs=30,                 # Количество эпох
        imgsz=640,                 # Размер изображения
        device="cpu",              # Использовать CPU
        project="models",          # Директория проекта
        name="ppe_model",          # Имя модели
        batch=16,                  # Размер батча (можно уменьшить для CPU)
        patience=10,               # Ранняя остановка после 10 эпох без улучшения
        save=True,                 # Сохранять чекпоинты
        plots=True                 # Генерировать графики обучения
    )
    
    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print("=" * 60)
    print(f"Лучшая модель сохранена в: models/ppe_model/weights/best.pt")
    print(f"Последняя модель сохранена в: models/ppe_model/weights/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем.")
    except Exception as e:
        print(f"\nОшибка при обучении: {e}")
        raise

