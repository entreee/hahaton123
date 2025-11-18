"""
Автоматическая предразметка кадров с использованием предобученной YOLOv8.

Использование:
    python auto_prelabel.py

Скрипт:
1. Загружает предобученную модель yolov8n.pt
2. Детектирует людей на всех изображениях в data/images/train/
3. Создает предварительные bounding boxes вокруг людей
4. Сохраняет разметку в data/labels/train/ в формате YOLO
5. Все детекции людей помечаются как класс 0 (helmet) для быстрой корректировки

После запуска откройте LabelImg и просто:
- Поменяйте класс с 0 на 1 для жилетов
- Удалите ненужные box
- Добавьте недостающие каски/жилеты

Это ускорит разметку в 3-5 раз!
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2


def auto_prelabel(images_dir="data/images/train", labels_dir="data/labels/train", conf_threshold=0.3):
    """
    Автоматически предразмечает изображения с использованием предобученной модели.
    
    Args:
        images_dir: Папка с изображениями
        labels_dir: Папка для сохранения разметки
        conf_threshold: Порог уверенности для детекции людей
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # Создаем папку для разметки
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(ext))
        image_files.extend(images_path.glob(ext.upper()))
    
    if len(image_files) == 0:
        print(f"❌ Не найдено изображений в {images_dir}")
        print(f"Поместите изображения в {images_dir}/ и запустите снова.")
        return
    
    print("=" * 60)
    print("АВТОМАТИЧЕСКАЯ ПРЕДРАЗМЕТКА КАДРОВ")
    print("=" * 60)
    print(f"Папка с изображениями: {images_dir}")
    print(f"Папка для разметки: {labels_dir}")
    print(f"Порог уверенности: {conf_threshold}")
    print(f"Найдено изображений: {len(image_files)}")
    print("=" * 60)
    
    # Загружаем предобученную модель
    print("\nЗагрузка предобученной модели YOLOv8n...")
    model = YOLO("yolov8n.pt")
    
    # COCO классы - человек это класс 0
    person_class = 0  # 'person' в COCO dataset
    
    total_annotations = 0
    processed = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Обработка: {image_file.name}")
        
        # Загружаем изображение
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  ⚠️  Не удалось загрузить: {image_file.name}")
            continue
        
        # Детектируем объекты
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Получаем bounding boxes
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"  - Нет детекций")
            processed += 1
            continue
        
        # Фильтруем только людей (класс 0 в COCO)
        person_boxes = []
        for box in boxes:
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # Если это человек
            if cls == person_class and conf >= conf_threshold:
                person_boxes.append(box)
        
        if len(person_boxes) == 0:
            print(f"  - Нет людей")
            processed += 1
            continue
        
        # Создаем файл разметки
        label_file = labels_path / (image_file.stem + ".txt")
        
        with open(label_file, 'w') as f:
            for box in person_boxes:
                # Получаем координаты
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Вычисляем нормализованные координаты
                img_h, img_w = image.shape[:2]
                
                center_x = (x1 + x2) / 2 / img_w
                center_y = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # Записываем в формате YOLO (класс 0 для всех людей)
                # В LabelImg потом можно поменять на 1 для жилетов
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"  ✓ Создано аннотаций: {len(person_boxes)}")
        total_annotations += len(person_boxes)
        processed += 1
    
    print("\n" + "=" * 60)
    print("✅ ПРЕДРАЗМЕТКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"Обработано изображений: {processed}/{len(image_files)}")
    print(f"Создано аннотаций: {total_annotations}")
    print(f"Файлы разметки сохранены в: {labels_dir}")
    print()
    
    print("📝 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Откройте LabelImg: labelImg")
    print("2. Загрузите папку: data/images/train/")
    print("3. Для каждого изображения:")
    print("   - Поменяйте класс с 0 на 1 для жилетов")
    print("   - Оставьте 0 для касок")
    print("   - Удалите ненужные box (F или Delete)")
    print("   - Добавьте недостающие объекты (W)")
    print("4. Сохраните и запустите обучение: python run_training.py")
    print()
    print("💡 Совет: используйте горячие клавиши:")
    print("   - 1 = класс helmet (каска)")
    print("   - 2 = класс vest (жилет)")
    print("   - W = новый box")
    print("   - D = следующее изображение")
    print("   - A = предыдущее изображение")
    print("=" * 60)


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Автоматическая предразметка кадров")
    parser.add_argument(
        "--images",
        type=str,
        default="data/images/train",
        help="Папка с изображениями (по умолчанию: data/images/train)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels/train",
        help="Папка для разметки (по умолчанию: data/labels/train)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Порог уверенности (по умолчанию: 0.3)"
    )
    
    args = parser.parse_args()
    
    try:
        auto_prelabel(args.images, args.labels, args.conf)
    except KeyboardInterrupt:
        print("\n\n⚠️  Предразметка прервана пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
