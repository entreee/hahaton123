"""
Скрипт для визуализации разметки (bounding boxes) на изображениях.

Использование:
    python visualize_labels.py                    # Визуализирует train изображения
    python visualize_labels.py --split val       # Визуализирует val изображения
    python visualize_labels.py --output custom   # Сохраняет в папку custom
    python visualize_labels.py --limit 10        # Обрабатывает только 10 изображений
"""

import cv2
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


# Цвета для классов (BGR формат для OpenCV)
CLASS_COLORS = {
    0: (0, 165, 255),    # Оранжевый для каски
    1: (0, 255, 255),    # Желтый для жилета
}

CLASS_NAMES = {
    0: "helmet",
    1: "vest",
}


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Парсит YOLO формат файла разметки.
    
    Формат YOLO: class_id center_x center_y width height (все в относительных координатах 0-1)
    
    Args:
        label_path: Путь к файлу разметки
        
    Returns:
        Список кортежей (class_id, center_x, center_y, width, height)
    """
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                annotations.append((class_id, center_x, center_y, width, height))
    except Exception as e:
        print(f"Ошибка при чтении {label_path}: {e}")
    
    return annotations


def yolo_to_bbox(
    center_x: float, 
    center_y: float, 
    width: float, 
    height: float,
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    Конвертирует YOLO формат (относительные координаты) в абсолютные координаты bounding box.
    
    Args:
        center_x, center_y: Центр в относительных координатах (0-1)
        width, height: Ширина и высота в относительных координатах (0-1)
        img_width, img_height: Размеры изображения
        
    Returns:
        (x1, y1, x2, y2) - координаты углов bounding box
    """
    # Конвертируем в абсолютные координаты
    abs_center_x = center_x * img_width
    abs_center_y = center_y * img_height
    abs_width = width * img_width
    abs_height = height * img_height
    
    # Вычисляем углы
    x1 = int(abs_center_x - abs_width / 2)
    y1 = int(abs_center_y - abs_height / 2)
    x2 = int(abs_center_x + abs_width / 2)
    y2 = int(abs_center_y + abs_height / 2)
    
    # Ограничиваем границами изображения
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    return (x1, y1, x2, y2)


def draw_bboxes(
    image: np.ndarray,
    annotations: List[Tuple[int, float, float, float, float]],
    show_class: bool = True,
    show_confidence: bool = False,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Рисует bounding boxes на изображении.
    
    Args:
        image: Изображение (BGR формат)
        annotations: Список аннотаций в формате YOLO
        show_class: Показывать название класса
        show_confidence: Показывать уверенность (не используется, т.к. в YOLO разметке нет confidence)
        line_thickness: Толщина линий
        
    Returns:
        Изображение с нарисованными bounding boxes
    """
    img_height, img_width = image.shape[:2]
    result_image = image.copy()
    
    for class_id, center_x, center_y, width, height in annotations:
        # Конвертируем в абсолютные координаты
        x1, y1, x2, y2 = yolo_to_bbox(center_x, center_y, width, height, img_width, img_height)
        
        # Получаем цвет и название класса
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Белый по умолчанию
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        
        # Рисуем прямоугольник
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Подпись
        if show_class:
            label = class_name
            
            # Размер текста
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Размер текста для фона
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Рисуем фон для текста
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Рисуем текст
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),  # Белый текст
                thickness
            )
    
    return result_image


def visualize_labels(
    images_dir: str = "data/images/train",
    labels_dir: str = "data/labels/train",
    output_dir: str = "output/visualized_labels",
    limit: Optional[int] = None,
    show_class: bool = True,
    line_thickness: int = 2
) -> dict:
    """
    Визуализирует разметку на изображениях.
    
    Args:
        images_dir: Папка с изображениями
        labels_dir: Папка с файлами разметки
        output_dir: Папка для сохранения результатов
        limit: Максимальное количество изображений для обработки (None = все)
        show_class: Показывать название класса
        line_thickness: Толщина линий
        
    Returns:
        Словарь со статистикой
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # Создаем выходную папку
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем существование папок
    if not images_path.exists():
        print(f"❌ Папка с изображениями не найдена: {images_dir}")
        return {'processed': 0, 'errors': 1, 'saved': 0}
    
    if not labels_path.exists():
        print(f"❌ Папка с разметкой не найдена: {labels_dir}")
        return {'processed': 0, 'errors': 1, 'saved': 0}
    
    # Находим все изображения
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(ext))
        image_files.extend(images_path.glob(ext.upper()))
    
    if not image_files:
        print(f"❌ Не найдено изображений в {images_dir}")
        return {'processed': 0, 'errors': 0, 'saved': 0}
    
    # Ограничиваем количество
    if limit:
        image_files = image_files[:limit]
    
    print(f"📁 Найдено изображений: {len(image_files)}")
    print(f"📁 Папка с разметкой: {labels_dir}")
    print(f"💾 Результаты будут сохранены в: {output_dir}")
    print()
    
    stats = {
        'processed': 0,
        'saved': 0,
        'errors': 0,
        'with_labels': 0,
        'without_labels': 0,
        'total_boxes': 0
    }
    
    for i, image_file in enumerate(image_files, 1):
        try:
            # Загружаем изображение
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"  [{i}/{len(image_files)}] ❌ Не удалось загрузить: {image_file.name}")
                stats['errors'] += 1
                continue
            
            # Ищем соответствующий файл разметки
            label_file = labels_path / f"{image_file.stem}.txt"
            
            if not label_file.exists():
                print(f"  [{i}/{len(image_files)}] ⚠️  Нет разметки: {image_file.name}")
                stats['without_labels'] += 1
                # Сохраняем оригинальное изображение без разметки
                output_file = output_path / f"{image_file.stem}_no_labels.jpg"
                cv2.imwrite(str(output_file), image)
                stats['saved'] += 1
                continue
            
            # Парсим разметку
            annotations = parse_yolo_label(label_file)
            
            if not annotations:
                print(f"  [{i}/{len(image_files)}] ⚠️  Пустая разметка: {image_file.name}")
                stats['without_labels'] += 1
                output_file = output_path / f"{image_file.stem}_empty.jpg"
                cv2.imwrite(str(output_file), image)
                stats['saved'] += 1
                continue
            
            # Рисуем bounding boxes
            result_image = draw_bboxes(
                image,
                annotations,
                show_class=show_class,
                line_thickness=line_thickness
            )
            
            # Сохраняем результат
            output_file = output_path / f"{image_file.stem}_labeled.jpg"
            cv2.imwrite(str(output_file), result_image)
            
            stats['processed'] += 1
            stats['saved'] += 1
            stats['with_labels'] += 1
            stats['total_boxes'] += len(annotations)
            
            print(f"  [{i}/{len(image_files)}] ✓ {image_file.name}: {len(annotations)} boxes")
            
        except Exception as e:
            print(f"  [{i}/{len(image_files)}] ❌ Ошибка при обработке {image_file.name}: {e}")
            stats['errors'] += 1
    
    print()
    print("=" * 60)
    print("СТАТИСТИКА:")
    print(f"  Обработано: {stats['processed']}")
    print(f"  Сохранено: {stats['saved']}")
    print(f"  С разметкой: {stats['with_labels']}")
    print(f"  Без разметки: {stats['without_labels']}")
    print(f"  Всего bounding boxes: {stats['total_boxes']}")
    print(f"  Ошибок: {stats['errors']}")
    print("=" * 60)
    print(f"✅ Результаты сохранены в: {output_dir}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация разметки (bounding boxes) на изображениях"
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Раздел датасета (train или val)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Папка для сохранения результатов (по умолчанию: output/visualized_labels_{split})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Максимальное количество изображений для обработки'
    )
    parser.add_argument(
        '--no-class',
        action='store_true',
        help='Не показывать названия классов'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Толщина линий bounding boxes (по умолчанию: 2)'
    )
    
    args = parser.parse_args()
    
    # Определяем пути
    split = args.split
    images_dir = f"data/images/{split}"
    labels_dir = f"data/labels/{split}"
    
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"output/visualized_labels_{split}"
    
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РАЗМЕТКИ НА ИЗОБРАЖЕНИЯХ")
    print("=" * 60)
    print(f"Раздел: {split}")
    print(f"Изображения: {images_dir}")
    print(f"Разметка: {labels_dir}")
    print()
    
    # Запускаем визуализацию
    stats = visualize_labels(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        limit=args.limit,
        show_class=not args.no_class,
        line_thickness=args.thickness
    )
    
    return stats


if __name__ == "__main__":
    main()

