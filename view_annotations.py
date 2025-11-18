"""
Скрипт для просмотра разметки изображений.

Использование:
    python view_annotations.py                    # Просмотр train данных
    python view_annotations.py --val              # Просмотр val данных
    python view_annotations.py --output output/    # Сохранить в папку
    python view_annotations.py --limit 10          # Показать только 10 изображений
"""

import cv2
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import ProjectConfig


# Цвета для классов (BGR формат для OpenCV)
CLASS_COLORS = {
    0: (0, 165, 255),    # Оранжевый для каски
    1: (0, 255, 255)     # Желтый для жилета
}

CLASS_NAMES = {
    0: "helmet (каска)",
    1: "vest (жилет)"
}


def draw_annotations(
    image_path: Path,
    label_path: Path,
    output_path: Path = None,
    show: bool = True
) -> None:
    """
    Рисует разметку на изображении.
    
    Args:
        image_path: Путь к изображению
        label_path: Путь к файлу разметки
        output_path: Путь для сохранения (если None, не сохраняется)
        show: Показывать ли изображение
    """
    # Загрузка изображения
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Чтение разметки
    if not label_path.exists():
        print(f"Предупреждение: файл разметки не найден {label_path}")
        if show:
            cv2.imshow(f"Image: {image_path.name} (без разметки)", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return
    
    annotations_count = 0
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                print(f"Ошибка в {label_path}:{line_num} - неверный формат (ожидается 5 значений)")
                continue
            
            try:
                class_id = int(parts[0])
                center_x = float(parts[1]) * width
                center_y = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height
                
                # Вычисляем координаты углов
                x1 = int(center_x - box_width / 2)
                y1 = int(center_y - box_height / 2)
                x2 = int(center_x + box_width / 2)
                y2 = int(center_y + box_height / 2)
                
                # Ограничиваем координаты
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Цвет для класса
                color = CLASS_COLORS.get(class_id, (255, 255, 255))
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                # Рисуем прямоугольник
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Рисуем текст с классом
                label_text = f"{class_name}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    image,
                    label_text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
                
                annotations_count += 1
                
            except (ValueError, IndexError) as e:
                print(f"Ошибка в {label_path}:{line_num} - {e}")
                continue
    
    # Добавляем информацию о файле
    info_text = f"{image_path.name} | {annotations_count} объектов"
    cv2.putText(
        image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Сохранение
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"Сохранено: {output_path}")
    
    # Показ
    if show:
        window_name = f"Image: {image_path.name} ({annotations_count} объектов)"
        cv2.imshow(window_name, image)
        print(f"Нажмите любую клавишу для продолжения... (ESC для выхода)")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == 27:  # ESC
            return False
    
    return True


def view_annotations(
    split: str = "train",
    limit: int = None,
    output_dir: Path = None,
    show: bool = True
) -> None:
    """
    Просматривает разметку для всех изображений в указанном сплите.
    
    Args:
        split: "train" или "val"
        limit: Максимальное количество изображений для просмотра
        output_dir: Папка для сохранения изображений (если None, не сохраняется)
        show: Показывать ли изображения
    """
    config = ProjectConfig()
    
    images_dir = config.data_dir / "images" / split
    labels_dir = config.data_dir / "labels" / split
    
    if not images_dir.exists():
        print(f"Ошибка: папка {images_dir} не найдена")
        return
    
    if not labels_dir.exists():
        print(f"Ошибка: папка {labels_dir} не найдена")
        return
    
    # Поиск всех изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(ext))
    
    if len(image_files) == 0:
        print(f"Не найдено изображений в {images_dir}")
        return
    
    print(f"Найдено изображений: {len(image_files)}")
    if limit:
        image_files = image_files[:limit]
        print(f"Будет показано: {len(image_files)} изображений")
    
    print(f"\nПросмотр разметки для {split} данных")
    print("=" * 60)
    
    saved_count = 0
    for i, image_path in enumerate(image_files, 1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        print(f"\n[{i}/{len(image_files)}] {image_path.name}")
        
        if output_dir:
            output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        else:
            output_path = None
        
        try:
            result = draw_annotations(image_path, label_path, output_path, show=show)
            if output_path and output_path.exists():
                saved_count += 1
            if result is False:  # ESC нажат
                print("Просмотр прерван пользователем")
                break
        except Exception as e:
            print(f"Ошибка при обработке {image_path.name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"Обработано: {i}/{len(image_files)} изображений")
    if output_dir and saved_count > 0:
        print(f"Сохранено: {saved_count} изображений в {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Просмотр разметки изображений",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python view_annotations.py                    # Просмотр train данных
  python view_annotations.py --val              # Просмотр val данных
  python view_annotations.py --output output/   # Сохранить в папку
  python view_annotations.py --limit 10         # Показать только 10 изображений
  python view_annotations.py --no-show          # Только сохранить, не показывать
        """
    )
    
    parser.add_argument(
        '--val',
        action='store_true',
        help='Просматривать validation данные вместо train'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Папка для сохранения изображений с разметкой'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Максимальное количество изображений для просмотра'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Не показывать изображения, только сохранять (если указан --output)'
    )
    
    args = parser.parse_args()
    
    split = "val" if args.val else "train"
    output_dir = Path(args.output) if args.output else None
    show = not args.no_show
    
    if output_dir:
        print(f"Изображения будут сохранены в: {output_dir}")
    
    view_annotations(
        split=split,
        limit=args.limit,
        output_dir=output_dir,
        show=show
    )


if __name__ == "__main__":
    main()

