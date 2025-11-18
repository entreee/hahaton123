"""
Скрипт для автоматического разделения датасета на train/val.

Использование:
    python split_dataset.py --ratio 0.2
"""

import os
import shutil
import argparse
from pathlib import Path
import random


def split_dataset(train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, val_ratio=0.2, seed=42):
    """
    Разделяет датасет на обучающую и валидационную выборки.
    
    Args:
        train_images_dir: Директория с изображениями для обучения
        train_labels_dir: Директория с разметкой для обучения
        val_images_dir: Директория для валидационных изображений
        val_labels_dir: Директория для валидационной разметки
        val_ratio: Доля данных для валидации (по умолчанию 0.2 = 20%)
        seed: Seed для воспроизводимости
    """
    # Устанавливаем seed для воспроизводимости
    random.seed(seed)
    
    # Создаем директории для валидации
    Path(val_images_dir).mkdir(parents=True, exist_ok=True)
    Path(val_labels_dir).mkdir(parents=True, exist_ok=True)
    
    # Получаем все изображения
    train_images_path = Path(train_images_dir)
    images = list(train_images_path.glob("*.jpg")) + list(train_images_path.glob("*.png"))
    
    if len(images) == 0:
        print(f"Ошибка: не найдено изображений в {train_images_dir}")
        return
    
    # Перемешиваем
    random.shuffle(images)
    
    # Вычисляем количество для валидации
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    
    print(f"Всего изображений: {len(images)}")
    print(f"Для валидации: {len(val_images)} ({val_ratio*100:.1f}%)")
    print(f"Для обучения: {len(images) - len(val_images)} ({(1-val_ratio)*100:.1f}%)")
    print()
    
    moved_count = 0
    
    # Перемещаем изображения и разметку
    for img in val_images:
        # Перемещаем изображение
        dest_img = Path(val_images_dir) / img.name
        shutil.move(str(img), str(dest_img))
        
        # Перемещаем разметку
        label_file = Path(train_labels_dir) / (img.stem + ".txt")
        if label_file.exists():
            dest_label = Path(val_labels_dir) / label_file.name
            shutil.move(str(label_file), str(dest_label))
            moved_count += 1
        else:
            print(f"Предупреждение: не найдена разметка для {img.name}")
    
    print(f"Перемещено {moved_count} пар изображение+разметка в валидационную выборку")


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Разделение датасета на train/val"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Доля данных для валидации (по умолчанию: 0.2 = 20%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для воспроизводимости (по умолчанию: 42)"
    )
    
    args = parser.parse_args()
    
    # Пути к директориям
    train_images_dir = "data/images/train"
    train_labels_dir = "data/labels/train"
    val_images_dir = "data/images/val"
    val_labels_dir = "data/labels/val"
    
    # Проверяем существование директорий
    if not os.path.exists(train_images_dir):
        print(f"Ошибка: директория не найдена: {train_images_dir}")
        return
    
    if not os.path.exists(train_labels_dir):
        print(f"Ошибка: директория не найдена: {train_labels_dir}")
        return
    
    try:
        split_dataset(
            train_images_dir,
            train_labels_dir,
            val_images_dir,
            val_labels_dir,
            args.ratio,
            args.seed
        )
    except Exception as e:
        print(f"Ошибка при разделении датасета: {e}")
        raise


if __name__ == "__main__":
    main()

