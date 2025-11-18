"""
Модуль для разделения датасета на train/validation.

Использование:
from src.data.split_dataset import split_dataset
split_dataset(
    train_images="data/images/train",
    train_labels="data/labels/train",
    val_ratio=0.2
)
"""

import random
import shutil
from pathlib import Path
from typing import Tuple


def split_dataset(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str = "data/images/val",
    val_labels_dir: str = "data/labels/val",
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[int, int]:
    """
    Разделяет датасет на обучающую и валидационную выборки.
    
    Args:
        train_images_dir: Директория с изображениями для обучения
        train_labels_dir: Директория с разметкой для обучения
        val_images_dir: Директория для валидационных изображений
        val_labels_dir: Директория для валидационной разметки
        val_ratio: Доля данных для валидации (0.2 = 20%)
        seed: Seed для воспроизводимости
        
    Returns:
        Кортеж (количество перемещенных изображений, количество перемещенных разметок)
    """
    random.seed(seed)
    
    # Пути
    train_images_path = Path(train_images_dir)
    train_labels_path = Path(train_labels_dir)
    val_images_path = Path(val_images_dir)
    val_labels_path = Path(val_labels_dir)
    
    # Создаем папки val
    val_images_path.mkdir(parents=True, exist_ok=True)
    val_labels_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие исходных данных
    if not train_images_path.exists():
        print(f"❌ Директория не найдена: {train_images_dir}")
        return 0, 0
    
    if not train_labels_path.exists():
        print(f"❌ Директория не найдена: {train_labels_dir}")
        return 0, 0
    
    # Получаем все изображения
    images = (
        list(train_images_path.glob("*.jpg")) +
        list(train_images_path.glob("*.png")) +
        list(train_images_path.glob("*.jpeg")) +
        list(train_images_path.glob("*.bmp"))
    )
    
    if len(images) == 0:
        print(f"❌ Не найдено изображений в {train_images_dir}")
        print("Поддерживаемые форматы: JPG, PNG, JPEG, BMP")
        return 0, 0
    
    # Перемешиваем
    random.shuffle(images)
    
    # Вычисляем количество для валидации
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]
    train_count = len(images) - val_count
    
    print(f"🔍 Всего изображений: {len(images)}")
    print(f"📊 Train: {train_count} ({(1-val_ratio)*100:.1f}%)")
    print(f"📊 Val: {val_count} ({val_ratio*100:.1f}%)")
    print()
    
    moved_images = 0
    moved_labels = 0
    missing_labels = 0
    
    # Перемещаем изображения и разметку
    for i, img in enumerate(val_images, 1):
        # Перемещаем изображение
        dest_img = val_images_path / img.name
        try:
            shutil.move(str(img), str(dest_img))
            moved_images += 1
        except Exception as e:
            print(f"❌ Ошибка перемещения {img.name}: {e}")
            continue
        
        # Перемещаем соответствующую разметку
        label_file = train_labels_path / (img.stem + ".txt")
        if label_file.exists():
            dest_label = val_labels_path / label_file.name
            try:
                shutil.move(str(label_file), str(dest_label))
                moved_labels += 1
            except Exception as e:
                print(f"❌ Ошибка перемещения разметки {label_file.name}: {e}")
        else:
            missing_labels += 1
            print(f"⚠️  [{i}/{len(val_images)}] Нет разметки для {img.name}")
    
    # Финальная статистика
    print(f"\n✅ РАЗДЕЛЕНИЕ ДАТАСЕТА ЗАВЕРШЕНО!")
    print(f"📁 Перемещено изображений: {moved_images}/{val_count}")
    print(f"📄 Перемещено разметок: {moved_labels}")
    print(f"⚠️  Разметок не найдено: {missing_labels}")
    
    if missing_labels > 0:
        print(f"\n💡 Рекомендация: проверьте {missing_labels} изображений без разметки")
    
    return moved_images, moved_labels


def check_split_result(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str
) -> dict:
    """
    Проверяет результат разделения датасета.
    
    Returns:
        Словарь со статистикой по папкам
    """
    stats = {
        'train_images': 0, 'train_labels': 0,
        'val_images': 0, 'val_labels': 0,
        'missing_labels': 0, 'extra_labels': 0
    }
    
    # Подсчет файлов
    for glob_pattern in ['*.jpg', '*.png', '*.jpeg']:
        stats['train_images'] += len(list(Path(train_images_dir).glob(glob_pattern)))
        stats['val_images'] += len(list(Path(val_images_dir).glob(glob_pattern)))
    
    stats['train_labels'] = len(list(Path(train_labels_dir).glob("*.txt")))
    stats['val_labels'] = len(list(Path(val_labels_dir).glob("*.txt")))
    
    # Проверка соответствия
    for img_file in Path(train_images_dir).glob("*.jpg"):
        label_file = Path(train_labels_dir) / f"{img_file.stem}.txt"
        if not label_file.exists():
            stats['missing_labels'] += 1
    
    for label_file in Path(train_labels_dir).glob("*.txt"):
        img_file = Path(train_images_dir) / f"{label_file.stem}.jpg"
        if not img_file.exists():
            stats['extra_labels'] += 1
    
    return stats


if __name__ == "__main__":
    # Пример использования
    print("=== РАЗДЕЛЕНИЕ ДАТАСЕТА ===")
    moved_img, moved_lbl = split_dataset(
        train_images_dir="data/images/train",
        train_labels_dir="data/labels/train",
        val_ratio=0.2
    )
    
    # Проверка результата
    if moved_img > 0:
        print("\n=== ПРОВЕРКА РЕЗУЛЬТАТА ===")
        stats = check_split_result(
            "data/images/train", "data/labels/train",
            "data/images/val", "data/labels/val"
        )
        
        print(f"Train: {stats['train_images']} изображений, {stats['train_labels']} разметок")
        print(f"Val: {stats['val_images']} изображений, {stats['val_labels']} разметок")
        
        if stats['missing_labels'] == 0 and stats['extra_labels'] == 0:
            print("✅ Разделение выполнено корректно!")
        else:
            print(f"⚠️  Найдено несоответствий: {stats['missing_labels']} + {stats['extra_labels']}")
    
    print(f"\nГотово! Перемещено: {moved_img} изображений, {moved_lbl} разметок")
