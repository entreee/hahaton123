"""
Скрипт для полного цикла обучения модели детекции СИЗ.

Автоматически:
1. Проверяет структуру данных
2. Разделяет датасет на train/val (если нужно)
3. Запускает обучение модели
4. Сохраняет результаты

Использование:
    python run_training.py
"""

import os
import sys
from pathlib import Path
from train import train_model
from split_dataset import split_dataset


def check_data_structure():
    """Проверяет структуру директорий с данными."""
    print("=" * 60)
    print("Проверка структуры данных")
    print("=" * 60)
    
    required_dirs = [
        "data/images/train",
        "data/labels/train"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            print(f"❌ Отсутствует: {dir_path}")
        else:
            print(f"✅ Найдено: {dir_path}")
    
    if missing_dirs:
        print("\n❌ Ошибка: отсутствуют необходимые директории!")
        print("Создайте следующие директории:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    
    # Проверяем наличие файлов
    train_images = list(Path("data/images/train").glob("*.jpg")) + \
                   list(Path("data/images/train").glob("*.png"))
    train_labels = list(Path("data/labels/train").glob("*.txt"))
    
    print(f"\nНайдено изображений в train: {len(train_images)}")
    print(f"Найдено разметок в train: {len(train_labels)}")
    
    if len(train_images) == 0:
        print("\n❌ Ошибка: не найдено изображений для обучения!")
        print("Поместите изображения в data/images/train/")
        return False
    
    if len(train_labels) == 0:
        print("\n❌ Ошибка: не найдено файлов разметки!")
        print("Поместите файлы разметки (.txt) в data/labels/train/")
        return False
    
    # Проверяем соответствие имен
    image_names = {img.stem for img in train_images}
    label_names = {label.stem for label in train_labels}
    
    missing_labels = image_names - label_names
    if missing_labels:
        print(f"\n⚠️  Предупреждение: {len(missing_labels)} изображений без разметки")
        if len(missing_labels) <= 10:
            for name in list(missing_labels)[:10]:
                print(f"  - {name}")
    
    extra_labels = label_names - image_names
    if extra_labels:
        print(f"\n⚠️  Предупреждение: {len(extra_labels)} файлов разметки без изображений")
    
    print("\n✅ Структура данных проверена")
    return True


def check_val_split():
    """Проверяет, нужно ли разделить датасет на train/val."""
    val_images_dir = Path("data/images/val")
    val_labels_dir = Path("data/labels/val")
    
    if val_images_dir.exists() and val_labels_dir.exists():
        val_images = list(val_images_dir.glob("*.jpg")) + \
                     list(val_images_dir.glob("*.png"))
        if len(val_images) > 0:
            print(f"\n✅ Валидационная выборка уже существует: {len(val_images)} изображений")
            return False
    
    print("\n⚠️  Валидационная выборка не найдена")
    return True


def main():
    """Основная функция полного цикла обучения."""
    print("\n" + "=" * 60)
    print("ПОЛНЫЙ ЦИКЛ ОБУЧЕНИЯ МОДЕЛИ ДЕТЕКЦИИ СИЗ")
    print("=" * 60)
    print("\nКлассы для детекции:")
    print("  0 - Защитная каска (оранжевая)")
    print("  1 - Сигнальный жилет")
    print("=" * 60 + "\n")
    
    # Шаг 1: Проверка структуры данных
    if not check_data_structure():
        print("\n❌ Прерывание: исправьте структуру данных и запустите снова")
        sys.exit(1)
    
    # Шаг 2: Разделение на train/val (если нужно)
    if check_val_split():
        print("\n" + "=" * 60)
        print("Разделение датасета на train/val")
        print("=" * 60)
        
        response = input("\nРазделить датасет автоматически? (y/n, по умолчанию y): ").strip().lower()
        if response == '' or response == 'y':
            ratio = input("Доля для валидации (по умолчанию 0.2 = 20%): ").strip()
            ratio = float(ratio) if ratio else 0.2
            
            try:
                split_dataset(
                    train_images_dir="data/images/train",
                    train_labels_dir="data/labels/train",
                    val_images_dir="data/images/val",
                    val_labels_dir="data/labels/val",
                    val_ratio=ratio
                )
                print("\n✅ Датасет успешно разделен")
            except Exception as e:
                print(f"\n❌ Ошибка при разделении датасета: {e}")
                sys.exit(1)
        else:
            print("Пропуск разделения. Убедитесь, что валидационная выборка создана вручную.")
    
    # Шаг 3: Проверка конфигурации
    config_path = "config/ppe_data.yaml"
    if not os.path.exists(config_path):
        print(f"\n❌ Ошибка: конфигурационный файл не найден: {config_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Начало обучения модели")
    print("=" * 60 + "\n")
    
    # Шаг 4: Обучение модели
    try:
        train_model()
        print("\n" + "=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 60)
        print("\nЛучшая модель сохранена в: models/ppe_model/weights/best.pt")
        print("Используйте эту модель для детекции:")
        print("  python detect_video.py --video input.mp4 --model models/ppe_model/weights/best.pt")
        print("  python detect_camera.py --model models/ppe_model/weights/best.pt")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

