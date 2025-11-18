"""
Полный автоматический запуск пайплайна для детекции СИЗ (каска + жилет).

Что делает этот скрипт:
- создает структуру проекта и конфигурацию (если их ещё нет);
- извлекает кадры из всех видео в папке `videos/` (если видео есть);
- делает автоматическую предразметку людей на кадрах;
- делит данные на train/val;
- проверяет корректность структуры и разметки;
- обучает модель YOLOv8 с подобранными параметрами;
- выполняет быстрый тест модели на одном изображении из валидации.

Запуск (из корня проекта):

    python run_full_pipeline.py
"""

from pathlib import Path
import sys
import os


def main() -> None:
    # Добавляем корень проекта в PYTHONPATH
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Импорты локальных модулей (после добавления пути)
    from src.utils.config import config, ProjectConfig
    from src.data.extract_frames import auto_extract_frames
    from src.data.auto_prelabel import auto_prelabel
    from src.data.split_dataset import split_dataset
    from src.data.data_utils import check_data_structure, get_dataset_stats
    from src.models.train_model import PPEDetectorTrainer
    from src.inference.detect_utils import PPEDetector

    print("=" * 70)
    print("ПОЛНЫЙ АВТОМАТИЧЕСКИЙ ПАЙПЛАЙН ДЛЯ ДЕТЕКЦИИ СИЗ (КАСКА + ЖИЛЕТ)")
    print("=" * 70)
    print(f"Корень проекта: {project_root}")
    print()

    # 1. Конфигурация и структура проекта
    print("1) Настройка конфигурации и структуры проекта...")
    paths_summary = config.get_paths_summary()
    config.validate_paths()
    config.create_dataset_config()
    config.create_classes_file()
    print()

    # 2. Извлечение кадров из видео (если есть видео)
    print("2) Извлечение кадров из видео (если видео есть в папке 'videos/')...")
    video_exts = config.video_extensions
    videos = []
    for ext in video_exts:
        videos.extend(config.videos_dir.glob(f"*{ext}"))
        videos.extend(config.videos_dir.glob(f"*{ext.upper()}"))

    if videos:
        print(f"Найдено видео файлов: {len(videos)}")
        total_frames = auto_extract_frames(
            videos_dir=str(config.videos_dir),
            output_dir=str(config.data_dir / "images" / "train"),
            step=30,  # фиксированный шаг, чтобы не было слишком много кадров
        )
        print(f"✅ Кадры извлечены: {total_frames}")
    else:
        print("⚠️  В папке 'videos/' не найдено видео. Шаг извлечения кадров пропущен.")
        print("    Если у вас есть видео, поместите их в папку 'videos/' и запустите скрипт снова.")
    print()

    # 3. Автоматическая предразметка (если есть изображения и нет разметки)
    print("3) Автоматическая предразметка (auto pre-labeling)...")
    train_images_dir = config.data_dir / "images" / "train"
    train_labels_dir = config.data_dir / "labels" / "train"
    train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")) + list(
        train_images_dir.glob("*.jpeg")
    )
    train_labels = list(train_labels_dir.glob("*.txt"))

    if train_images and not train_labels:
        print(f"Найдено изображений для разметки: {len(train_images)}")
        stats = auto_prelabel(
            images_dir=str(train_images_dir),
            labels_dir=str(train_labels_dir),
            conf_threshold=config.prelabel_conf_threshold,
        )
        print(f"✅ Авторазметка завершена: {stats}")
        print("   Рекомендуется после этого пройтись по разметке в LabelImg и подправить сложные случаи.")
    else:
        print("⚠️  Авторазметка пропущена:")
        if not train_images:
            print("   - нет изображений в data/images/train/")
        else:
            print("   - разметка уже существует в data/labels/train/ (скрипт не перезаписывает существующие *.txt)")
    print()

    # 4. Разделение на train/val (если val пустой)
    print("4) Разделение датасета на train/val...")
    val_images_dir = config.data_dir / "images" / "val"
    val_labels_dir = config.data_dir / "labels" / "val"
    val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(
        val_images_dir.glob("*.jpeg")
    )

    if train_images and not val_images:
        moved_images, moved_labels = split_dataset(
            train_images_dir=str(train_images_dir),
            train_labels_dir=str(train_labels_dir),
            val_images_dir=str(val_images_dir),
            val_labels_dir=str(val_labels_dir),
            val_ratio=config.val_ratio,
            seed=config.random_seed,
        )
        print(f"✅ Разделение выполнено: {moved_images} изображений, {moved_labels} разметок перемещено в val/")
    else:
        print("⚠️  Разделение train/val пропущено:")
        if not train_images:
            print("   - нет изображений в data/images/train/")
        else:
            print("   - валидационная выборка уже существует (data/images/val/ не пустая)")
    print()

    # 5. Проверка структуры данных и разметки
    print("5) Проверка структуры данных и разметки...")
    data_ok = check_data_structure(data_root=str(config.data_dir))
    dataset_stats = get_dataset_stats(data_root=str(config.data_dir))
    total_images = dataset_stats.get("total_images", 0)
    class_distribution = dataset_stats.get("class_distribution", {})
    print()

    if total_images == 0:
        print("❌ Не найдено данных для обучения (нет изображений в data/images/train/ и data/images/val/).")
        print("   Загрузите данные (или извлеките кадры из видео) и запустите скрипт снова.")
        return

    print(f"Всего изображений: {total_images}")
    print(f"Распределение классов (train): {class_distribution}")
    print()

    # 6. Обучение модели
    print("6) Обучение модели YOLOv8...")
    trainer = PPEDetectorTrainer(
        model_name=config.model_name,
        config_path=str(config.config_dir / "ppe_data.yaml"),
        project_dir=str(config.models_dir),
        experiment_name=config.experiment_name,
    )

    # Автоматический выбор параметров в зависимости от устройства
    if config.device == "cpu":
        epochs = 20
        batch_size = 8
    else:
        epochs = config.epochs
        batch_size = config.batch_size

    train_results = trainer.train(
        epochs=epochs,
        img_size=config.img_size,
        batch_size=batch_size,
        patience=config.patience,
        workers=config.workers,
    )

    if not train_results.get("success", False):
        print("❌ Обучение завершилось с ошибкой.")
        print(f"Ошибка: {train_results.get('error')}")
        return

    best_model_path = Path(train_results.get("best_model", ""))
    print()
    print("✅ Обучение завершено успешно!")
    print(f"Лучшая модель: {best_model_path}")
    print()

    # 7. Быстрый тест модели на одном изображении
    print("7) Быстрый тест обученной модели на одном изображении из валидации...")
    if not best_model_path.exists():
        print("⚠️  Файл лучшей модели не найден, пропускаю тест инференса.")
    else:
        val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(
            val_images_dir.glob("*.jpeg")
        )
        if not val_images:
            print("⚠️  Нет изображений в data/images/val/ для теста.")
        else:
            test_img = val_images[0]
            print(f"Тестируем на: {test_img}")
            detector = PPEDetector(str(best_model_path))
            try:
                result_img, detections = detector.detect_image(str(test_img), save_result=True)
                print(f"🎯 Найдено детекций: {len(detections)}")
                for det in detections:
                    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
                print("💾 Результат детекции сохранен в папке 'output/detections/'.")
            except Exception as e:
                print(f"⚠️  Ошибка при тестовом инференсе: {e}")

    print()
    print("=" * 70)
    print("🎉 ПАЙПЛАЙН ЗАВЕРШЕН!")
    print("=" * 70)
    print("Что сделано:")
    print("- Структура проекта и конфигурация подготовлены;")
    print("- Кадры из видео извлечены (если видео были);")
    print("- Предразметка выполнена (если не было разметки);")
    print("- Данные разделены на train/val;")
    print("- Модель обучена;")
    print("- Быстрый тест модели на одном изображении выполнен.")
    print()
    print("Дальше вы можете:")
    print("- Открыть ноутбук 'notebooks/inference.ipynb' для интерактивных тестов;")
    print("- Использовать 'src/inference/detect_utils.py' для детекции в своих скриптах.")


if __name__ == "__main__":
    main()


