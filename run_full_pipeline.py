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
import logging
from datetime import datetime


def setup_logging(log_dir: Path = None) -> logging.Logger:
    """
    Настраивает логирование в файл и консоль.
    
    Args:
        log_dir: Директория для логов (если None - logs/ в корне проекта)
        
    Returns:
        Настроенный logger
    """
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Имя файла лога с временной меткой
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Настройка root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("ПОЛНЫЙ АВТОМАТИЧЕСКИЙ ПАЙПЛАЙН ДЛЯ ДЕТЕКЦИИ СИЗ")
    logger.info("=" * 70)
    logger.info(f"Логи сохраняются в: {log_file}")
    logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def main() -> None:
    # Настройка логирования
    logger = setup_logging()
    
    # Добавляем корень проекта в PYTHONPATH
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    logger.info(f"Корень проекта: {project_root}")
    
    try:
        logger.info("Начало импорта модулей...")
        # Импорты локальных модулей (после добавления пути)
        logger.info("Импорт config...")
        import time
        start_time = time.time()
        from src.utils.config import config, ProjectConfig
        logger.info(f"Импорт config завершен за {time.time() - start_time:.2f} сек")
        
        logger.info("Импорт data модулей...")
        start_time = time.time()
        
        logger.info("  Импорт extract_frames...")
        from src.data.extract_frames import auto_extract_frames
        logger.info(f"  extract_frames импортирован за {time.time() - start_time:.2f} сек")
        
        start_time = time.time()
        logger.info("  Импорт auto_prelabel...")
        from src.data.auto_prelabel import auto_prelabel
        logger.info(f"  auto_prelabel импортирован за {time.time() - start_time:.2f} сек")
        
        start_time = time.time()
        logger.info("  Импорт split_dataset...")
        from src.data.split_dataset import split_dataset
        logger.info(f"  split_dataset импортирован за {time.time() - start_time:.2f} сек")
        
        start_time = time.time()
        logger.info("  Импорт data_utils...")
        from src.data.data_utils import check_data_structure, get_dataset_stats
        logger.info(f"  data_utils импортирован за {time.time() - start_time:.2f} сек")
        
        logger.info("Импорт data модулей завершен")
        
        logger.info("Импорт models модулей...")
        start_time = time.time()
        from src.models.train_model import PPEDetectorTrainer
        logger.info(f"Импорт models модулей завершен за {time.time() - start_time:.2f} сек")
        
        logger.info("Импорт inference модулей...")
        start_time = time.time()
        from src.inference.detect_utils import PPEDetector
        logger.info(f"Импорт inference модулей завершен за {time.time() - start_time:.2f} сек")
        
        logger.info("Все модули успешно импортированы")
        
        # 1. Конфигурация и структура проекта
        logger.info("=" * 70)
        logger.info("ШАГ 1: Настройка конфигурации и структуры проекта")
        logger.info("=" * 70)
        try:
            logger.info("Получение сводки путей...")
            paths_summary = config.get_paths_summary()
            logger.info(f"Пути получены: {len(paths_summary)} элементов")
            
            logger.info("Валидация путей...")
            config.validate_paths()
            logger.info("Валидация путей завершена")
            
            logger.info("Создание конфигурации датасета...")
            config.create_dataset_config()
            logger.info("Конфигурация датасета создана")
            
            logger.info("Создание файла классов...")
            config.create_classes_file()
            logger.info("Файл классов создан")
            
            logger.info("Конфигурация и структура проекта подготовлены")
        except Exception as e:
            logger.error(f"Ошибка при настройке конфигурации: {e}", exc_info=True)
            raise
        
        # 2. Извлечение кадров из видео (если есть видео)
        logger.info("=" * 70)
        logger.info("ШАГ 2: Извлечение кадров из видео")
        logger.info("=" * 70)
        try:
            logger.info(f"Поиск видео в директории: {config.videos_dir}")
            video_exts = config.video_extensions
            logger.info(f"Поддерживаемые расширения: {video_exts}")
            
            videos = []
            for ext in video_exts:
                logger.debug(f"Поиск файлов с расширением: {ext}")
                videos.extend(config.videos_dir.glob(f"*{ext}"))
                videos.extend(config.videos_dir.glob(f"*{ext.upper()}"))
            
            logger.info(f"Найдено видео файлов: {len(videos)}")
            if videos:
                for i, video in enumerate(videos, 1):
                    logger.info(f"  [{i}] {video.name}")
            
            if videos:
                logger.info("Запуск извлечения кадров...")
                total_frames = auto_extract_frames(
                    videos_dir=str(config.videos_dir),
                    output_dir=str(config.data_dir / "images" / "train"),
                    step=30,  # фиксированный шаг, чтобы не было слишком много кадров
                )
                logger.info(f"Кадры извлечены: {total_frames}")
            else:
                logger.warning("В папке 'videos/' не найдено видео. Шаг извлечения кадров пропущен.")
                logger.info("Если у вас есть видео, поместите их в папку 'videos/' и запустите скрипт снова.")
        except Exception as e:
            logger.error(f"Ошибка при извлечении кадров: {e}", exc_info=True)
            logger.warning("Продолжаем выполнение пайплайна...")
        
        # 3. Автоматическая предразметка (если есть изображения и нет разметки)
        logger.info("=" * 70)
        logger.info("ШАГ 3: Автоматическая предразметка")
        logger.info("=" * 70)
        try:
            train_images_dir = config.data_dir / "images" / "train"
            train_labels_dir = config.data_dir / "labels" / "train"
            logger.info(f"Проверка изображений в: {train_images_dir}")
            logger.info(f"Проверка разметки в: {train_labels_dir}")
            
            train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")) + list(
                train_images_dir.glob("*.jpeg")
            )
            train_labels = list(train_labels_dir.glob("*.txt"))
            
            logger.info(f"Найдено изображений: {len(train_images)}")
            logger.info(f"Найдено файлов разметки: {len(train_labels)}")
            
            if train_images and not train_labels:
                logger.info(f"Запуск авторазметки для {len(train_images)} изображений...")
                stats = auto_prelabel(
                    images_dir=str(train_images_dir),
                    labels_dir=str(train_labels_dir),
                    conf_threshold=config.prelabel_conf_threshold,
                )
                logger.info(f"Авторазметка завершена: обработано {stats.get('processed', 0)}, аннотаций {stats.get('annotations', 0)}, ошибок {stats.get('errors', 0)}")
                logger.info("Рекомендуется после этого пройтись по разметке в LabelImg и подправить сложные случаи.")
            else:
                if not train_images:
                    logger.warning("Авторазметка пропущена: нет изображений в data/images/train/")
                else:
                    logger.info("Авторазметка пропущена: разметка уже существует в data/labels/train/")
        except Exception as e:
            logger.error(f"Ошибка при авторазметке: {e}", exc_info=True)
            logger.warning("Продолжаем выполнение пайплайна...")
        
        # 4. Разделение на train/val (если val пустой)
        logger.info("=" * 70)
        logger.info("ШАГ 4: Разделение датасета на train/val")
        logger.info("=" * 70)
        try:
            val_images_dir = config.data_dir / "images" / "val"
            val_labels_dir = config.data_dir / "labels" / "val"
            val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(
                val_images_dir.glob("*.jpeg")
            )
            
            train_images = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png")) + list(
                train_images_dir.glob("*.jpeg")
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
                logger.info(f"Разделение выполнено: {moved_images} изображений, {moved_labels} разметок перемещено в val/")
            else:
                if not train_images:
                    logger.warning("Разделение train/val пропущено: нет изображений в data/images/train/")
                else:
                    logger.info("Разделение train/val пропущено: валидационная выборка уже существует")
        except Exception as e:
            logger.error(f"Ошибка при разделении датасета: {e}", exc_info=True)
            logger.warning("Продолжаем выполнение пайплайна...")
        
        # 5. Проверка структуры данных и разметки
        logger.info("=" * 70)
        logger.info("ШАГ 5: Проверка структуры данных и разметки")
        logger.info("=" * 70)
        try:
            data_ok = check_data_structure(data_root=str(config.data_dir))
            dataset_stats = get_dataset_stats(data_root=str(config.data_dir))
            total_images = dataset_stats.get("total_images", 0)
            class_distribution = dataset_stats.get("class_distribution", {})
            
            logger.info(f"Всего изображений: {total_images}")
            logger.info(f"Распределение классов (train): {class_distribution}")
            
            if total_images == 0:
                logger.error("Не найдено данных для обучения (нет изображений в data/images/train/ и data/images/val/).")
                logger.error("Загрузите данные (или извлеките кадры из видео) и запустите скрипт снова.")
                return
        except Exception as e:
            logger.error(f"Ошибка при проверке данных: {e}", exc_info=True)
            raise
        
        # 6. Обучение модели
        logger.info("=" * 70)
        logger.info("ШАГ 6: Обучение модели YOLOv8")
        logger.info("=" * 70)
        try:
            logger.info("Инициализация PPEDetectorTrainer...")
            logger.info(f"  model_name: {config.model_name}")
            logger.info(f"  config_path: {config.config_dir / 'ppe_data.yaml'}")
            logger.info(f"  project_dir: {config.models_dir}")
            logger.info(f"  experiment_name: {config.experiment_name}")
            
            trainer = PPEDetectorTrainer(
                model_name=config.model_name,
                config_path=str(config.config_dir / "ppe_data.yaml"),
                project_dir=str(config.models_dir),
                experiment_name=config.experiment_name,
            )
            logger.info("PPEDetectorTrainer инициализирован")
            
            # Автоматический выбор параметров в зависимости от устройства
            # Для маленьких объектов используем увеличенный размер изображения
            logger.info(f"Текущее устройство: {config.device}")
            logger.info("Оптимизация для маленьких объектов и высокого угла обзора:")
            logger.info(f"  - Размер изображения: {config.img_size}x{config.img_size} (увеличен для лучшей детекции)")
            logger.info(f"  - Модель: {config.model_name} (более крупная модель)")
            logger.info(f"  - Порог уверенности: {config.conf_threshold} (понижен для маленьких объектов)")
            
            if config.device == "cpu":
                epochs = 50  # Увеличено для CPU
                batch_size = 2  # Уменьшено из-за очень большого размера изображения
                img_size = 1280  # Уменьшено для CPU, но все еще больше чем было
                logger.info(f"Используется CPU: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
            else:
                epochs = config.epochs
                batch_size = config.batch_size
                img_size = config.img_size
                logger.info(f"Используется GPU: epochs={epochs}, batch_size={batch_size}, img_size={img_size}")
            
            logger.info("Запуск обучения...")
            logger.info(f"Параметры: epochs={epochs}, img_size={img_size}, batch_size={batch_size}, patience={config.patience}, workers={config.workers}")
            
            train_results = trainer.train(
                epochs=epochs,
                img_size=img_size,  # Используем адаптированный размер
                batch_size=batch_size,
                patience=config.patience,
                workers=config.workers,
            )
            
            logger.info(f"Результат обучения получен: success={train_results.get('success', False)}")
            
            if not train_results.get("success", False):
                logger.error("Обучение завершилось с ошибкой.")
                logger.error(f"Ошибка: {train_results.get('error')}")
                return
            
            best_model_path = Path(train_results.get("best_model", ""))
            logger.info("Обучение завершено успешно!")
            logger.info(f"Лучшая модель: {best_model_path}")
            logger.info(f"Модель существует: {best_model_path.exists()}")
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}", exc_info=True)
            raise
        
        # 7. Быстрый тест модели на одном изображении
        logger.info("=" * 70)
        logger.info("ШАГ 7: Быстрый тест обученной модели")
        logger.info("=" * 70)
        try:
            if not best_model_path.exists():
                logger.warning("Файл лучшей модели не найден, пропускаю тест инференса.")
            else:
                val_images = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png")) + list(
                    val_images_dir.glob("*.jpeg")
                )
                if not val_images:
                    logger.warning("Нет изображений в data/images/val/ для теста.")
                else:
                    test_img = val_images[0]
                    logger.info(f"Тестируем на: {test_img}")
                    detector = PPEDetector(str(best_model_path))
                    try:
                        result_img, detections = detector.detect_image(str(test_img), save_result=True)
                        logger.info(f"Найдено детекций: {len(detections)}")
                        for det in detections:
                            logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}")
                        logger.info("Результат детекции сохранен в папке 'output/detections/'.")
                    except Exception as e:
                        logger.warning(f"Ошибка при тестовом инференсе: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Ошибка при тестировании модели: {e}", exc_info=True)
            logger.warning("Продолжаем...")
        
        # Финальное сообщение
        logger.info("=" * 70)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЕН!")
        logger.info("=" * 70)
        logger.info("Что сделано:")
        logger.info("- Структура проекта и конфигурация подготовлены;")
        logger.info("- Кадры из видео извлечены (если видео были);")
        logger.info("- Предразметка выполнена (если не было разметки);")
        logger.info("- Данные разделены на train/val;")
        logger.info("- Модель обучена;")
        logger.info("- Быстрый тест модели на одном изображении выполнен.")
        logger.info("")
        logger.info("Дальше вы можете:")
        logger.info("- Открыть ноутбук 'notebooks/inference.ipynb' для интерактивных тестов;")
        logger.info("- Использовать 'src/inference/detect_utils.py' для детекции в своих скриптах.")
        
    except Exception as e:
        logger.critical(f"Критическая ошибка в пайплайне: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
