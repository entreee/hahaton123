"""
Модуль для обучения модели YOLOv8.

Использование:
from src.models.train_model import PPEDetectorTrainer
trainer = PPEDetectorTrainer()
trainer.train(epochs=30, batch_size=16)
"""

from pathlib import Path
import torch
import logging
from typing import Optional, Dict, List
from datetime import datetime
# YOLO импортируется лениво внутри методов, чтобы не замедлять импорт модуля


class PPEDetectorTrainer:
    """
    Класс для обучения модели детекции СИЗ.
    
    Управляет процессом обучения YOLOv8 с логированием и мониторингом.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        config_path: str = "config/ppe_data.yaml",
        project_dir: str = "models",
        experiment_name: str = "ppe_detection"
    ):
        self.model_name = model_name
        self.config_path = Path(config_path)
        self.project_dir = Path(project_dir)
        self.experiment_name = experiment_name
        
        # Настройка логирования
        self.setup_logging()
        
        # Проверка конфигурации
        self._validate_config()
        
        # Устройство
        self.device = self._detect_device()
        self.logger.info(f"Используется устройство: {self.device}")
    
    def setup_logging(self):
        """Настройка логирования."""
        log_dir = self.project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
        self.logger.info(f"Логи будут сохранены в: {log_file}")
    
    def _validate_config(self):
        """Проверяет наличие конфигурационного файла."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Конфигурационный файл не найден: {self.config_path}\n"
                f"Создайте файл {self.config_path} с классами helmet и vest"
            )
        self.logger.info(f"Конфигурация загружена: {self.config_path}")
    
    def _detect_device(self) -> str:
        """Определяет доступное устройство (GPU/CPU) с детальной диагностикой."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            
            self.logger.info(f"CUDA доступна: версия {cuda_version}")
            if cudnn_version:
                self.logger.info(f"cuDNN доступен: версия {cudnn_version}")
            
            # Детальная информация о GPU
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory / (1024**3)  # GB
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_free = memory_total - memory_allocated
                capability = f"{props.major}.{props.minor}"
                
                self.logger.info(f"GPU {i}: {props.name}")
                self.logger.info(f"  Память: {memory_free:.2f} GB свободно / {memory_total:.2f} GB всего")
                self.logger.info(f"  CUDA Capability: {capability}")
                self.logger.info(f"  Multiprocessors: {props.multi_processor_count}")
            
            if gpu_count > 1:
                device = "0"  # Используем первую GPU
                self.logger.info(f"Обнаружено {gpu_count} GPU, используется GPU {device}")
            else:
                device = "0"
                self.logger.info(f"Используется GPU {device}")
            
            # Тест производительности
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                _ = test_tensor @ test_tensor
                torch.cuda.synchronize()
                self.logger.info("Тест GPU: успешно")
            except Exception as e:
                self.logger.warning(f"Тест GPU не прошел: {e}")
            
            return device
        else:
            self.logger.warning("CUDA недоступна, используется CPU")
            return "cpu"
    
    def train(
        self,
        epochs: int = 30,
        img_size: int = 640,
        batch_size: int = 16,
        patience: int = 10,
        workers: int = 8,
        save: bool = True,
        plots: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Обучает модель YOLOv8.
        
        Args:
            epochs: Количество эпох
            img_size: Размер входного изображения
            batch_size: Размер батча
            patience: Ранняя остановка после N эпох без улучшения
            workers: Количество worker'ов для загрузки данных
            save: Сохранять чекпоинты
            plots: Генерировать графики
            verbose: Подробный вывод
            
        Returns:
            Словарь с результатами обучения
        """
        self.logger.info("=== НАЧАЛО ОБУЧЕНИЯ ===")
        self.logger.info(f"Параметры: epochs={epochs}, img_size={img_size}, batch={batch_size}")
        
        # Создаем директорию проекта
        experiment_dir = self.project_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем модель (ленивый импорт YOLO)
        self.logger.info(f"Импорт YOLO...")
        from ultralytics import YOLO
        self.logger.info(f"Загрузка модели: {self.model_name}")
        try:
            model = YOLO(self.model_name)
            self.logger.info("Модель успешно загружена")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise
        
        # Параметры обучения (оптимизированы для маленьких объектов и высокого угла обзора)
        train_params = {
            'data': str(self.config_path),
            'epochs': epochs,
            'imgsz': img_size,
            'device': self.device,
            'project': str(self.project_dir),
            'name': self.experiment_name,
            'batch': batch_size,
            'patience': patience,
            'workers': workers,
            'save': save,
            'plots': plots,
            'verbose': verbose,
            'exist_ok': True,  # Перезаписывать результаты
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 2.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            # Augmentation оптимизирована для ОЧЕНЬ маленьких объектов и высокого угла обзора
            'hsv_h': 0.02,  # Увеличено для разнообразия цветов (белые и оранжевые каски)
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,  # Без поворота для высокого угла обзора
            'translate': 0.3,  # Еще больше увеличено для разнообразия позиций
            'scale': 0.95,  # Максимально увеличено для обучения на разных масштабах (критично для очень маленьких объектов)
            'shear': 0.0,
            'perspective': 0.0,  # Без перспективы для высокого угла обзора
            'flipud': 0.0,  # Без вертикального отражения для высокого угла
            'fliplr': 0.5,  # Горизонтальное отражение
            'mosaic': 1.0,  # Mosaic критичен для маленьких объектов
            'mixup': 0.2,  # Увеличено mixup для разнообразия
            'copy_paste': 0.4,  # Увеличено copy-paste для маленьких объектов
            'cfg': None,
            'tracker': None,
            'save_dir': str(experiment_dir),
            # Дополнительные параметры для маленьких объектов
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,  # Валидация во время обучения
        }
        
        self.logger.info(f"Директория эксперимента: {experiment_dir}")
        
        # Запуск обучения
        try:
            self.logger.info("Запуск обучения...")
            results = model.train(**train_params)
            
            self.logger.info("Обучение успешно завершено!")
            
            # Статистика результатов
            best_model = experiment_dir / "weights" / "best.pt"
            last_model = experiment_dir / "weights" / "last.pt"
            
            self.logger.info(f"Лучшая модель: {best_model}")
            self.logger.info(f"Последняя модель: {last_model}")
            
            # Метрики
            results_csv = experiment_dir / "results.csv"
            if results_csv.exists():
                import pandas as pd
                df = pd.read_csv(results_csv)
                final_metrics = df.iloc[-1]
                self.logger.info(
                    f"Финальные метрики: mAP50={final_metrics.get('metrics/mAP50(B)', 'N/A'):.3f}, "
                    f"mAP50-95={final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}"
                )
            
            return {
                'success': True,
                'experiment_dir': str(experiment_dir),
                'best_model': str(best_model),
                'last_model': str(last_model),
                'results': results,
                'metrics': final_metrics if 'final_metrics' in locals() else None
            }
            
        except KeyboardInterrupt:
            self.logger.warning("Обучение прервано пользователем")
            return {'success': False, 'error': 'Interrupted by user'}
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def validate_model(
        self,
        model_path: Optional[str] = None,
        data_path: str = "config/ppe_data.yaml",
        conf_threshold: float = 0.5
    ) -> Dict:
        """
        Валидирует обученную модель на валидационной выборке.
        
        Args:
            model_path: Путь к модели (если None - использует лучшую)
            data_path: Путь к конфигурации данных
            conf_threshold: Порог уверенности для детекции
            
        Returns:
            Результаты валидации
        """
        if model_path is None:
            model_path = self.project_dir / self.experiment_name / "weights" / "best.pt"
        
        if not Path(model_path).exists():
            self.logger.error(f"Модель не найдена: {model_path}")
            return {'success': False, 'error': f'Model not found: {model_path}'}
        
        self.logger.info(f"Валидация модели: {model_path}")
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model.val(
                data=data_path,
                conf=conf_threshold,
                verbose=True,
                save_json=True,
                project=self.project_dir,
                name=f"{self.experiment_name}_validation"
            )
            
            self.logger.info("Валидация завершена")
            self.logger.info(f"mAP50: {results.box.map50:.3f}")
            self.logger.info(f"mAP50-95: {results.box.map:.3f}")
            
            return {
                'success': True,
                'map50': results.box.map50,
                'map': results.box.map,
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_sample(
        self,
        image_path: str,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        save_result: bool = True
    ) -> Optional[Path]:
        """
        Тестирует модель на одном изображении.
        
        Args:
            image_path: Путь к тестовому изображению
            model_path: Путь к модели
            conf_threshold: Порог уверенности
            save_result: Сохранить результат
            
        Returns:
            Путь к сохраненному изображению с детекциями
        """
        if model_path is None:
            model_path = self.project_dir / self.experiment_name / "weights" / "best.pt"
        
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model.predict(
                image_path,
                conf=conf_threshold,
                save=save_result,
                project="output",
                name="test_prediction",
                exist_ok=True,
                verbose=False
            )
            
            if save_result:
                output_path = Path("output/test_prediction") / Path(image_path).name
                self.logger.info(f"Результат сохранен: {output_path}")
                return output_path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}")
            return None


def create_trainer_config(
    config_path: str = "config/ppe_data.yaml",
    classes: List[str] = None
) -> Path:
    """
    Создает конфигурационный файл для обучения.
    
    Args:
        config_path: Путь к конфигурации
        classes: Список классов (если None - стандартные)
        
    Returns:
        Путь к созданному файлу
    """
    if classes is None:
        classes = ['helmet', 'vest']
    
    config_content = f"""path: ./data
train: images/train
val: images/val

nc: {len(classes)}
names: 
"""
    
    for i, class_name in enumerate(classes):
        config_content += f"  {i}: {class_name}\n"
    
    config = Path(config_path)
    config.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config, 'w') as f:
        f.write(config_content)
    
        print(f"Конфигурация создана: {config}")
    print(f"Классы: {classes}")
    
    return config


if __name__ == "__main__":
    # Пример использования
    trainer = PPEDetectorTrainer()
    
    # Обучение
    results = trainer.train(epochs=30, batch_size=16)
    
    if results['success']:
        print(f"Обучение завершено успешно!")
        print(f"Модель: {results['best_model']}")
        
        # Валидация
        val_results = trainer.validate_model()
        if val_results['success']:
            print(f"mAP50: {val_results['map50']:.3f}")
            print(f"mAP50-95: {val_results['map']:.3f}")
        
        # Тест на изображении
        trainer.predict_sample("data/images/val/sample.jpg", save_result=True)
    
    else:
        print(f"Ошибка обучения: {results.get('error', 'Unknown error')}")
