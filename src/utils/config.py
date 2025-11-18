"""
Конфигурация проекта детекции СИЗ.

Централизованное управление всеми настройками.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectConfig:
    """Основная конфигурация проекта."""
    
    # Пути
    project_root: Path = Path(".").resolve()
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    output_dir: Path = project_root / "output"
    videos_dir: Path = project_root / "videos"
    config_dir: Path = project_root / "config"
    notebooks_dir: Path = project_root / "notebooks"
    src_dir: Path = project_root / "src"
    
    # Данные
    image_extensions: List[str] = None
    video_extensions: List[str] = None
    classes: Dict[int, str] = None
    class_colors: Dict[int, tuple] = None
    
    # Модель
    model_name: str = "yolov8n.pt"
    experiment_name: str = "ppe_detection"
    conf_threshold: float = 0.5
    
    # Обучение
    epochs: int = 30
    img_size: int = 640
    batch_size: int = 16
    patience: int = 10
    workers: int = 8
    device: str = "auto"
    
    # Разметка
    val_ratio: float = 0.2
    random_seed: int = 42
    prelabel_conf_threshold: float = 0.3
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        if self.image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
        
        if self.classes is None:
            self.classes = {0: 'helmet', 1: 'vest'}
        
        if self.class_colors is None:
            self.class_colors = {
                0: (0, 165, 255),  # Оранжевый для каски (BGR)
                1: (0, 255, 255)   # Желтый для жилета (BGR)
            }
        
        # Создание директорий
        self._create_directories()
        
        # Определение устройства
        self._detect_device()
    
    def _create_directories(self):
        """Создает необходимые директории."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.output_dir,
            self.videos_dir,
            self.config_dir,
            self.src_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("📁 Директории созданы/проверены")
    
    def _detect_device(self):
        """Определяет доступное устройство."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    self.device = str(gpu_count)  # Multi-GPU
                else:
                    self.device = "0"  # Single GPU
                print(f"🔥 GPU доступно: {self.device}")
            else:
                self.device = "cpu"
                print("💻 Используется CPU")
        except ImportError:
            self.device = "cpu"
            print("⚠️  PyTorch не установлен, используется CPU")
    
    def load_dataset_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Загружает конфигурацию датасета из YAML файла.
        
        Args:
            config_path: Путь к YAML файлу (если None - config/ppe_data.yaml)
            
        Returns:
            Словарь с конфигурацией датасета
        """
        if config_path is None:
            config_path = self.config_dir / "ppe_data.yaml"
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            self.create_dataset_config(config_path)
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"📄 Конфигурация загружена: {config_file}")
            print(f"Классов: {config.get('nc', 'N/A')}")
            print(f"Классы: {config.get('names', 'N/A')}")
            
            return config
            
        except Exception as e:
            print(f"❌ Ошибка загрузки конфигурации: {e}")
            return {}
    
    def create_dataset_config(self, config_path: Optional[str] = None) -> Path:
        """
        Создает стандартную конфигурацию датасета.
        
        Args:
            config_path: Путь для сохранения (если None - config/ppe_data.yaml)
            
        Returns:
            Путь к созданному файлу
        """
        if config_path is None:
            config_path = self.config_dir / "ppe_data.yaml"
        
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_content = f"""# Конфигурация датасета для детекции СИЗ
# Классы: 0=helmet (защитная каска), 1=vest (сигнальный жилет)

path: {self.data_dir}  # Корневая папка с данными
train: images/train     # Путь к обучающим изображениям (относительно path)
val: images/val         # Путь к валидационным изображениям (относительно path)

# Количество классов
nc: {len(self.classes)}

# Имена классов
names:
"""
        
        for class_id, class_name in self.classes.items():
            config_content += f"  {class_id}: {class_name}\n"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            print(f"✅ Конфигурация создана: {config_file}")
            print(f"📊 Классы: {list(self.classes.values())}")
            
            return config_file
            
        except Exception as e:
            print(f"❌ Ошибка создания конфигурации: {e}")
            return config_file
    
    def create_classes_file(self, classes_path: Optional[str] = None) -> Path:
        """
        Создает файл классов для LabelImg.
        
        Args:
            classes_path: Путь для сохранения (если None - data/predefined_classes.txt)
            
        Returns:
            Путь к созданному файлу
        """
        if classes_path is None:
            classes_path = self.data_dir / "predefined_classes.txt"
        
        classes_file = Path(classes_path)
        classes_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(classes_file, 'w', encoding='utf-8') as f:
                for class_name in self.classes.values():
                    f.write(f"{class_name}\n")
            
            print(f"✅ Файл классов создан: {classes_file}")
            print(f"📝 Классы для LabelImg: {list(self.classes.values())}")
            
            return classes_file
            
        except Exception as e:
            print(f"❌ Ошибка создания файла классов: {e}")
            return classes_file
    
    def get_paths_summary(self) -> Dict[str, Path]:
        """
        Возвращает сводку по путям проекта.
        
        Returns:
            Словарь с основными путями
        """
        return {
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'models_dir': self.models_dir,
            'output_dir': self.output_dir,
            'videos_dir': self.videos_dir,
            'config_dir': self.config_dir,
            'dataset_config': self.config_dir / "ppe_data.yaml",
            'classes_file': self.data_dir / "predefined_classes.txt"
        }
    
    def validate_paths(self) -> Dict[str, bool]:
        """
        Проверяет существование основных путей.
        
        Returns:
            Словарь {путь: существует}
        """
        paths = self.get_paths_summary()
        validation = {}
        
        print("🔍 ПРОВЕРКА ПУТЕЙ ПРОЕКТА")
        print("-" * 40)
        
        for name, path in paths.items():
            exists = path.exists()
            validation[name] = exists
            
            status = "✅" if exists else "❌"
            print(f"{status} {name}: {path}")
        
        # Подробная проверка данных
        if self.data_dir.exists():
            img_count = len(list(self.data_dir.rglob("*.jpg"))) + \
                       len(list(self.data_dir.rglob("*.png")))
            label_count = len(list(self.data_dir.rglob("*.txt")))
            
            print(f"\n📊 Данные: {img_count} изображений, {label_count} разметок")
        
        missing_count = sum(1 for exists in validation.values() if not exists)
        if missing_count == 0:
            print("\n🎉 Все пути корректны!")
        else:
            print(f"\n⚠️  Отсутствует {missing_count} путей")
        
        return validation


# Глобальная конфигурация проекта
config = ProjectConfig()


if __name__ == "__main__":
    # Демонстрация использования
    print("=== КОНФИГУРАЦИЯ ПРОЕКТА ===")
    
    # Проверка путей
    validation = config.validate_paths()
    
    # Создание конфигурации (если нужно)
    if not config.config_dir.exists() or not (config.config_dir / "ppe_data.yaml").exists():
        config.create_dataset_config()
    
    # Создание файла классов
    config.create_classes_file()
    
    # Загрузка конфигурации датасета
    dataset_config = config.load_dataset_config()
    
    print(f"\n🎯 Конфигурация готова!")
    print(f"Классов: {dataset_config.get('nc', 0)}")
    print(f"Устройство: {config.device}")
    
    # Пример путей
    paths = config.get_paths_summary()
    print(f"\n📁 Основные пути:")
    for name, path in paths.items():
        status = "✅" if path.exists() else "⚠️"
        print(f"  {status} {name}: {path}")
