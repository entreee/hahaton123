"""
Модуль для инференса (использования) обученной модели.

Содержит функции для:
- Детекции на изображениях
- Обработки видео файлов
- Детекции в реальном времени с камеры
- Визуализации результатов
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Настройки отображения
CLASS_NAMES = {
    0: "helmet (каска)",
    1: "vest (жилет)"
}

CLASS_COLORS = {
    0: (0, 165, 255),    # Оранжевый для каски (BGR)
    1: (0, 255, 255)     # Желтый для жилета (BGR)
}

CONFIDENCE_THRESHOLD = 0.5


class PPEDetector:
    """
    Класс для инференса модели детекции СИЗ.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = CONFIDENCE_THRESHOLD,
        device: str = "auto"
    ):
        """
        Инициализация детектора.
        
        Args:
            model_path: Путь к обученной модели (.pt)
            conf_threshold: Порог уверенности для детекции
            device: Устройство ('cpu', '0', 'auto')
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Загрузка модели
        self.model = YOLO(str(model_path))
        
        # Устройство
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model.to(device)
        
        print(f"✅ Детектор загружен: {model_path}")
        print(f"📱 Устройство: {device}")
        print(f"🎯 Порог уверенности: {conf_threshold}")
    
    def detect_image(
        self,
        image_path: str,
        save_result: bool = False,
        output_dir: str = "output/detections",
        show_confidence: bool = True,
        line_thickness: int = 2
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Детекция на одном изображении.
        
        Args:
            image_path: Путь к изображению
            save_result: Сохранить результат с детекциями
            output_dir: Папка для сохранения результатов
            show_confidence: Показывать уверенность в подписи
            line_thickness: Толщина линий bounding box
            
        Returns:
            (изображение с детекциями, список детекций)
        """
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        original_image = image.copy()
        height, width = image.shape[:2]
        
        # Детекция
        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device
        )
        
        detections = []
        boxes = results[0].boxes
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Координаты
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Класс и уверенность
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                
                if confidence >= self.conf_threshold:
                    # Цвет и название класса
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Белый по умолчанию
                    class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
                    
                    # Рисуем bounding box
                    cv2.rectangle(
                        image, (x1, y1), (x2, y2),
                        color, line_thickness
                    )
                    
                    # Подпись
                    label = f"{class_name}"
                    if show_confidence:
                        label += f": {confidence:.2f}"
                    
                    # Фон для подписи
                    label_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    cv2.rectangle(
                        image,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        color, -1
                    )
                    
                    # Текст подписи
                    cv2.putText(
                        image, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2  # Белый текст
                    )
                    
                    # Сохраняем информацию о детекции
                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        # Сохранение результата
        if save_result:
            output_path = Path(output_dir) / Path(image_path).name
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)
            print(f"💾 Результат сохранен: {output_path}")
        
        # Конвертация BGR -> RGB для matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb, detections
    
    def detect_video(
        self,
        video_path: str,
        output_path: str = None,
        conf_threshold: float = None,
        show_progress: bool = True
    ) -> Path:
        """
        Детекция на видео файле.
        
        Args:
            video_path: Путь к входному видео
            output_path: Путь для сохранения результата (если None - output/detected_video.mp4)
            conf_threshold: Порог уверенности (если None - использует self.conf_threshold)
            show_progress: Показывать прогресс обработки
            
        Returns:
            Путь к выходному видео с детекциями
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        # Путь для сохранения
        if output_path is None:
            output_dir = Path("output/videos")
            output_dir.mkdir(parents=True, exist_ok=True)
            video_name = Path(video_path).stem
            output_path = output_dir / f"{video_name}_detected.mp4"
        
        # Открытие видео
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        # Параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"🎬 Обработка видео: {video_path}")
        print(f"📏 Размер: {width}x{height}, FPS: {fps}, кадров: {total_frames}")
        print(f"💾 Результат: {output_path}")
        
        # Настройка записи
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция на кадре
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                verbose=False,
                device=self.device
            )
            
            # Рисуем детекции (используем тот же код, что и для изображений)
            boxes = results[0].boxes
            frame_detections = 0
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    if confidence >= conf_threshold:
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{CLASS_NAMES.get(class_id, 'unknown')}"
                        if confidence < 0.9:  # Показываем уверенность для неуверенных
                            label += f": {confidence:.2f}"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(
                            frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1
                        )
                        cv2.putText(
                            frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )
                        
                        frame_detections += 1
                        total_detections += 1
            
            # Информация о кадре
            cv2.putText(
                frame, f"Frame: {frame_count}/{total_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Прогресс
            if show_progress and frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                print(f"  📈 Прогресс: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Запись кадра
            out.write(frame)
            frame_count += 1
        
        # Завершение
        cap.release()
        out.release()
        
        print(f"\n✅ Видео обработано!")
        print(f"📊 Кадров обработано: {frame_count}")
        print(f"🎯 Детекций найдено: {total_detections}")
        print(f"💾 Результат сохранен: {output_path}")
        
        return output_path
    
    def detect_camera(
        self,
        camera_id: int = 0,
        conf_threshold: float = None,
        window_name: str = "PPE Detection - Real Time",
        max_frames: Optional[int] = None
    ) -> None:
        """
        Детекция в реальном времени с камеры.
        
        Args:
            camera_id: ID камеры (0 = основная веб-камера)
            conf_threshold: Порог уверенности
            window_name: Название окна
            max_frames: Максимальное количество кадров (None = бесконечно)
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Не удалось открыть камеру {camera_id}")
            print("Проверьте подключение камеры и попробуйте camera_id=1")
            return
        
        # Настройка камеры
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"📹 Камера запущена ({camera_id})")
        print(f"Нажмите 'q' для выхода, 's' для скриншота")
        print(f"Порог уверенности: {conf_threshold}")
        
        frame_count = 0
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Потеряно соединение с камерой")
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Детекция
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                verbose=False,
                device=self.device
            )
            
            # Рисуем детекции
            boxes = results[0].boxes
            frame_detections = 0
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    if confidence >= conf_threshold:
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{CLASS_NAMES.get(class_id, 'unknown')}"
                        if confidence < 0.9:
                            label += f": {confidence:.2f}"
                        
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(
                            frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1
                        )
                        cv2.putText(
                            frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )
                        
                        frame_detections += 1
            
            # Информация на кадре
            info_y = 30
            cv2.putText(
                frame, f"Frame: {frame_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            info_y += 30
            cv2.putText(
                frame, f"Detections: {frame_detections}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            info_y += 30
            cv2.putText(
                frame, "Press 'q' to quit, 's' for screenshot", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
            
            # Показ кадра
            cv2.imshow(window_name, frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Выход по нажатию 'q'")
                break
            elif key == ord('s'):
                # Скриншот
                screenshot_dir = Path("output/screenshots")
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"screenshot_{frame_count:06d}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                screenshot_count += 1
                print(f"📸 Скриншот сохранен: {screenshot_path}")
                print(f"Всего скриншотов: {screenshot_count}")
            
            frame_count += 1
        
        # Очистка
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📹 Детекция с камеры завершена")
        print(f"🎬 Обработано кадров: {frame_count}")
        print(f"📸 Сохранено скриншотов: {screenshot_count}")
    
    def batch_predict(
        self,
        image_folder: str,
        output_folder: str = "output/batch_detections",
        conf_threshold: float = None,
        save_results: bool = True
    ) -> Dict[str, int]:
        """
        Пакетная обработка папки с изображениями.
        
        Args:
            image_folder: Папка с изображениями для обработки
            output_folder: Папка для сохранения результатов
            conf_threshold: Порог уверенности
            save_results: Сохранять результаты
            
        Returns:
            Статистика обработки
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        image_path = Path(image_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_path.glob(ext))
            image_files.extend(image_path.glob(ext.upper()))
        
        if len(image_files) == 0:
            print(f"❌ Не найдено изображений в {image_folder}")
            return {'processed': 0, 'detections': 0, 'errors': 0}
        
        stats = {'processed': 0, 'detections': 0, 'errors': 0}
        
        print(f"📂 Пакетная обработка: {len(image_files)} изображений")
        print(f"💾 Результаты: {output_folder}")
        
        for i, image_file in enumerate(image_files, 1):
            try:
                # Детекция
                results, detections = self.detect_image(
                    str(image_file),
                    save_result=save_results,
                    output_dir=str(output_path),
                    show_confidence=True
                )
                
                stats['processed'] += 1
                stats['detections'] += len(detections)
                
                if i % 10 == 0:
                    progress = i / len(image_files) * 100
                    print(f"  📈 Прогресс: {progress:.1f}% ({i}/{len(image_files)})")
            
            except Exception as e:
                print(f"❌ Ошибка обработки {image_file.name}: {e}")
                stats['errors'] += 1
        
        print(f"\n✅ Пакетная обработка завершена!")
        print(f"📊 Обработано: {stats['processed']}/{len(image_files)}")
        print(f"🎯 Детекций: {stats['detections']}")
        print(f"⚠️  Ошибок: {stats['errors']}")
        
        return stats
    
    def get_model_info(self) -> Dict:
        """
        Получает информацию о модели.
        
        Returns:
            Словарь с информацией о модели
        """
        try:
            model = YOLO(self.model_path)
            info = {
                'model_path': str(self.model_path),
                'num_classes': len(model.names),
                'class_names': model.names,
                'device': self.device,
                'conf_threshold': self.conf_threshold
            }
            
            # Размер модели (примерно)
            model_size = self.model_path.stat().st_size / (1024*1024)  # MB
            info['model_size_mb'] = round(model_size, 2)
            
            print(f"ℹ️  Информация о модели:")
            print(f"  📁 Путь: {info['model_path']}")
            print(f"  📦 Размер: {info['model_size_mb']} MB")
            print(f"  🎯 Классов: {info['num_classes']}")
            print(f"  📱 Устройство: {info['device']}")
            print(f"  🔢 Классы: {list(info['class_names'].values())}")
            
            return info
            
        except Exception as e:
            print(f"❌ Ошибка получения информации: {e}")
            return {}


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Dict = CLASS_NAMES,
    class_colors: Dict = CLASS_COLORS,
    figsize: Tuple[float, float] = (12, 8)
) -> plt.Figure:
    """
    Визуализирует детекции на изображении с помощью matplotlib.
    
    Args:
        image: Изображение (RGB или BGR)
        detections: Список детекций
        class_names: Словарь классов
        class_colors: Словарь цветов
        figsize: Размер фигуры
        
    Returns:
        Matplotlib фигура
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Конвертация BGR -> RGB если нужно
    if len(image.shape) == 3 and image.shape[2] == 3:
        if image[0, 0, 0] + image[0, 0, 1] + image[0, 0, 2] > 500:  # Вероятно BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ax.imshow(image)
    ax.set_title("Детекция СИЗ", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Рисуем bounding boxes
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Цвет и название
        color = class_colors.get(class_id, 'white')
        class_name = class_names.get(class_id, f'class_{class_id}')
        
        # Bounding box
        rect = Rectangle(
            (x1, y1), (x2 - x1), (y2 - y1),
            linewidth=2, edgecolor=color, facecolor='none',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Подпись
        label = f"{class_name}: {confidence:.2f}"
        ax.text(
            x1, y1 - 5, label,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            fontsize=10, fontweight='bold', color='white',
            verticalalignment='top'
        )
    
    # Легенда
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=class_colors[0], label=class_names[0]) for i in range(1)
    ] + [
        plt.Rectangle((0,0),1,1, facecolor=class_colors[1], label=class_names[1]) for i in range(1)
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Пример использования
    from pathlib import Path
    
    # Инициализация
    detector = PPEDetector("models/ppe_model/weights/best.pt")
    
    # Информация о модели
    info = detector.get_model_info()
    
    # Тест на изображении
    if Path("data/images/val").exists():
        test_images = list(Path("data/images/val").glob("*.jpg"))
        if test_images:
            result_img, detections = detector.detect_image(str(test_images[0]))
            
            # Визуализация
            fig = visualize_detections(result_img, detections)
            plt.show()
            
            print(f"Найдено детекций: {len(detections)}")
            for det in detections:
                print(f"  {det['class_name']}: {det['confidence']:.2f}")
