"""
Автоматическое извлечение кадров из всех видео в папке.

Поместите все видео в папку videos/ и запустите этот скрипт.
Он автоматически извлечет кадры из всех видео и сохранит их в data/images/train/

Использование:
    python auto_extract_frames.py
"""

import cv2
import os
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, step=30):
    """
    Извлекает кадры из видео с заданным шагом.
    
    Args:
        video_path: Путь к видео файлу
        output_dir: Директория для сохранения кадров
        step: Извлекать каждый N-й кадр (по умолчанию 30)
    
    Returns:
        Количество сохраненных кадров
    """
    # Открываем видео
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"⚠️  Не удалось открыть видео: {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    # Получаем имя видео файла для префикса
    video_name = video_path.stem
    
    print(f"  Обработка: {video_path.name}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Сохраняем каждый N-й кадр
        if frame_count % step == 0:
            # Формируем имя файла с префиксом имени видео
            frame_filename = output_dir / f"{video_name}_frame_{saved_count:06d}.jpg"
            
            # Сохраняем кадр
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"  ✓ Сохранено кадров: {saved_count} (из {frame_count} всего)")
    return saved_count


def auto_extract_frames(videos_dir="videos", output_dir="data/images/train", step=30):
    """
    Автоматически извлекает кадры из всех видео в указанной папке.
    
    Args:
        videos_dir: Папка с видео файлами (по умолчанию: videos/)
        output_dir: Папка для сохранения кадров (по умолчанию: data/images/train/)
        step: Извлекать каждый N-й кадр (по умолчанию: 30)
    """
    videos_path = Path(videos_dir)
    output_path = Path(output_dir)
    
    # Создаем директорию для сохранения, если её нет
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Проверяем существование папки с видео
    if not videos_path.exists():
        print(f"❌ Ошибка: папка '{videos_dir}' не найдена!")
        print(f"\nСоздайте папку '{videos_dir}' и поместите туда видео файлы:")
        print(f"  - {videos_dir}/")
        print(f"    ├── video1.mp4")
        print(f"    ├── video2.avi")
        print(f"    └── ...")
        return
    
    # Поддерживаемые форматы видео
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
    
    # Ищем все видео файлы
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(f"*{ext}"))
        video_files.extend(videos_path.glob(f"*{ext.upper()}"))
    
    if len(video_files) == 0:
        print(f"❌ Ошибка: не найдено видео файлов в папке '{videos_dir}'!")
        print(f"\nПоддерживаемые форматы: {', '.join(video_extensions)}")
        print(f"Поместите видео файлы в папку '{videos_dir}/' и запустите скрипт снова.")
        return
    
    print("=" * 60)
    print("АВТОМАТИЧЕСКОЕ ИЗВЛЕЧЕНИЕ КАДРОВ ИЗ ВИДЕО")
    print("=" * 60)
    print(f"Папка с видео: {videos_dir}/")
    print(f"Папка для кадров: {output_dir}/")
    print(f"Шаг извлечения: каждый {step}-й кадр")
    print(f"Найдено видео файлов: {len(video_files)}")
    print("=" * 60)
    print()
    
    total_saved = 0
    
    # Обрабатываем каждое видео
    for i, video_file in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] {video_file.name}")
        saved = extract_frames_from_video(video_file, output_path, step)
        total_saved += saved
        print()
    
    print("=" * 60)
    print("✅ ИЗВЛЕЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"Всего обработано видео: {len(video_files)}")
    print(f"Всего сохранено кадров: {total_saved}")
    print(f"Кадры сохранены в: {output_dir}/")
    print()
    print("📝 Следующие шаги:")
    print("1. Разметьте кадры с помощью LabelImg или другого инструмента")
    print("2. Сохраните разметку в формате YOLO в папку: data/labels/train/")
    print("3. Запустите обучение: python run_training.py")
    print("=" * 60)


def main():
    """Основная функция."""
    import sys
    
    # Можно указать папку с видео как аргумент
    videos_dir = "videos"
    if len(sys.argv) > 1:
        videos_dir = sys.argv[1]
    
    # Можно указать шаг извлечения как второй аргумент
    step = 30
    if len(sys.argv) > 2:
        try:
            step = int(sys.argv[2])
        except ValueError:
            print("⚠️  Неверный шаг, используется значение по умолчанию: 30")
    
    auto_extract_frames(videos_dir, "data/images/train", step)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Извлечение прервано пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

