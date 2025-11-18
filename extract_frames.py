"""
Скрипт для извлечения кадров из видео для последующей разметки.

Использование:
    python extract_frames.py --video path/to/video.mp4 --output data/images/train --step 30
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, step=30):
    """
    Извлекает кадры из видео с заданным шагом.
    
    Args:
        video_path (str): Путь к входному видео
        output_dir (str): Директория для сохранения кадров
        step (int): Извлекать каждый N-й кадр (по умолчанию 30)
    """
    # Создаем директорию для сохранения, если её нет
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    frame_count = 0
    saved_count = 0
    
    print(f"Начинаю извлечение кадров из {video_path}")
    print(f"Шаг извлечения: каждый {step}-й кадр")
    print(f"Кадры будут сохранены в: {output_dir}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Сохраняем каждый N-й кадр
        if frame_count % step == 0:
            # Формируем имя файла с нумерацией
            frame_filename = os.path.join(
                output_dir, 
                f"frame_{saved_count:06d}.jpg"
            )
            
            # Сохраняем кадр
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Сохранено кадров: {saved_count}")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nИзвлечение завершено!")
    print(f"Всего кадров в видео: {frame_count}")
    print(f"Сохранено кадров: {saved_count}")
    print(f"Кадры сохранены в: {output_dir}")


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Извлечение кадров из видео для разметки"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Путь к входному видео файлу"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/images/train",
        help="Директория для сохранения кадров (по умолчанию: data/images/train)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=30,
        help="Извлекать каждый N-й кадр (по умолчанию: 30)"
    )
    
    args = parser.parse_args()
    
    # Проверяем существование видео файла
    if not os.path.exists(args.video):
        print(f"Ошибка: файл {args.video} не найден!")
        return
    
    try:
        extract_frames(args.video, args.output, args.step)
    except Exception as e:
        print(f"Ошибка при извлечении кадров: {e}")


if __name__ == "__main__":
    main()

