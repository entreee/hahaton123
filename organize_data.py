"""
Скрипт для организации файлов данных в правильную структуру.
"""

import sys
import shutil
from pathlib import Path

# Исправление кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def create_structure():
    """Создает необходимую структуру папок."""
    required_dirs = [
        'data/images/train',
        'data/images/val',
        'data/labels/train',
        'data/labels/val'
    ]
    
    print("Sozdanie struktury papok...")
    for dir_path in required_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  OK: {dir_path}/")
    print("Struktura sozdana!")

def check_and_organize():
    """Проверяет и предлагает организовать файлы."""
    print("=" * 70)
    print("ORGANIZATSIYA DANYKH")
    print("=" * 70)
    
    # Создаем структуру
    create_structure()
    
    print("\n" + "=" * 70)
    print("INSTRUKTSIYA")
    print("=" * 70)
    print("\n1. RAZMESHCHENIE IZOBRAZHENIY:")
    print("   - Train izobrazheniya -> data/images/train/")
    print("   - Val izobrazheniya -> data/images/val/")
    print("\n2. RAZMESHCHENIE RAZMETKI:")
    print("   - Train razmetka -> data/labels/train/")
    print("   - Val razmetka -> data/labels/val/")
    print("\n3. VAZHNO:")
    print("   - Imena faylov dolzhny sovpadat'")
    print("   - Primer: image001.jpg i image001.txt")
    print("   - Format razmetki: YOLO (class_id center_x center_y width height)")
    print("\n4. PROVERKA:")
    print("   - Zapustite: python check_data_structure.py")
    print("   - Ili: python view_annotations.py")

if __name__ == "__main__":
    check_and_organize()

