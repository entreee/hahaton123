"""
Скрипт для проверки структуры данных и расположения файлов.
"""

import sys
import os
from pathlib import Path

# Исправление кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_utils import check_data_structure

print("=" * 70)
print("PROVERKA STRUKTURY DANNYKH")
print("=" * 70)

try:
    stats = check_data_structure('data')
    
    print("\n" + "=" * 70)
    print("ITOGOVAYA STATISTIKA")
    print("=" * 70)
    print(f"Train:")
    print(f"  - Izobrazheniy: {stats['train_images']}")
    print(f"  - Razmetok: {stats['train_labels']}")
    print(f"  - Bez razmetki: {stats['missing_labels']} izobrazheniy")
    print(f"  - Lishnikh razmetok: {stats['extra_labels']}")
    print(f"\nVal:")
    print(f"  - Izobrazheniy: {stats['val_images']}")
    print(f"  - Razmetok: {stats['val_labels']}")
    print(f"\nVsego:")
    print(f"  - Izobrazheniy: {stats['total_images']}")
    print(f"  - Razmetok: {stats['total_labels']}")
    
    print("\n" + "=" * 70)
    print("PROVERKA RASPOLOZHENIYA FAILOV")
    print("=" * 70)
    
    data_path = Path("data")
    required_dirs = [
        "images/train",
        "images/val",
        "labels/train",
        "labels/val"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = data_path / dir_path
        if full_path.exists():
            print(f"OK: {dir_path}/")
        else:
            print(f"MISSING: {dir_path}/")
            all_ok = False
    
    if all_ok:
        print("\nVse neobkhodimye papki sushchestvuyut!")
    else:
        print("\nVNIMANIE: Nekotorye papki otsutstvuyut!")
        print("Sozdayte nedostayushchie papki:")
        for dir_path in required_dirs:
            full_path = data_path / dir_path
            if not full_path.exists():
                print(f"  mkdir -p {full_path}")
    
    print("\n" + "=" * 70)
    print("REKOMENDUEMAYA STRUKTURA")
    print("=" * 70)
    print("data/")
    print("  images/")
    print("    train/     <- izobrazheniya dlya obucheniya")
    print("    val/       <- izobrazheniya dlya validatsii")
    print("  labels/")
    print("    train/     <- razmetka dlya obucheniya (.txt fayly)")
    print("    val/       <- razmetka dlya validatsii (.txt fayly)")
    print("\nVAZHNO:")
    print("- Imena faylov izobrazheniy i razmetki dolzhny sovpadat'")
    print("  Primer: image001.jpg i image001.txt")
    print("- Format razmetki: YOLO (class_id center_x center_y width height)")
    
except Exception as e:
    print(f"Oshibka: {e}")
    import traceback
    traceback.print_exc()

