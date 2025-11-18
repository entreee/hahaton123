"""
Скрипт для поиска файлов данных и проверки их расположения.
"""

import os
from pathlib import Path

def find_files(root_dir, extensions, max_depth=5):
    """Находит все файлы с указанными расширениями."""
    files = []
    root = Path(root_dir)
    if not root.exists():
        return files
    
    for ext in extensions:
        for depth in range(max_depth + 1):
            pattern = "*/" * depth + f"*{ext}"
            files.extend(root.glob(pattern))
            files.extend(root.glob(pattern.upper()))
    
    return files

print("=" * 70)
print("POISK FAILOV DANYKH")
print("=" * 70)

# Ищем изображения
print("\nPoisk izobrazheniy...")
image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
images = find_files('data', image_exts, max_depth=5)
print(f"  Naydeno izobrazheniy: {len(images)}")

if images:
    # Группируем по папкам
    folders = {}
    for img in images[:20]:  # Показываем первые 20
        folder = str(img.parent)
        if folder not in folders:
            folders[folder] = 0
        folders[folder] += 1
    
    print("\n  Raspolozhenie (pervye 20):")
    for folder, count in sorted(folders.items(), key=lambda x: x[1], reverse=True):
        print(f"    {folder}: {count} izobrazheniy")

# Ищем разметку
print("\nPoisk razmetki...")
label_files = find_files('data', ['.txt'], max_depth=5)
print(f"  Naydeno razmetok: {len(label_files)}")

if label_files:
    # Группируем по папкам
    folders = {}
    for label in label_files[:20]:  # Показываем первые 20
        folder = str(label.parent)
        if folder not in folders:
            folders[folder] = 0
        folders[folder] += 1
    
    print("\n  Raspolozhenie (pervye 20):")
    for folder, count in sorted(folders.items(), key=lambda x: x[1], reverse=True):
        print(f"    {folder}: {count} razmetok")

# Проверяем правильную структуру
print("\n" + "=" * 70)
print("PROVERKA PRAVIL'NOY STRUKTURY")
print("=" * 70)

required_structure = {
    'data/images/train': 'izobrazheniya dlya obucheniya',
    'data/images/val': 'izobrazheniya dlya validatsii',
    'data/labels/train': 'razmetka dlya obucheniya',
    'data/labels/val': 'razmetka dlya validatsii'
}

for path_str, desc in required_structure.items():
    path = Path(path_str)
    if path.exists():
        # Считаем файлы
        if 'images' in path_str:
            files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.jpeg'))
        else:
            files = list(path.glob('*.txt'))
        print(f"OK: {path_str}/ ({len(files)} faylov) - {desc}")
    else:
        print(f"MISSING: {path_str}/ - {desc}")

# Проверяем соответствие имен
print("\n" + "=" * 70)
print("PROVERKA SOOTVETSTVIYA IMEN")
print("=" * 70)

train_images_dir = Path('data/images/train')
train_labels_dir = Path('data/labels/train')

if train_images_dir.exists() and train_labels_dir.exists():
    image_files = {f.stem: f for f in train_images_dir.glob('*.jpg')}
    label_files = {f.stem: f for f in train_labels_dir.glob('*.txt')}
    
    images_without_labels = set(image_files.keys()) - set(label_files.keys())
    labels_without_images = set(label_files.keys()) - set(image_files.keys())
    
    if images_without_labels:
        print(f"  Izobrazheniy bez razmetki: {len(images_without_labels)}")
        if len(images_without_labels) <= 10:
            for name in list(images_without_labels)[:10]:
                print(f"    - {name}")
    
    if labels_without_images:
        print(f"  Razmetok bez izobrazheniy: {len(labels_without_images)}")
        if len(labels_without_images) <= 10:
            for name in list(labels_without_images)[:10]:
                print(f"    - {name}")
    
    if not images_without_labels and not labels_without_images:
        print("  Vse fayly sootvetstvuyut!")
else:
    print("  Papki ne sushchestvuyut - ne mozhem proverit' sootvetstvie")

print("\n" + "=" * 70)
print("REKOMENDATSIYA")
print("=" * 70)
print("Esli fayly nakhodyatsya ne v pravil'nykh papkakh:")
print("1. Peremestite izobrazheniya v data/images/train/ i data/images/val/")
print("2. Peremestite razmetku v data/labels/train/ i data/labels/val/")
print("3. Ubeдитесь, chto imena faylov sovpadayut (image001.jpg i image001.txt)")

