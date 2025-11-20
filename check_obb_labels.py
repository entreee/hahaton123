"""
Проверка формата OBB аннотаций.

Использование:
    python check_obb_labels.py
    python check_obb_labels.py --labels data/labels/train
"""

import argparse
from pathlib import Path


def check_obb_format(label_file: Path) -> tuple:
    """
    Проверяет формат OBB аннотации.
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            warnings.append("Файл пуст")
            return False, errors, warnings
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # OBB формат: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 значений)
            if len(parts) != 9:
                errors.append(f"Строка {line_num}: ожидается 9 значений для OBB, получено {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Проверка координат
                if len(coords) != 8:
                    errors.append(f"Строка {line_num}: должно быть 8 координат (4 точки), получено {len(coords)}")
                    continue
                
                # Проверка диапазона координат
                for i, coord in enumerate(coords):
                    if coord < 0 or coord > 1:
                        errors.append(f"Строка {line_num}: координата {i+1} вне диапазона [0,1]: {coord}")
                
                # Проверка, что точки образуют валидный четырехугольник
                x_coords = [coords[i] for i in range(0, 8, 2)]  # x1, x2, x3, x4
                y_coords = [coords[i] for i in range(1, 8, 2)]  # y1, y2, y3, y4
                
                if len(set(x_coords)) < 2 or len(set(y_coords)) < 2:
                    warnings.append(f"Строка {line_num}: точки могут быть вырожденными (не образуют четырехугольник)")
                
            except ValueError as e:
                errors.append(f"Строка {line_num}: ошибка парсинга - {e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
        
    except Exception as e:
        errors.append(f"Ошибка чтения файла: {e}")
        return False, errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Проверка формата OBB аннотаций"
    )
    
    parser.add_argument(
        "--labels", "-l",
        type=str,
        default="data/labels/train",
        help="Папка с аннотациями (по умолчанию: data/labels/train)"
    )
    
    args = parser.parse_args()
    
    labels_dir = Path(args.labels)
    
    if not labels_dir.exists():
        print(f"❌ Папка не найдена: {labels_dir}")
        return 1
    
    label_files = list(labels_dir.glob("*.txt"))
    
    if len(label_files) == 0:
        print(f"❌ Не найдено файлов аннотаций в {labels_dir}")
        return 1
    
    print("=" * 70)
    print("ПРОВЕРКА ФОРМАТА OBB АННОТАЦИЙ")
    print("=" * 70)
    print(f"Папка: {labels_dir}")
    print(f"Найдено файлов: {len(label_files)}")
    print("-" * 70)
    
    total_valid = 0
    total_invalid = 0
    total_errors = 0
    total_warnings = 0
    
    for label_file in label_files:
        is_valid, errors, warnings = check_obb_format(label_file)
        
        if is_valid:
            total_valid += 1
        else:
            total_invalid += 1
            print(f"\n❌ {label_file.name}:")
            for error in errors:
                print(f"   ERROR: {error}")
                total_errors += 1
            for warning in warnings:
                print(f"   WARNING: {warning}")
                total_warnings += 1
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("=" * 70)
    print(f"Валидных файлов: {total_valid}/{len(label_files)}")
    print(f"Невалидных файлов: {total_invalid}/{len(label_files)}")
    print(f"Всего ошибок: {total_errors}")
    print(f"Всего предупреждений: {total_warnings}")
    
    if total_invalid == 0:
        print("\n✅ Все аннотации в правильном OBB формате!")
        return 0
    else:
        print(f"\n⚠️  Найдено проблем в {total_invalid} файлах")
        print("Исправьте ошибки перед обучением")
        return 1


if __name__ == "__main__":
    exit(main())

