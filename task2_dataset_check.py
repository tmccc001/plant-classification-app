import os
import hashlib
from PIL import Image
from tabulate import tabulate
from labels import label_map

DATA_DIR = "data"

def file_hash(filepath):
    """Return MD5 hash of a file for duplicate checking."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def count_images(split_dir):
    counts = {}
    corrupted = []
    duplicates = []
    if not os.path.exists(split_dir):
        return counts, corrupted, duplicates
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = os.listdir(class_dir)
        valid_count = 0
        seen_hashes = {}
        for file in files:
            path = os.path.join(class_dir, file)
            try:
                img = Image.open(path)
                img.verify()
                h = file_hash(path)
                if h in seen_hashes:
                    duplicates.append((path, seen_hashes[h]))
                else:
                    seen_hashes[h] = path
                valid_count += 1
            except:
                corrupted.append(path)
        counts[class_name] = valid_count
    return counts, corrupted, duplicates

if __name__ == "__main__":
    train_counts, corrupted_train, dup_train = count_images(os.path.join(DATA_DIR, "train"))
    test_counts, corrupted_test, dup_test = count_images(os.path.join(DATA_DIR, "test"))

    corrupted = corrupted_train + corrupted_test
    duplicates = dup_train + dup_test

    # summary table
    rows = []
    all_classes = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    for class_name in all_classes:
        bilingual = label_map.get(class_name, class_name)
        train = train_counts.get(class_name, 0)
        test = test_counts.get(class_name, 0)
        total = train + test
        rows.append([bilingual, train, test, total])

    print("=== Dataset Summary ===")
    print(tabulate(rows, headers=["Class", "Train", "Test", "Total"], tablefmt="fancy_grid"))

    print(f"\nTotal vegetation classes used: {len(all_classes)}")

    print("\n=== Duplicates Found ===")
    if duplicates:
        for d in duplicates:
            print(f"Duplicate: {d[0]} and {d[1]}")
    else:
        print("No duplicates found.")

    print("\n=== Corrupted Files ===")
    if corrupted:
        for c in corrupted:
            print(c)
    else:
        print("No corrupted files found.")

    print("\n=== Classes Detected ===")
    for c in all_classes:
        print(label_map.get(c, c))
