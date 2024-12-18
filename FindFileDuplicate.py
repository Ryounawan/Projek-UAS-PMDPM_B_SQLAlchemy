import os
import imagehash
from PIL import Image

def calculate_image_hash(image_path):
    """Menghitung perceptual hash (pHash) dari gambar."""
    img = Image.open(image_path)
    return imagehash.phash(img)

def is_image_file(file_name):
    """Memeriksa apakah sebuah file adalah file gambar berdasarkan ekstensi."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return os.path.splitext(file_name)[1].lower() in valid_extensions

def find_duplicates(train_folder, test_folder, hash_threshold=5):
    """Mencari gambar yang mirip berdasarkan perceptual hash."""
    if not os.path.exists(train_folder) or not os.path.exists(test_folder):
        raise FileNotFoundError("One or both folders do not exist. Please check the paths.")

    if not os.listdir(train_folder) or not os.listdir(test_folder):
        raise ValueError("One or both folders are empty. Please check the contents.")

    train_hashes = {
        calculate_image_hash(os.path.join(train_folder, f)): f
        for f in os.listdir(train_folder) if is_image_file(f)
    }
    test_hashes = {
        calculate_image_hash(os.path.join(test_folder, f)): f
        for f in os.listdir(test_folder) if is_image_file(f)
    }

    duplicates = []
    for train_hash, train_file in train_hashes.items():
        for test_hash, test_file in test_hashes.items():
            if train_hash - test_hash <= hash_threshold:
                duplicates.append((train_file, test_file))

    return duplicates

train_folder = "train_data"
test_folder = "test_data"

try:
    duplicates = find_duplicates(train_folder, test_folder)
    if duplicates:
        print("Duplicates found:")
        for train_file, test_file in duplicates:
            print(f"Train: {os.path.join(train_folder, train_file)}")
            print(f"Test: {os.path.join(test_folder, test_file)}")
    else:
        print("No duplicates found.")
except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
