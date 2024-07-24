import os
import shutil
import random
from tqdm import tqdm  # Ensure the correct import from standard module

# Define paths
train_path_img = "./yolo_data/images/train/"
train_path_label = "./yolo_data/labels/train/"
val_path_img = "./yolo_data/images/val/"
val_path_label = "./yolo_data/labels/val/"
test_path = "./yolo_data/test"

def train_test_split(path, neg_path=None, split=0.2):
    print("------ PROCESS STARTED -------")

    # List and remove duplicate file names
    files = list(set([name[:-4] for name in os.listdir(path)]))
    print(f"--- This folder has a total number of {len(files)} images---")
    
    random.seed(42)
    random.shuffle(files)

    # Calculate split sizes
    test_size = int(len(files) * split)
    train_size = len(files) - test_size

    # Create directories
    os.makedirs(train_path_img, exist_ok=True)
    os.makedirs(train_path_label, exist_ok=True)
    os.makedirs(val_path_img, exist_ok=True)
    os.makedirs(val_path_label, exist_ok=True)

    # Copy images and labels to training directory
    for filex in tqdm(files[:train_size]):
        if filex == 'classes':
            continue
        try:            
            shutil.copy2(path + filex + '.jpg', f"{train_path_img}/" + filex + '.jpg')# jpg-jpeg-png
            shutil.copy2(path + filex + '.txt', f"{train_path_label}/" + filex + '.txt')
        except FileNotFoundError:
            print(f"File {filex} not found.")

    print(f"------ Training data created with 80% split: {len(files[:train_size])} images -------")

    # Add negative images if provided
    if neg_path:
        neg_images = list(set([name[:-4] for name in os.listdir(neg_path)]))
        for filex in tqdm(neg_images):
            try:
                shutil.copy2(neg_path + filex + ".jpg", f"{train_path_img}/" + filex + '.jpg')
            except FileNotFoundError:
                print(f"Negative file {filex} not found.")

        print(f"------ Total {len(neg_images)} negative images added to the training data -------")
        print(f"------ TOTAL Training data created with {len(files[:train_size]) + len(neg_images)} images -------")

    # Copy images and labels to validation directory
    for filex in tqdm(files[train_size:]):
        if filex == 'classes':
            continue
        try:
            shutil.copy2(path + filex + '.jpg', f"{val_path_img}/" + filex + '.jpg')
            shutil.copy2(path + filex + '.txt', f"{val_path_label}/" + filex + '.txt')
        except FileNotFoundError:
            print(f"File {filex} not found.")

    print(f"------ Testing data created with a total of {len(files[train_size:])} images ----------")
    print("------ TASK COMPLETED -------")

# Split the data into train-test and create train.txt and test.txt files
train_test_split('data/')  # without negative images
# train_test_split('./data/', './negative_images/')  # if you want to feed negative images
