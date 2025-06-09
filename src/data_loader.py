import pandas as pd
import numpy as np
import random
from pathlib import Path
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import os


def load_image_data(
    csv_file="../data/image_data_rel.csv",
    img_size=(256, 256),
    images_per_label=20,
    validation_split=0.2,
    random_seed=42,
):
    """
    Loads image data from a CSV file with columns: ["url", "label"]
    
    Args:
        csv_file (str or Path): Path to CSV file.
        img_size (tuple): Target size (width, height) for image resizing.
        images_per_label (int): Max number of images to load per label.
        validation_split (float): Fraction of data reserved for validation.
        random_seed (int): Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_val, y_train, y_val, label_map)
    """

    df = pd.read_csv(csv_file)
    labels = sorted(df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    X = []
    y = []

    project_root = Path(csv_file).resolve().parent

    for label in labels:
        images = df[df["label"] == label]["url"].tolist()
        selected_images = random.sample(images, min(len(images), images_per_label))

        for rel_path in selected_images:
            image_path = project_root / rel_path
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0

                X.append(img_array)
                y.append(label_map[label])
            except Exception as e:
                print(f"Error loading {image_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    y = to_categorical(y, num_classes=len(label_map)) # todo if not always categorical needed -> add method parameter to select

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, stratify=y, random_state=random_seed
    )

    return X_train, X_val, y_train, y_val, label_map



def load_image_data_with_augmentation(
    csv_file="../data/image_data_rel.csv",
    img_size=(256, 256),
    images_per_label=20,
    validation_split=0.2,
    random_seed=42,
    batch_size=32,
):
    """
    Loads image data and returns augmented training data generator and validation data.

    Args:
        csv_file (str): Path to CSV file with columns ["url", "label"].
        img_size (tuple): Image resizing target.
        images_per_label (int): Max images per label.
        validation_split (float): Fraction for validation.
        random_seed (int): Random seed.
        batch_size (int): Batch size for data generators.

    Returns:
        Tuple: (train_generator, X_val, y_val, label_map)
    """

    df = pd.read_csv(csv_file)
    labels = sorted(df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    X = []
    y = []

    project_root = Path(csv_file).resolve().parent

    for label in labels:
        images = df[df["label"] == label]["url"].tolist()
        selected_images = random.sample(images, min(len(images), images_per_label))

        for rel_path in selected_images:
            image_path = project_root / rel_path
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(img_size)
                img_array = np.array(img)

                X.append(img_array)
                y.append(label_map[label])
            except Exception as e:
                print(f"Error loading {image_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    y = to_categorical(y, num_classes=len(label_map))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, stratify=y, random_state=random_seed
    )

    # Data augmentation only for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True, # TODO is this critical for leaves?
        brightness_range=(0.8, 1.2),
        fill_mode='nearest',
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )

    # For validation, return preprocessed arrays instead of generator
    X_val = val_datagen.flow(X_val, batch_size=len(X_val), shuffle=False).next()[0]

    return train_generator, X_val, y_val, label_map


def create_image_generators(X_train, y_train, X_val, y_val):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2)
    )

    val_datagen = ImageDataGenerator()  # No augmentation for validation

    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=32, shuffle=True
    )

    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=32, shuffle=False
    )

    return train_generator, val_generator




def augment_and_save_images(X, y, label_map, output_dir="augmented_images", augmentations_per_image=5):
    datagen = ImageDataGenerator(
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    os.makedirs(output_dir, exist_ok=True)
    for idx, (img_arr, label_idx) in enumerate(zip(X, y.argmax(axis=1))):
        label = list(label_map.keys())[list(label_map.values()).index(label_idx)]
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        img = array_to_img(img_arr)
        x = img_to_array(img).reshape((1,) + img_arr.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            aug_img = array_to_img(batch[0])
            aug_img.save(os.path.join(label_dir, f"{idx}_aug_{i}.jpg"))
            i += 1
            if i >= augmentations_per_image:
                break