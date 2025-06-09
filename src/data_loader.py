import pandas as pd
import numpy as np
import random
from pathlib import Path
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


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