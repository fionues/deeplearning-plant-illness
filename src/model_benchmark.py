
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB3,
    InceptionV3, MobileNetV2,
    DenseNet121, Xception
)
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten,
    Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from PIL import Image
from pathlib import Path
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


def load_sample_data(csv_path, img_size=(299, 299), images_per_label=20, validation_split=0.2):
    df = pd.read_csv(csv_path)
    labels = sorted(df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}

    X, y = [], []
    base_path = Path(csv_path).resolve().parent

    for label in labels:
        images = df[df["label"] == label]["url"].tolist()
        selected = random.sample(images, min(len(images), images_per_label))
        for rel_path in selected:
            path = base_path / rel_path
            try:
                img = Image.open(path).convert("RGB").resize(img_size)
                X.append(np.array(img) / 255.0)
                y.append(label_map[label])
            except:
                continue

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(label_map))

    return train_test_split(X, y, test_size=validation_split, stratify=y), label_map


def create_image_generator(X_train, y_train):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2)
    )

    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=32, shuffle=True
    )

    return train_generator


def head_simple(base, num_classes):
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

def head_dense_dropout(base, num_classes):
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

def head_batchnorm_dropout(base, num_classes):
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

def head_conv(base, num_classes):
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(base.output)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=output)

def default_normalization_preprocessing(X):
    X_processed = []
    for x in X:
        X_processed.append(x / 255.0)
    return X_processed

def efficient_net_preproprecssing(X):
    X_processed = []
    for x in X:
        X_processed.append(effnet_preprocess(np.array(x)))
    return X_processed

backbones = {
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB3': EfficientNetB3,
    'InceptionV3': InceptionV3,
    'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121,
    'Xception': Xception
}

img_size_map = {
    'EfficientNetB0': (224, 224),
    'EfficientNetB3': (300, 300),
    'InceptionV3': (299, 299),
    'MobileNetV2': (224, 224),
    'DenseNet121': (224, 224),
    'Xception': (299, 299)
}

heads_map = {
    'EfficientNetB0': ['simple', 'dense_dropout', 'batchnorm_dropout'],
    'EfficientNetB3': ['simple', 'dense_dropout', 'batchnorm_dropout'],
    'InceptionV3': ['simple', 'dense_dropout'],
    'MobileNetV2': ['simple', 'dense_dropout'],
    'DenseNet121': ['simple', 'dense_dropout'],
    'Xception': ['simple', 'dense_dropout'],
}

preprocessing_map = {
    'EfficientNetB0': efficient_net_preproprecssing,
    'EfficientNetB3': efficient_net_preproprecssing,
    'InceptionV3': default_normalization_preprocessing,
    'MobileNetV2': default_normalization_preprocessing,
    'DenseNet121': default_normalization_preprocessing,
    'Xception': default_normalization_preprocessing,
}

heads = {
    'simple': head_simple,
    'dense_dropout': head_dense_dropout,
    'batchnorm_dropout': head_batchnorm_dropout,
    'conv': head_conv
}


def evaluate_models(csv_path, img_size=(299, 299), epochs=3, images_per_label=20):
    (X_train, X_val, y_train, y_val), label_map = load_sample_data(csv_path, img_size=img_size, images_per_label=images_per_label)
    num_classes = len(label_map)
    results = []

    for bname, bmodel in backbones.items():
        base = bmodel(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        base.trainable = False

        for hname, head_fn in heads.items():
            model = head_fn(base, num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"Training {bname} with head {hname}...")
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16, verbose=0)
            print(f"Evaluating {bname} with head {hname}...")

            y_pred = model.predict(X_val)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_val, axis=1)

            acc = accuracy_score(y_true_labels, y_pred_labels)
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

            results.append({
                "backbone": bname,
                "head": hname,
                "accuracy": acc,
                "f1_score": f1,
                # "start_loss": history.history['loss'][0],
                # "end_loss": history.history['loss'][-1],
                # "start_val_loss": history.history['val_loss'][0],
                "end_val_loss": history.history['val_loss'][-1],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss'],
                "y_true": y_true_labels,
                "y_pred": y_pred_labels,
                "label_map": label_map
            })

    return pd.DataFrame(results)


###################

def evaluate_model(X_train, X_val, y_train, y_val, label_map, bname, epochs=20, doAutostop=False, heads_map=heads_map):
    num_classes = len(label_map)

    if bname not in backbones:
        raise ValueError(f"Backbone {bname} not found in predefined backbones.")
    
        # Configure early stopping conditionally
    callbacks = []
    if doAutostop:
        early_stopping = EarlyStopping(
            monitor='val_loss',        # Metric to monitor
            patience=3,                # Epochs to wait after no improvement
            restore_best_weights=True # Revert to the best weights
        )
        callbacks.append(early_stopping)
    
    results = []
    bmodel = backbones[bname]

    base = bmodel(weights='imagenet', include_top=False, input_shape=img_size_map[bname] + (3,))
    base.trainable = False

    selected_heads = heads_map[bname]
    print(selected_heads)

    for head in selected_heads:
        hname = head
        print(head)
        head_fn = heads[head]
        print(head_fn)
    # for hname, head_fn in heads.items():
        model = head_fn(base, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Training {bname} with head {hname}...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16, verbose=0, callbacks=callbacks)
        print(f"Evaluating {bname} with head {hname}...")

        y_pred = model.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_val, axis=1)

        acc = accuracy_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

        results.append({
            "backbone": bname,
            "head": hname,
            "accuracy": acc,
            "f1_score": f1,
                        "end_val_loss": history.history['val_loss'][-1],
            "history": history.history,
            # "start_loss": history.history['loss'][0],
            # "end_loss": history.history['loss'][-1],
            # "start_val_loss": history.history['val_loss'][0],
            # "loss": history.history['loss'],
            # "val_loss": history.history['val_loss'],
            "y_true": y_true_labels,
            "y_pred": y_pred_labels
        })

    return pd.DataFrame(results)


def evaluate_initial_loss(X_val, y_val, label_map, bname, heads_map=heads_map):
    num_classes = len(label_map)

    if bname not in backbones:
        raise ValueError(f"Backbone {bname} not found in predefined backbones.")
    
    results = []
    bmodel = backbones[bname]

    base = bmodel(weights='imagenet', include_top=False, input_shape=img_size_map[bname] + (3,))
    base.trainable = False

    selected_heads = heads_map[bname]
    print(selected_heads)

    for head in selected_heads:
        hname = head
        print(head)
        head_fn = heads[head]
        print(head_fn)

        model = head_fn(base, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Capture initial (untrained) performance
        initial_eval = model.evaluate(X_val, y_val, verbose=0)
        initial_loss = initial_eval[0]
        initial_accuracy = initial_eval[1]

        results.append({
            "backbone": bname,
            "head": hname,
            "initial_val_loss": initial_loss,
            "initial_val_accuracy": initial_accuracy,
        })

    return pd.DataFrame(results)

#####################

best_backbones = {
    # 'EfficientNetB0': EfficientNetB0,
    # 'EfficientNetB3': EfficientNetB3,
    # 'InceptionV3': InceptionV3,
    'MobileNetV2': MobileNetV2,
    # 'DenseNet121': DenseNet121,
    # 'Xception': Xception
}

best_heads = {
    # 'simple': head_simple,
    'dense_dropout': head_dense_dropout,
    # 'batchnorm_dropout': head_batchnorm_dropout,
    # 'conv': head_conv
}

def finalize_models(csv_path, img_size=(299, 299), epochs=3, doAutostop=True, backbones=best_backbones, heads=best_heads):
    (X_train, X_val, y_train, y_val), label_map = load_sample_data(csv_path, img_size=img_size)
    train_generator = create_image_generator(X_train, y_train)
    num_classes = len(label_map)
    results = []

    # Configure early stopping conditionally
    callbacks = []
    if doAutostop:
        early_stopping = EarlyStopping(
            monitor='val_loss',        # Metric to monitor
            patience=3,                # Epochs to wait after no improvement
            restore_best_weights=True # Revert to the best weights
        )
        callbacks.append(early_stopping)

    for bname, bmodel in backbones.items():
        base = bmodel(weights='imagenet', include_top=False, input_shape=img_size + (3,))
        base.trainable = False

        for hname, head_fn in heads.items():
            model = head_fn(base, num_classes)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"Training {bname} with head {hname}...")
            history = model.fit(train_generator, validation_data=(X_val, y_val), epochs=epochs, batch_size=16, verbose=0, callbacks=callbacks)
            print(f"Evaluating {bname} with head {hname}...")

            y_pred = model.predict(X_val)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(y_val, axis=1)

            acc = accuracy_score(y_true_labels, y_pred_labels)
            f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

            results.append({
                "backbone": bname,
                "head": hname,
                "accuracy": acc,
                "f1_score": f1,
                # "start_loss": history.history['loss'][0],
                # "end_loss": history.history['loss'][-1],
                # "start_val_loss": history.history['val_loss'][0],
                "end_val_loss": history.history['val_loss'][-1],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss'],
                "y_true": y_true_labels,
                "y_pred": y_pred_labels,
                "label_map": label_map
            })

    return pd.DataFrame(results)