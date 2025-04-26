import pandas as pd
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def load_data_batch(df, batch_size=1000):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        images, labels = [], []
        for _, row in batch.iterrows():
            img_path = os.path.join(Config.IMAGE_DIR, row['image_id'] + '.jpg')
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(row['dx'])
        if images:  # Only yield if batch has data
            yield np.array(images), np.array(labels)

def preprocess_and_save():
    # Load metadata
    df = pd.read_csv(Config.METADATA_PATH)
    print(f"Total rows in metadata: {len(df)}")

    # Label encoding
    le = LabelEncoder()
    labels_all = le.fit_transform(df['dx'])
    print(f"Classes: {le.classes_}")

    # Load images in batches and save
    X_all, y_all = [], []
    for images, labels in load_data_batch(df):
        X_all.append(images)
        y_all.append(le.transform(labels))
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(f"Loaded images: {X_all.shape[0]}")

    # Normalize
    X_all = X_all.astype(np.float32) / 255.0

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # Save preprocessed data
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    np.save("label_encoder.npy", le.classes_)

    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    return X_train, X_test, y_train, y_test, le

if __name__ == "__main__":
    preprocess_and_save()