import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from config import Config

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),  # Increased units
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load preprocessed data
    X_train = np.load("X_train.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_test = np.load("y_test.npy")
    label_encoder_classes = np.load("label_encoder.npy", allow_pickle=True)
    num_classes = len(label_encoder_classes)

    # Build and train model
    model = build_model(num_classes)
    model.summary()  # Model architecture dekho

    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            Config.MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=20,  # Increased epochs
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    # Save final model
    model.save(Config.MODEL_PATH)
    print("Model saved at:", Config.MODEL_PATH)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")