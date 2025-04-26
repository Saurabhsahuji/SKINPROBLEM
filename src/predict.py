import tensorflow as tf
import cv2
import numpy as np
from config import Config
import logging
import os

logging.basicConfig(filename=Config.LOG_PATH, level=logging.INFO)

# Mapping between old abbreviations and new full names
ABBREV_TO_FULL = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions',
    'scc': 'Squamous Cell Carcinoma'
}

# The order of classes as they were during training (abbreviations)
TRAINING_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'scc']

def load_model():
    """Load the trained model."""
    if not os.path.exists(Config.MODEL_PATH):
        logging.error("Model file not found!")
        raise FileNotFoundError("Model not found!")
    model = tf.keras.models.load_model(Config.MODEL_PATH)
    logging.info("Model loaded successfully.")
    return model

def preprocess_image(image_path):
    """Preprocess a single image for prediction."""
    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Failed to load image: {image_path}")
        raise ValueError(f"Failed to load image: {image_path}")
    
    img = cv2.resize(img, Config.IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    logging.info(f"Image preprocessed: {image_path}")
    return img

def predict_disease(image_path, model, label_map):
    """Predict disease and suggest solution with top-2 predictions."""
    try:
        img = preprocess_image(image_path)
        pred = model.predict(img)
        
        # Get top-2 predictions
        top_indices = np.argsort(pred[0])[-2:][::-1]
        top_confidences = pred[0][top_indices] * 100
        # Map indices to training classes (abbreviations), then to full names
        top_diseases = [ABBREV_TO_FULL[TRAINING_CLASSES[idx]] for idx in top_indices]
        
        disease = top_diseases[0]
        confidence = top_confidences[0]
        
        solution = Config.SOLUTIONS.get(disease, "Consult a doctor for further evaluation.")
        top_predictions = [
            f"{top_diseases[0]}: {top_confidences[0]:.2f}%",
            f"{top_diseases[1]}: {top_confidences[1]:.2f}%"
        ]
        
        logging.info(f"Prediction: {disease} with {confidence:.2f}% confidence. Top-2: {top_predictions}")
        return disease, solution, confidence, top_predictions
    except Exception as e:
        logging.error(f"Prediction failed for {image_path}: {str(e)}")
        # Return a default result in case of error
        return "Prediction Failed", f"Error: {str(e)}", 0.0, ["N/A: 0.00%", "N/A: 0.00%"]

if __name__ == "__main__":
    model = load_model()
    label_map = {v: k for k, v in enumerate(TRAINING_CLASSES)}
    disease, solution, confidence, top_predictions = predict_disease("test_image.jpg", model, label_map)
    print(f"Disease: {disease}\nSolution: {solution}\nConfidence: {confidence:.2f}%\nTop Predictions: {top_predictions}")