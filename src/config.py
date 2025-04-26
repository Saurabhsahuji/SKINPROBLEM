# src/config.py
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "../dataset")
    METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
    IMG_SIZE = (224, 224)
    IMAGE_DIR = os.path.join(DATASET_DIR, "images")
    MODEL_PATH = os.path.join(BASE_DIR, "../models/skin_disease_model.h5")
    LOG_PATH = os.path.join(BASE_DIR, "../logs/app.log")
    
    DISEASE_CLASSES = [
        'Actinic Keratosis / Intraepithelial Carcinoma',
        'Basal Cell Carcinoma',
        'Benign Keratosis-like Lesions',
        'Dermatofibroma',
        'Melanoma',
        'Melanocytic Nevi',
        'Vascular Lesions',
        'Squamous Cell Carcinoma'  # Added to match SOLUTIONS
    ]

    SOLUTIONS = {
        'Benign Keratosis-like Lesions': 'Usually benign, monitor for changes.',
        'Melanocytic Nevi': 'Benign mole, no treatment needed.',
        'Actinic Keratosis / Intraepithelial Carcinoma': 'Use sunscreen. Treatment: Cryotherapy.',
        'Melanoma': 'Consult a dermatologist. Surgery possible.',
        'Basal Cell Carcinoma': 'See a specialist. Mohs surgery recommended.',
        'Squamous Cell Carcinoma': 'Doctor visit needed. Surgical removal.',
        'Dermatofibroma': 'Benign, optional removal if symptomatic.',
        'Vascular Lesions': 'Consult a doctor for evaluation.'  # Added since it was missing
    }