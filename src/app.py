from flask import Flask, request, render_template
from predict import load_model, predict_disease, TRAINING_CLASSES
from config import Config
import os
import logging
import uuid
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model()
label_map = {v: k for k, v in enumerate(TRAINING_CLASSES)}

logging.basicConfig(filename=Config.LOG_PATH, level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            logging.error("No file uploaded.")
            return render_template('index.html', error="No file uploaded!")
        
        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected.")
            return render_template('index.html', error="No file selected!")
        
        # Sanitize the filename
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()
        safe_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Ensure the upload directory exists (relative to app.py)
        upload_dir = os.path.join(os.path.dirname(__file__), 'static/uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Save the file and log the path
        try:
            file.save(file_path)
            logging.info(f"Image saved to: {file_path}")
            logging.info(f"Absolute path: {os.path.abspath(file_path)}")
            if not os.path.exists(file_path):
                logging.error(f"Image not found after saving: {file_path}")
                return render_template('index.html', error="Failed to save image: File not found after saving.")
            else:
                file_size = os.path.getsize(file_path)
                logging.info(f"Image file size: {file_size} bytes")
                if file_size == 0:
                    logging.error(f"Image file is empty: {file_path}")
                    return render_template('index.html', error="Failed to save image: File is empty.")
        except Exception as e:
            logging.error(f"Failed to save image: {str(e)}")
            return render_template('index.html', error=f"Failed to save image: {str(e)}")
        
        # Set the image URL for the template
        img_url = f"uploads/{safe_filename}"
        logging.info(f"Image URL set to: {img_url}")
        logging.info(f"Resolved static URL: /static/{img_url}")
        
        # Encode the image as base64 for fallback
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        base64_url = f"data:image/{file_extension[1:]};base64,{base64_image}"
        
        # Get prediction result
        try:
            disease, solution, confidence, top_predictions = predict_disease(file_path, model, label_map)
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            disease = "Prediction Failed"
            solution = f"Error: {str(e)}"
            confidence = 0.0
            top_predictions = ["N/A: 0.00%", "N/A: 0.00%"]
        
        # Show the result for all predictions
        result = {
            'problem': disease,
            'solution': solution,
            'confidence': f"Confidence: {confidence:.2f}%",
            'top_predictions': top_predictions
        }
        
        return render_template('result.html', result=result, img_path=img_url, base64_url=base64_url)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)