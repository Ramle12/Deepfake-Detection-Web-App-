from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
from mtcnn import MTCNN

# ✅ Import EfficientNet (CRITICAL - matches your training!)
from efficientnet.tfkeras import EfficientNetB0

app = Flask(__name__)

# ✅ Create required directories
os.makedirs('static', exist_ok=True)
os.makedirs('model', exist_ok=True)

# ✅ Load the model (matching your training path structure)
try:
    # Try multiple possible model paths
    model_paths = [
        'model/best_model.h5',  # Local copy
        'best_model.h5',        # Root directory
        'deepfake_checkpoint/best_model.h5'  # If you copied the checkpoint folder
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = load_model(path)
            print(f"✅ Model loaded successfully from: {path}")
            break
    
    if model is None:
        print("❌ Model file not found. Please copy your trained model to one of these paths:")
        for path in model_paths:
            print(f"   - {path}")
            
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

detector = MTCNN()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # ✅ Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filepath = os.path.join('static', 'temp.jpg')
    file.save(filepath)

    try:
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_img)

        if not faces:
            pil_img = Image.fromarray(rgb_img)
            annotated_path = os.path.join('static', 'annotated.jpg')
            pil_img.save(annotated_path)

            return jsonify({
                'imagePath': annotated_path,
                'faces': [],
                'message': 'No face detected'
            }), 200

        pil_img = Image.fromarray(rgb_img)
        draw = ImageDraw.Draw(pil_img)

        predictions = []

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)

            # ✅ Extract and preprocess face (EXACTLY matching your training)
            face_img = pil_img.crop((x, y, x + w, y + h)).resize((128, 128))
            face_array = image.img_to_array(face_img) / 255.0  # Same rescale as training
            face_array = np.expand_dims(face_array, axis=0)

            # ✅ Make prediction
            pred = model.predict(face_array, verbose=0)[0][0]
            
            # ✅ Interpret prediction (based on your training class indices)
            # Your training: fake=0, real=1 (from flow_from_directory alphabetical order)
            # So pred > 0.5 means "real", pred < 0.5 means "fake" 
            is_fake = pred < 0.5  # CORRECTED: Lower values = fake
            confidence = float(1 - pred if is_fake else pred)  # Confidence in prediction
            
            color = 'red' if is_fake else 'green'

            # ✅ Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

            predictions.append({
                'bbox': [x, y, w, h],
                'isDeepfake': bool(is_fake),
                'confidence': round(confidence, 3)
            })

        annotated_path = os.path.join('static', 'annotated.jpg')
        pil_img.save(annotated_path)

        return jsonify({
            'imagePath': annotated_path,
            'faces': predictions
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)