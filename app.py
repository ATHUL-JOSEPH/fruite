"""
Fruit Freshness Prediction — Flask API
Serves the PyTorch MobileNetV3 model through a web interface.
"""

import os
import io
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pytorch_model', 'fruit_freshness.pth')
LABELS_PATH = os.path.join(os.path.dirname(__file__), 'pytorch_model', 'class_names.json')
IMG_SIZE = 224

# ──────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────
app = Flask(__name__, static_folder='static')
CORS(app)

# ──────────────────────────────────────────
# Load model & labels once at startup
# ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(LABELS_PATH, 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)


def load_model():
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model


model = load_model()

print("=" * 60)
print("Fruit Freshness Prediction Server")
print("=" * 60)
print(f"  Device   : {device}")
print(f"  Classes  : {num_classes}")
print(f"  Model    : {MODEL_PATH}")
print("=" * 60)

# ──────────────────────────────────────────
# Image transform (same as predict.py)
# ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ──────────────────────────────────────────
# Prediction helper
# ──────────────────────────────────────────
def predict_image(img: Image.Image) -> dict:
    """Run prediction on a PIL Image and return results dict."""
    img_rgb = img.convert('RGB')
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_prob, top_idx = probs.max(0)
    predicted_class = class_names[top_idx.item()]
    confidence = top_prob.item() * 100

    is_fresh = 'fresh' in predicted_class.lower()
    status = 'FRESH' if is_fresh else 'ROTTEN'
    fruit = predicted_class.replace('fresh', '').replace('rotten', '').strip().capitalize()
    color = '#2ecc71' if is_fresh else '#e74c3c'

    # Top 5
    top5_probs, top5_idx = probs.topk(min(5, len(class_names)))
    top5 = [
        {
            'name': class_names[i.item()],
            'probability': round(p.item() * 100, 2),
            'is_fresh': 'fresh' in class_names[i.item()].lower()
        }
        for i, p in zip(top5_idx, top5_probs)
    ]

    return {
        'class': predicted_class,
        'fruit': fruit,
        'status': status,
        'confidence': round(confidence, 2),
        'color': color,
        'top5': top5
    }


# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        result = predict_image(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400

    results = []
    for file in files:
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
            result = predict_image(img)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})

    return jsonify(results)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': str(device), 'classes': num_classes})


# ──────────────────────────────────────────
# Run
# ──────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
