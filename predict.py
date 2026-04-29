import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
class Config:
    model_path  = './pytorch_model/fruit_freshness.pth'
    labels_path = './pytorch_model/class_names.json'
    img_size    = 224

# ──────────────────────────────────────────
# Load model
# ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(Config.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ──────────────────────────────────────────
# Transform
# ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ──────────────────────────────────────────
# Predict single image
# ──────────────────────────────────────────
def predict(image_path, model, class_names):
    # Load and preprocess
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    # Top prediction
    top_prob, top_idx = probs.max(0)
    predicted_class   = class_names[top_idx.item()]
    confidence        = top_prob.item() * 100

    # Parse label
    is_fresh  = 'fresh'  in predicted_class.lower()
    status    = 'FRESH'  if is_fresh else 'ROTTEN'
    fruit     = predicted_class.replace('fresh', '').replace('rotten', '').strip().capitalize()
    color     = '#2ecc71' if is_fresh else '#e74c3c'

    # Top 5 predictions
    top5_probs, top5_idx = probs.topk(min(5, len(class_names)))
    top5 = [(class_names[i.item()], p.item() * 100)
            for i, p in zip(top5_idx, top5_probs)]

    return {
        'class'     : predicted_class,
        'fruit'     : fruit,
        'status'    : status,
        'confidence': confidence,
        'color'     : color,
        'top5'      : top5,
        'image'     : img
    }

# ──────────────────────────────────────────
# Display result
# ──────────────────────────────────────────
def show_result(result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Left: image ──
    ax1.imshow(result['image'])
    ax1.axis('off')
    ax1.set_title(
        f"{result['fruit']}  —  {result['status']}\n{result['confidence']:.2f}% confidence",
        fontsize=14, fontweight='bold',
        color=result['color'],
        pad=12
    )
    # Border color
    for spine in ax1.spines.values():
        spine.set_edgecolor(result['color'])
        spine.set_linewidth(3)

    # ── Right: bar chart of top 5 ──
    ax2.set_facecolor('#16213e')
    names  = [r[0] for r in result['top5']]
    values = [r[1] for r in result['top5']]
    colors = ['#2ecc71' if 'fresh' in n.lower() else '#e74c3c' for n in names]

    bars = ax2.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor='none', height=0.5)

    for bar, val in zip(bars, values[::-1]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}%', va='center', color='white', fontsize=10)

    ax2.set_xlim(0, 115)
    ax2.set_xlabel('Confidence (%)', color='white')
    ax2.set_title('Top Predictions', color='white', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['left'].set_color('#444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fresh_patch  = mpatches.Patch(color='#2ecc71', label='Fresh')
    rotten_patch = mpatches.Patch(color='#e74c3c', label='Rotten')
    ax2.legend(handles=[fresh_patch, rotten_patch], loc='lower right',
               facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────
# Predict folder of images
# ──────────────────────────────────────────
def predict_folder(folder_path, model, class_names):
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images    = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in valid_ext]

    if not images:
        print("No images found in folder.")
        return

    print(f"\nPredicting {len(images)} images in '{folder_path}'")
    print("=" * 60)
    print(f"{'Image':<30} {'Fruit':<15} {'Status':<8} {'Confidence':>10}")
    print("-" * 60)

    results = []
    for fname in images:
        path   = os.path.join(folder_path, fname)
        result = predict(path, model, class_names)
        results.append((fname, result))
        status_icon = '✅' if result['status'] == 'FRESH' else '❌'
        print(f"{fname:<30} {result['fruit']:<15} {status_icon} {result['status']:<6} {result['confidence']:>9.2f}%")

    # Summary
    fresh_count  = sum(1 for _, r in results if r['status'] == 'FRESH')
    rotten_count = len(results) - fresh_count
    print("=" * 60)
    print(f"Total  : {len(results)}  |  ✅ Fresh: {fresh_count}  |  ❌ Rotten: {rotten_count}")

    return results


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────
if __name__ == '__main__':
    # Load class names
    with open(Config.labels_path, 'r') as f:
        class_names = json.load(f)

    print("=" * 60)
    print("Fruit Freshness Predictor")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Classes     : {len(class_names)}")
    print(f"Model       : {Config.model_path}")
    print("=" * 60)

    # Load model
    model = load_model(len(class_names))
    print("✓ Model loaded successfully\n")

    # ── CHOOSE MODE ──────────────────────────────
    # MODE 1: Predict a single image
    #   → Set image_path to your image file
    # MODE 2: Predict all images in a folder
    #   → Set folder_path to your folder
    # ─────────────────────────────────────────────

    MODE = 'single'   # change to 'folder' for batch prediction

    if MODE == 'single':
        image_path = r"C:\Users\LOQ\Music\fruit_dataset\dataset\Test\rottenapples\a_r015.png"  # ← change this path

        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            print("   Update the 'image_path' variable in the script.")
        else:
            print(f"Predicting: {image_path}")
            result = predict(image_path, model, class_names)

            print("\n" + "=" * 60)
            print("Result:")
            print("=" * 60)
            print(f"  Fruit      : {result['fruit']}")
            print(f"  Status     : {result['status']}")
            print(f"  Confidence : {result['confidence']:.2f}%")
            print(f"  Class      : {result['class']}")
            print("\nTop 5 predictions:")
            for name, prob in result['top5']:
                bar = '█' * int(prob / 5)
                print(f"  {name:<25} {prob:6.2f}%  {bar}")
            print("=" * 60)

            show_result(result)

    elif MODE == 'folder':
        folder_path = r'C:\Users\LOQ\Pictures\test_fruits'   # ← change this path

        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            print("   Update the 'folder_path' variable in the script.")
        else:
            results = predict_folder(folder_path, model, class_names)
