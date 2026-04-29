import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ──────────────────────────────────────────
# GPU Check
# ──────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("PyTorch Setup Check")
print("=" * 60)
print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {device}")
if device.type == 'cuda':
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)


# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
class Config:
    train_path = r'C:\Users\LOQ\Music\fruit_dataset\dataset\Train'
    test_path  = r'C:\Users\LOQ\Music\fruit_dataset\dataset\Test'

    batch_size    = 32
    img_size      = 224
    epochs        = 30
    learning_rate = 0.001
    num_workers   = 4      # 4 parallel workers — much faster data loading

    save_dir    = './pytorch_model'
    model_path  = './pytorch_model/fruit_freshness.pth'
    labels_path = './pytorch_model/class_names.json'

os.makedirs(Config.save_dir, exist_ok=True)


# ──────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ──────────────────────────────────────────
# Helper functions — defined at module level
# so Windows multiprocessing can pickle them
# ──────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


# ──────────────────────────────────────────
# MAIN — required guard for Windows workers
# ──────────────────────────────────────────
if __name__ == '__main__':

    # ── Datasets ──
    print("\nLoading datasets...")
    print("-" * 60)

    train_dataset = datasets.ImageFolder(Config.train_path, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(Config.test_path,  transform=test_transforms)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Fix class mismatch — remap test to train indices
    test_dataset.class_to_idx = train_dataset.class_to_idx
    test_dataset.samples = [
        (path, train_dataset.class_to_idx[test_dataset.classes[label]])
        for path, label in test_dataset.samples
        if test_dataset.classes[label] in train_dataset.class_to_idx
    ]
    test_dataset.targets = [s[1] for s in test_dataset.samples]

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=True,  # keeps workers alive between epochs
        prefetch_factor=2         # loads next batch while GPU trains
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print(f"✓ {num_classes} classes found:")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}. {name}")
    print(f"\nTraining samples : {len(train_dataset)}")
    print(f"Test samples     : {len(test_dataset)}")
    print("-" * 60)

    with open(Config.labels_path, 'w') as f:
        json.dump(class_names, f)

    # ── Model ──
    print("\nBuilding model...")
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # cuDNN auto-tuner — finds fastest conv algorithm after first batch
    torch.backends.cudnn.benchmark = True

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable_params:,}")
    print("-" * 60)

    # ── Loss / Optimizer / Scheduler ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )

    # ── Training Loop ──
    print("\n" + "=" * 60)
    print("Training on RTX 3050 🚀")
    print("=" * 60)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc     = 0.0
    patience_counter = 0
    PATIENCE         = 7

    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}  |  LR: {optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.model_path)
            print(f"  ✓ Best model saved  (val_acc={best_val_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("\n  Early stopping triggered.")
                break

    print(f"\n✓ Training complete!  Best Val Accuracy: {best_val_acc*100:.2f}%")

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_acc'], label='Train', lw=2)
    ax1.plot(history['val_acc'],   label='Val',   lw=2)
    ax1.set_title('Accuracy'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['train_loss'], label='Train', lw=2)
    ax2.plot(history['val_loss'],   label='Val',   lw=2)
    ax2.set_title('Loss'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'training_history.png'), dpi=150)
    plt.show()

    # ── Final Evaluation ──
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(torch.load(Config.model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Testing"):
            outputs = model(images.to(device, non_blocking=True))
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    present_indices = sorted(set(all_labels))
    present_names   = [class_names[i] for i in present_indices]

    test_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy : {test_acc*100:.2f}%")
    print("\n" + "=" * 60)
    print(classification_report(all_labels, all_preds, target_names=present_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_names, yticklabels=present_names,
                annot_kws={'size': 9})
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix — Fruit Freshness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_dir, 'confusion_matrix.png'), dpi=150)
    plt.show()

    print("\nPer-class Accuracy:")
    for i, name in zip(present_indices, present_names):
        mask = all_labels == i
        acc  = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0.0
        print(f"  {name:28s}: {acc*100:6.2f}%")

    fresh_idx  = [i for i, n in enumerate(class_names) if 'fresh'  in n.lower()]
    rotten_idx = [i for i, n in enumerate(class_names) if 'rotten' in n.lower()]

    def group_acc(indices):
        accs = [(all_preds[all_labels == i] == i).mean()
                for i in indices if (all_labels == i).sum() > 0]
        return np.mean(accs) if accs else 0.0

    print(f"\nFresh  avg accuracy : {group_acc(fresh_idx)*100:.2f}%")
    print(f"Rotten avg accuracy : {group_acc(rotten_idx)*100:.2f}%")

    print("\n" + "=" * 60)
    print("🎉  All done!")
    print(f"  Model  → {Config.model_path}")
    print(f"  Labels → {Config.labels_path}")
    print("=" * 60)


################### USING TENSORFLOW 2.13.0 + DIRECTML BACKEND (GPU-ACCELERATED ON WINDOWS) ###################

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # Fix DirectML + Intel TF conflict — must be set BEFORE importing TF
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Conv2D, MaxPooling2D, Flatten, Dense,
#     Dropout, BatchNormalization, GlobalAveragePooling2D
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2   # ← lighter than ResNet50, works better on DirectML
# from sklearn.metrics import classification_report, confusion_matrix

# tf.get_logger().setLevel('ERROR')

# # ──────────────────────────────────────────
# print("=" * 60)
# print("TensorFlow Setup Check")
# print("=" * 60)
# print(f"TensorFlow version : {tf.__version__}")

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"✓ GPU Found: {len(gpus)} GPU(s)")
#     for gpu in gpus:
#         print(f"  - {gpu.name}")
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("✓ Memory growth enabled")
#     except RuntimeError as e:
#         print(f"  Memory growth note: {e}")
# else:
#     print("⚠️  No GPU found — running on CPU (slower but fine)")
# print("=" * 60)


# # ──────────────────────────────────────────
# class Config:
#     # ↓ UPDATE IF YOUR PATHS DIFFER
#     train_path = r'C:\Users\LOQ\Music\fruit_dataset\dataset\Train'
#     test_path  = r'C:\Users\LOQ\Music\fruit_dataset\dataset\Test'

#     batch_size    = 16          # smaller batch → safer for DirectML / CPU
#     img_size      = (224, 224)
#     epochs        = 30
#     learning_rate = 0.001
#     num_workers   = 0           # 0 = no multiprocessing (avoids Windows worker crash)

#     save_dir         = './tensorflow_model'
#     model_path       = './tensorflow_model/fruit_freshness_model.h5'
#     class_names_path = './tensorflow_model/class_names.json'


# os.makedirs(Config.save_dir, exist_ok=True)


# # ──────────────────────────────────────────
# # Data generators
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.0   # no internal split — we have a separate Test folder
# )

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# print("\nLoading datasets…")
# print("-" * 60)

# train_generator = train_datagen.flow_from_directory(
#     Config.train_path,
#     target_size=Config.img_size,
#     batch_size=Config.batch_size,
#     class_mode='categorical',
#     shuffle=True
# )

# test_generator = test_datagen.flow_from_directory(
#     Config.test_path,
#     target_size=Config.img_size,
#     batch_size=Config.batch_size,
#     class_mode='categorical',
#     shuffle=False       # MUST be False for correct evaluation order
# )

# num_classes = len(train_generator.class_indices)
# class_names = list(train_generator.class_indices.keys())

# print(f"\n✓ {num_classes} classes found:")
# for i, name in enumerate(class_names):
#     print(f"  {i:2d}. {name}")
# print(f"\nTraining samples : {train_generator.samples}")
# print(f"Test samples     : {test_generator.samples}")
# print("-" * 60)


# # ──────────────────────────────────────────
# # Model — MobileNetV2 base (lighter, DirectML-friendly)
# print("\nBuilding model…")

# base_model = MobileNetV2(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(224, 224, 3)
# )
# base_model.trainable = False   # freeze base weights for initial training

# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(256, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.4),
#     Dense(128, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(
#     optimizer=Adam(learning_rate=Config.learning_rate),
#     loss=CategoricalCrossentropy(),
#     metrics=['accuracy']
# )

# model.summary()
# trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
# print(f"\nTrainable parameters: {trainable:,}")
# print("-" * 60)


# # ──────────────────────────────────────────
# # Callbacks
# callbacks = [
#     EarlyStopping(
#         monitor='val_accuracy',
#         patience=7,
#         restore_best_weights=True,
#         verbose=1
#     ),
#     ModelCheckpoint(
#         Config.model_path,
#         monitor='val_accuracy',
#         save_best_only=True,
#         mode='max',
#         verbose=1
#     ),
#     ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=3,
#         min_lr=1e-7,
#         verbose=1
#     )
# ]


# # ──────────────────────────────────────────
# # Phase 1 — Train head only
# print("\n" + "=" * 60)
# print("Phase 1: Training classification head (base frozen)")
# print("=" * 60)

# history = model.fit(
#     train_generator,
#     epochs=Config.epochs,
#     validation_data=test_generator,
#     callbacks=callbacks,
#     workers=Config.num_workers,    # 0 → avoids multiprocessing issues on Windows
#     use_multiprocessing=False,
#     verbose=1
# )

# # ── Phase 2 — Fine-tune top layers of base model ──
# print("\n" + "=" * 60)
# print("Phase 2: Fine-tuning top layers of MobileNetV2")
# print("=" * 60)

# base_model.trainable = True
# # Freeze all but the last 30 layers
# for layer in base_model.layers[:-30]:
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(learning_rate=1e-5),   # much lower LR for fine-tuning
#     loss=CategoricalCrossentropy(),
#     metrics=['accuracy']
# )

# history_fine = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=test_generator,
#     callbacks=callbacks,
#     workers=Config.num_workers,
#     use_multiprocessing=False,
#     verbose=1
# )

# print("\n✓ Training complete!")


# # ──────────────────────────────────────────
# # Save class names
# with open(Config.class_names_path, 'w') as f:
#     json.dump(class_names, f)

# # Save final model
# model.save(Config.model_path)


# # ──────────────────────────────────────────
# # Plot training history (both phases combined)
# def combine_histories(h1, h2, key):
#     return h1.history[key] + h2.history[key]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# ax1.plot(combine_histories(history, history_fine, 'accuracy'),      label='Train', lw=2)
# ax1.plot(combine_histories(history, history_fine, 'val_accuracy'),  label='Val',   lw=2)
# ax1.axvline(len(history.history['accuracy']), color='gray', linestyle='--', label='Fine-tune start')
# ax1.set_title('Accuracy', fontsize=14)
# ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
# ax1.legend(); ax1.grid(True, alpha=0.3)

# ax2.plot(combine_histories(history, history_fine, 'loss'),      label='Train', lw=2)
# ax2.plot(combine_histories(history, history_fine, 'val_loss'),  label='Val',   lw=2)
# ax2.axvline(len(history.history['loss']), color='gray', linestyle='--', label='Fine-tune start')
# ax2.set_title('Loss', fontsize=14)
# ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
# ax2.legend(); ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(os.path.join(Config.save_dir, 'training_history.png'), dpi=150)
# plt.show()


# # ──────────────────────────────────────────
# # Evaluation
# print("\n" + "=" * 60)
# print("Evaluating on test set…")
# print("=" * 60)

# test_generator.reset()   # important — resets internal index to 0
# predictions = model.predict(test_generator, verbose=1)
# y_pred = np.argmax(predictions, axis=1)
# y_true = test_generator.classes

# test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
# print(f"\nTest Loss     : {test_loss:.4f}")
# print(f"Test Accuracy : {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")

# print("\n" + "=" * 60)
# print("Classification Report:")
# print("=" * 60)
# print(classification_report(y_true, y_pred, target_names=class_names))

# # Confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(14, 12))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=class_names, yticklabels=class_names,
#             annot_kws={'size': 9})
# plt.xlabel('Predicted', fontsize=12)
# plt.ylabel('Actual', fontsize=12)
# plt.title('Confusion Matrix — Fruit Freshness Classification', fontsize=14)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig(os.path.join(Config.save_dir, 'confusion_matrix.png'), dpi=150)
# plt.show()

# # Per-class accuracy
# print("\n" + "=" * 60)
# print("Per-class Accuracy:")
# print("=" * 60)
# for i, name in enumerate(class_names):
#     row_sum = cm[i, :].sum()
#     acc = cm[i, i] / row_sum if row_sum > 0 else 0.0
#     print(f"  {name:25s}: {acc*100:6.2f}%")

# # Fresh vs Rotten summary
# fresh_idx  = [i for i, n in enumerate(class_names) if 'fresh'  in n.lower()]
# rotten_idx = [i for i, n in enumerate(class_names) if 'rotten' in n.lower()]

# def avg_acc(indices):
#     accs = []
#     for i in indices:
#         s = cm[i, :].sum()
#         accs.append(cm[i, i] / s if s > 0 else 0.0)
#     return np.mean(accs) if accs else 0.0

# print("\n" + "=" * 60)
# print("Fresh vs Rotten Summary:")
# print("=" * 60)
# print(f"  Fresh  avg accuracy : {avg_acc(fresh_idx)*100:.2f}%")
# print(f"  Rotten avg accuracy : {avg_acc(rotten_idx)*100:.2f}%")

# print("\n" + "=" * 60)
# print("🎉  All done!")
# print(f"  Model  → {Config.model_path}")
# print(f"  Labels → {Config.class_names_path}")
# print("=" * 60)