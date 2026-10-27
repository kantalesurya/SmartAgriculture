import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

DATASET_DIR = "/Users/kantalesurya/Downloads/plantvillage dataset/Tomato"

image_paths = []
labels = []

# Class 0: Tomato___Blight (from Early_blight and Late_blight)
# Class 1: Tomato___Healthy (from healthy)
class_mapping = {
    "Tomato___Early_blight": 0,
    "Tomato___Late_blight": 0,
    "Tomato___healthy": 1
}

for folder, label in class_mapping.items():
    folder_path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"Warning: {folder_path} does not exist!")
        continue
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images in {folder} -> Label {label}")
    for f in files:
        image_paths.append(os.path.join(folder_path, f))
        labels.append(label)

print(f"Total samples: {len(image_paths)}")
print(f"Blight count (0): {labels.count(0)}")
print(f"Healthy count (1): {labels.count(1)}")

# Balance dataset or stratify split
train_paths, test_paths, train_y, test_y = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")

# Load base MobileNetV2
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)
base.trainable = False

def extract_features(paths, batch_size=64):
    features = []
    current_batch = []
    total = len(paths)
    for idx, p in enumerate(paths):
        try:
            img = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            current_batch.append(arr)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue
        
        if len(current_batch) == batch_size or idx == total - 1:
            batch_arr = preprocess_input(np.array(current_batch, dtype=np.float32))
            feats = base(batch_arr, training=False).numpy()
            features.append(feats)
            current_batch = []
            if (idx + 1) % 500 == 0 or idx == total - 1:
                print(f"Processed {idx + 1}/{total} images...")
    return np.vstack(features)

print("\nExtracting train features...")
X_train = extract_features(train_paths)
y_train = np.array(train_y[:len(X_train)])

print("\nExtracting test features...")
X_test = extract_features(test_paths)
y_test = np.array(test_y[:len(X_test)])

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Compute class weights to handle slight imbalance (Blight: ~2908, Healthy: ~1591)
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: float(w) for i, w in enumerate(weights)}
print("Class weights:", class_weight_dict)

# Build classification head
head_input = tf.keras.layers.Input(shape=(1280,))
x = tf.keras.layers.Dropout(0.3)(head_input)
out = tf.keras.layers.Dense(2, activation="softmax", name="dense")(x)
head_model = tf.keras.Model(inputs=head_input, outputs=out)

head_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

head_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate on test set
preds = np.argmax(head_model.predict(X_test), axis=1)
print("\n=== Classification Report on Test Set ===")
print(classification_report(y_test, preds, target_names=["Tomato___Blight", "Tomato___Healthy"], digits=4))

# Now build the end-to-end full model matching the exact architecture of disease_model.keras
base_full = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_full.trainable = False

full_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    base_full,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation="softmax", name="dense")
])

# Transfer trained weights from head_model
dense_layer_head = head_model.get_layer("dense")
dense_layer_full = full_model.get_layer("dense")
dense_layer_full.set_weights(dense_layer_head.get_weights())

# Save model
output_path = "/Users/kantalesurya/SmartAgriculture/disease_model.keras"
full_model.save(output_path)
print(f"\nTrained full model successfully saved to {output_path}!")
