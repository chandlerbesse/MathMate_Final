import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import (ImageDataGenerator,
                                                  img_to_array, load_img)
from tensorflow.keras.utils import image_dataset_from_directory


# Preprocess image function
def preprocess_image(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image, label


# Dataset path
dataset_path = "basic_arithmetic"

# Check class balance
class_counts = {}
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

print("\nClass distribution:")
total_images = sum(class_counts.values())
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images ({count / total_images * 100:.2f}%)")

# Load images
dataset = image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",
    image_size=(64, 64),
    batch_size=32,
    shuffle=True,
)

# Get class names
class_names = dataset.class_names
print("Labels (classes):\n", class_names)

# Calculate dataset size
dataset_size = tf.data.experimental.cardinality(dataset).numpy()
print("Total batches:", dataset_size)
total_images = dataset_size * 32
print("Total images (approx.):", total_images)

# Define split ratios
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Calculate number of images for each split
train_size = int(train_ratio * total_images)
val_size = int(val_ratio * total_images)
test_size = int(test_ratio * total_images)

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# Calculate number of batches for each split
train_batches = train_size // 32
val_batches = val_size // 32
test_batches = test_size // 32

print(
    f"Train batches: {train_batches}, Val batches: {val_batches}, Test batches: {test_batches}"
)

# Split the dataset
train_dataset = dataset.take(train_batches)
remaining_dataset = dataset.skip(train_batches)
val_dataset = remaining_dataset.take(val_batches)
test_dataset = remaining_dataset.skip(val_batches)

# Apply preprocessing to all splits
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Optimize datasets for performance
train_dataset = train_dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)
# cache() stores dataset in memory after 1st pass to speed up subsequent epochs
# shuffle(1000) randomly shuffles training dataset with buffer size of 1000 to ensure varied batch order during training
# prefatch(tf.data.AUTOTUNE) prepares next batch while the current batch is being processed

# Verify split sizes
print(
    "Train dataset size (batches):",
    tf.data.experimental.cardinality(train_dataset).numpy(),
)
print(
    "Validation dataset size (batches):",
    tf.data.experimental.cardinality(val_dataset).numpy(),
)
print(
    "Test dataset size (batches):",
    tf.data.experimental.cardinality(test_dataset).numpy(),
)

# Define CNN Model
cnn_model = models.Sequential(
    [
        # Augmentation layers
        layers.RandomRotation(
            factor=0.05, input_shape=(64, 64, 1)
        ),  # originally factor=0.1
        layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1
        ),  # originally height_factor=0.1, width_factor=0.1
        layers.RandomZoom(height_factor=0.1),  # originally height_factor=0.1
        layers.RandomContrast(factor=0.1),  # originally factor=0.1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(
            128, (1, 1), activation="relu", padding="same"
        ),  # 1x1 Conv2D for fine details
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(
            0.5
        ),  # Randomly drops 50% of the units during training to prevent overfitting
        layers.Dense(len(class_names), activation="softmax"),  # 15 neurons
    ]
)

# Compile model
cnn_model.compile(
    optimizer="adam",  # Adapts learning rates for efficient gradient descent
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

cnn_model.summary()

# Define class weights to address class imbalance and prioritize certain classes
class_weights = {
    # Used classification accuracies from original model to update and fine-tune weights
    0: 1.0,  # 0: 93.75%
    1: 1.0,  # 1: 100%
    2: 1.2,  # 2: 81.25%
    3: 1.0,  # 3: 93.75%
    4: 1.5,  # 4: 81.25%    (Originally 1.2)
    5: 1.0,  # 5: 93.75%
    6: 1.0,  # 6: 100%
    7: 1.5,  # 7: 93.75%    (Originally 1.0)
    8: 1.2,  # 8: 87.5%     (Originally 1.1)
    9: 1.2,  # 9: 93.75%
    10: 2.5,  # add: 75%    (Originally 1.8)
    11: 1.5,  # div: 75%
    12: 3.0,  # eq: 37.5%   (Originally 2.0)
    13: 2.0,  # mul: 87.5%  (Originally 1.8)
    14: 1.0,  # sub: 93.75%
}

# Train model with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,  # increased from 3 for more convergence
    restore_best_weights=True,
)

# Train model with class weights
history = cnn_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stopping],
    class_weight=class_weights,  # adjusts the loss for imbalanced classes
    verbose=1,
)

# Save the model with a new name to avoid confusion
cnn_model.save("mathmate_basic_arithmetic_model8.keras")
print("Model saved to mathmate_basic_arithmetic_model8.keras")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_history_basic_arithmetic8.png")
plt.show()

# Evaluate on test set
test_loss, test_accuracy = cnn_model.evaluate(test_dataset, verbose=1)
print(f"\nTest Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Per-class accuracy and confusion matrix
y_true = []
y_pred = []
for images, labels in test_dataset:
    y_true.extend(labels.numpy())
    preds = cnn_model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

print("\nPer-class performance:")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix_basic_arithmetic8.png")
plt.show()


# Test predictions on multiple equations
def predict_symbol(image_path, model, class_names):
    img = load_img(image_path, target_size=(64, 64), color_mode="rgb")
    img_array = img_to_array(img)
    img_array = tf.image.rgb_to_grayscale(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)
    pred_class = class_names[np.argmax(pred)]
    return pred_class


def parse_equation(image_paths, model, class_names):
    equation = ""
    for path in image_paths:
        pred = predict_symbol(path, model, class_names)
        if pred in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            equation += pred
        elif pred == "add":
            equation += "+"
        elif pred == "sub":
            equation += "-"
        elif pred == "mul":
            equation += "*"
        elif pred == "div":
            equation += "/"
        elif pred == "eq":
            equation += "="
        else:
            equation += f"[{pred}]"
    return equation


# Sample equations
sample_paths = [
    # 2 + 3 =
    [
        "basic_arithmetic/2/9MUvhAZ0.png",
        "basic_arithmetic/add/140.jpg",
        "basic_arithmetic/3/3295.jpg",
        "basic_arithmetic/eq/204.jpg",
    ],
    # 7 - 4 =
    [
        "basic_arithmetic/7/6zDWtZhZ.png",
        "basic_arithmetic/sub/137.jpg",
        "basic_arithmetic/4/18ZD4Xnr.png",
        "basic_arithmetic/eq/201.jpg",
    ],
    # 6 * 5 =
    [
        "basic_arithmetic/6/3FrLVGOK.png",
        "basic_arithmetic/mul/195.jpg",
        "basic_arithmetic/5/5lxHEBgG.png",
        "basic_arithmetic/eq/228.jpg",
    ],
]

print("\nEquation predictions:")
for i, eq_paths in enumerate(sample_paths, 1):
    try:
        parsed_eq = parse_equation(eq_paths, cnn_model, class_names)
        print(f"Equation {i}: {parsed_eq}")
    except FileNotFoundError as e:
        print(f"Equation {i}: File not found ({e.filename}), skipping")
