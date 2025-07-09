import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def predict_symbol(image_path, model, class_names, return_probs=False):
    img = load_img(image_path, target_size=(64, 64), color_mode="rgb")
    img_array = img_to_array(img)
    img_array = tf.image.rgb_to_grayscale(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0]
    pred_class = class_names[np.argmax(pred)]
    if return_probs:
        return pred_class, pred
    return pred_class


# Function to classify all images in a folder with detailed statistics
def classify_folder(folder_path, model, class_names):
    predictions = []
    total_classifications = 0
    total_misclassifications = 0
    # Initialize dictionaries to track correct and incorrect counts per class
    correct_counts = {cls: 0 for cls in class_names}
    incorrect_counts = {cls: 0 for cls in class_names}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            try:
                predicted_class = predict_symbol(image_path, model, class_names)
                # Extract expected class from filename (e.g., 'eq_1.jpg' is 'eq', 'one_1.jpg' is 'one')
                expected_class_raw = filename.split("_")[0]
                # Map digit names to class_names (e.g., 'one' is '1'), keep operators as is
                expected_class = digit_mapping.get(
                    expected_class_raw, expected_class_raw
                )
                if expected_class in class_names:
                    total_classifications += 1
                    if predicted_class == expected_class:
                        correct_counts[expected_class] += 1
                    else:
                        incorrect_counts[expected_class] += 1
                        total_misclassifications += 1
                predictions.append((filename, predicted_class))
                print(f"Image: {filename} -> Predicted: {predicted_class}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    # Print overall statistics
    accuracy = (
        (total_classifications - total_misclassifications) / total_classifications
        if total_classifications > 0
        else 0
    )
    print(f"\nTotal Classifications: {total_classifications}")
    print(f"Total Misclassifications: {total_misclassifications}")
    print(
        f"Overall Accuracy: {(total_classifications - total_misclassifications)}/{total_classifications} ({accuracy:.4f})"
    )

    # Print per-label statistics
    print("\nPer-Label Statistics:")
    print(f"{'Label':<10} {'Correct':<10} {'Incorrect':<10} {'Accuracy':<10}")
    print("-" * 40)
    for cls in class_names:
        total_cls = correct_counts[cls] + incorrect_counts[cls]
        cls_accuracy = correct_counts[cls] / total_cls if total_cls > 0 else 0
        print(
            f"{cls:<10} {correct_counts[cls]:<10} {incorrect_counts[cls]:<10} {cls_accuracy:<10.4f}"
        )

    # Save predictions to output_file
    output_file = "my_handwritten_symbols4.txt"
    with open(output_file, "w") as f:
        for filename, pred in predictions:
            f.write(f"Image: {filename}, Predicted: {pred}\n")

    # Save statistics to output_file
    with open(output_file, "a") as f:
        f.write(f"\nTotal Classifications: {total_classifications}\n")
        f.write(f"Total Misclassifications: {total_misclassifications}\n")
        f.write(
            f"Overall Accuracy: {(total_classifications - total_misclassifications)}/{total_classifications} ({accuracy:.4f})\n"
        )
        f.write("\nPer-Label Statistics:\n")
        f.write(f"{'Label':<10} {'Correct':<10} {'Incorrect':<10} {'Accuracy':<10}\n")
        f.write("-" * 40 + "\n")
        for cls in class_names:
            total_cls = correct_counts[cls] + incorrect_counts[cls]
            cls_accuracy = correct_counts[cls] / total_cls if total_cls > 0 else 0
            f.write(
                f"{cls:<10} {correct_counts[cls]:<10} {incorrect_counts[cls]:<10} {cls_accuracy:<10.4f}\n"
            )

    print(f"\nPredictions saved to {output_file}")
    return predictions


# Function to classify a single image with probability output
def classify_single_image(image_path, model, class_names):
    try:
        predicted_class, probs = predict_symbol(
            image_path, model, class_names, return_probs=True
        )
        print(f"Image: {os.path.basename(image_path)}")
        print("Probabilities:")
        for idx, (prob, cls) in enumerate(zip(probs, class_names)):
            print(f"  Index {idx}: {cls} ({prob:.4f})")
        print(f"Predicted: {predicted_class}")
        return predicted_class
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


# Load the model
model_file = "mathmate_basic_arithmetic_model8.keras"
model = tf.keras.models.load_model(model_file)
print(f"{model_file} loaded successfully.")

class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "add",
    "div",
    "eq",
    "mul",
    "sub",
]
print("Using default class names (matches training output):", class_names)

handwritten_folder = "my_handwritten_symbols4"

# Mapping for digit filenames to class_names
digit_mapping = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

print(f"\nClassifying all images in {handwritten_folder}:")
predictions = classify_folder(handwritten_folder, model, class_names)

# single_pred = classify_single_image("mom_handwritten_symbols/div_67.jpg", model, class_names)
