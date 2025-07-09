import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import matplotlib.pyplot as plt


def predict_symbol(image_path, model, class_names, return_probs=False):
    img = load_img(image_path, target_size=(64, 64), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = tf.image.rgb_to_grayscale(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array, verbose=0)[0]
    pred_class = class_names[np.argmax(pred)]
    if return_probs:
        return pred_class, pred
    return pred_class


model_file = "mathmate_basic_arithmetic_model8.keras"

model = tf.keras.models.load_model(model_file)
print(f"{model_file} loaded successfully.")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'eq', 'mul', 'sub']
print("Class names:", class_names)

handwritten_folder = "my_handwritten_symbols4"

# Mapping for digit filenames to class_names
digit_mapping = {
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9'
}

# Mapping for operator classes to symbols
operator_mapping = {
    'add': '+',
    'div': '/',
    'eq': '=',
    'mul': '*',
    'sub': '-'
}


# Group images by class
def group_images_by_class(folder_path):
    images_by_class = {cls: [] for cls in class_names}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            class_raw = filename.split('_')[0]
            class_name = digit_mapping.get(class_raw, class_raw)
            if class_name in class_names:
                images_by_class[class_name].append(filename)
    return images_by_class


# Display a single image
def display_image(image_path, title):
    img = Image.open(image_path)
    plt.figure(figsize=(4, 5))
    plt.imshow(img, cmap='gray' if img.mode != 'RGB' else None)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Concatenate images horizontally
def concatenate_images(image_paths):
    images = [Image.open(path).convert('RGB') for path in image_paths]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img


# Display concatenated images forming the equation
def display_equation_images(image_paths, title):
    full_equation_img = concatenate_images(image_paths)
    plt.figure(figsize=(len(image_paths) * 2, 3))
    plt.imshow(full_equation_img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Individual symbol classification
def classify_single_symbol(folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    all_images = []
    for cls in class_names:
        for img in images_by_class[cls]:
            all_images.append((cls, img))
    if not all_images:
        return None, None, None, None

    expected_class, image_file = random.choice(all_images)
    image_path = os.path.join(folder_path, image_file)
    predicted_class = predict_symbol(image_path, model, class_names)
    is_correct = (expected_class == predicted_class)

    print(f"\nIndividual Symbol Classification:")
    print(f"Image: {image_file}")
    print(f"Expected classification: {expected_class}")
    print(f"Predicted classification: {predicted_class}")
    print(f"{'Correct!' if is_correct else 'Incorrect...'}")
    display_image(image_path, f"Image: {image_file}\nExpected: {expected_class}, Predicted: {predicted_class}"
                              f"\n{'Correct!' if is_correct else 'Incorrect...'}")

    return image_file, expected_class, predicted_class, is_correct


# Generate random single-digit equation
def generate_single_digit_equation(folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    digits = [cls for cls in class_names if cls in '0123456789']
    operators = ['add', 'div', 'mul', 'sub']

    # Choose first digit
    digit1 = random.choice(digits)
    digit1_img = random.choice(images_by_class[digit1]) if images_by_class[digit1] else None
    digit1_expected = digit1

    # Choose operator
    op = random.choice(operators)
    op_img = random.choice(images_by_class[op]) if images_by_class[op] else None
    op_expected = op

    # Choose second digit (avoid 0 for div)
    digit2 = random.choice([d for d in digits if d != '0']) if op == 'div' else random.choice(digits)
    digit2_img = random.choice(images_by_class[digit2]) if images_by_class[digit2] else None
    digit2_expected = digit2

    if not (digit1_img and op_img and digit2_img):
        return None, None, None, None

    # Classify images
    image_files = [digit1_img, op_img, digit2_img]
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    predictions = [predict_symbol(path, model, class_names) for path in image_paths]

    # Form expected equation
    expected_classes = [digit1_expected, op_expected, digit2_expected]
    expected_equation = ''
    for i, cls in enumerate(expected_classes):
        if cls in operator_mapping:
            expected_equation += operator_mapping[cls]
        else:
            expected_equation += cls
        if i == 0 or i == 1:
            expected_equation += ' '
    expected_equation = expected_equation.strip()

    # Form predicted equation
    predicted_equation = ''
    for i, pred in enumerate(predictions):
        if pred in operator_mapping:
            predicted_equation += operator_mapping[pred]
        else:
            predicted_equation += pred
        if i == 0 or i == 1:
            predicted_equation += ' '
    predicted_equation = predicted_equation.strip()

    is_correct = (expected_equation == predicted_equation)

    print(f"\nSingle-Digit Equation Classification:")
    print(f"Images: {', '.join(image_files)}")
    print(f"Expected equation: {expected_equation}")
    print(f"Predicted equation: {predicted_equation}")
    print(f"{'Correct!' if is_correct else 'Incorrect...'}")
    display_equation_images(image_paths, f"Expected: {expected_equation}\nPredicted: {predicted_equation}"
                                         f"\n{'Correct!' if is_correct else 'Incorrect...'}")

    return image_files, expected_equation, predicted_equation, is_correct


# Generate random multi-digit equation
def generate_multi_digit_equation(folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    digits = [cls for cls in class_names if cls in '0123456789']
    operators = ['add', 'div', 'mul', 'sub']

    # Choose operator
    op = random.choice(operators)
    op_img = random.choice(images_by_class[op]) if images_by_class[op] else None
    op_expected = op

    # Choose first number (single or double digit before operator)
    is_double_first = random.choice([True, False])
    first_imgs = []
    first_expected = []
    first_preds = []
    if is_double_first:
        for _ in range(2):
            digit = random.choice(digits)
            img = random.choice(images_by_class[digit]) if images_by_class[digit] else None
            if img:
                first_imgs.append(img)
                first_expected.append(digit)
                first_preds.append(predict_symbol(os.path.join(folder_path, img), model, class_names))
    else:
        digit = random.choice(digits)
        img = random.choice(images_by_class[digit]) if images_by_class[digit] else None
        if img:
            first_imgs.append(img)
            first_expected.append(digit)
            first_preds.append(predict_symbol(os.path.join(folder_path, img), model, class_names))

    # Choose second number (single or double digit after operator)
    is_double_second = random.choice([True, False])
    second_imgs = []
    second_expected = []
    second_preds = []
    valid_digits = [d for d in digits if d != '0'] if op == 'div' else digits
    if is_double_second:
        if op == 'div':
            first_digit = random.choice(valid_digits)
            second_digit = random.choice(digits)
            imgs = [
                random.choice(images_by_class[first_digit]) if images_by_class[first_digit] else None,
                random.choice(images_by_class[second_digit]) if images_by_class[second_digit] else None
            ]
            if all(imgs):
                second_imgs.extend(imgs)
                second_expected.extend([first_digit, second_digit])
                second_preds.extend([
                    predict_symbol(os.path.join(folder_path, imgs[0]), model, class_names),
                    predict_symbol(os.path.join(folder_path, imgs[1]), model, class_names)
                ])
        else:
            for _ in range(2):
                digit = random.choice(digits)
                img = random.choice(images_by_class[digit]) if images_by_class[digit] else None
                if img:
                    second_imgs.append(img)
                    second_expected.append(digit)
                    second_preds.append(predict_symbol(os.path.join(folder_path, img), model, class_names))
    else:
        digit = random.choice(valid_digits) if op == 'div' else random.choice(digits)
        img = random.choice(images_by_class[digit]) if images_by_class[digit] else None
        if img:
            second_imgs.append(img)
            second_expected.append(digit)
            second_preds.append(predict_symbol(os.path.join(folder_path, img), model, class_names))

    if not (op_img and first_imgs and second_imgs):
        return None, None, None, None

    # Form expected equation
    expected_classes = first_expected + [op_expected] + second_expected
    expected_equation = ''
    for i, cls in enumerate(expected_classes):
        if cls in operator_mapping:
            expected_equation += operator_mapping[cls]
        else:
            expected_equation += cls
        if i == len(first_expected) - 1 or i == len(first_expected):
            expected_equation += ' '
    expected_equation = expected_equation.strip()

    # Form predicted equation
    image_files = first_imgs + [op_img] + second_imgs
    predictions = first_preds + [predict_symbol(os.path.join(folder_path, op_img), model, class_names)] + second_preds
    predicted_equation = ''
    for i, pred in enumerate(predictions):
        if pred in operator_mapping:
            predicted_equation += operator_mapping[pred]
        else:
            predicted_equation += pred
        if i == len(first_preds) - 1 or i == len(first_preds):
            predicted_equation += ' '
    predicted_equation = predicted_equation.strip()

    is_correct = (expected_equation == predicted_equation)

    print(f"\nMulti-Digit Equation Classification:")
    print(f"Images: {', '.join(image_files)}")
    print(f"Expected equation: {expected_equation}")
    print(f"Predicted equation: {predicted_equation}")
    print(f"{'Correct!' if is_correct else 'Incorrect...'}")
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    display_equation_images(image_paths, f"Expected: {expected_equation}\nPredicted: {predicted_equation}"
                                         f"\n{'Correct!' if is_correct else 'Incorrect...'}")

    return image_files, expected_equation, predicted_equation, is_correct


def generate_double_digit_equation(folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    digits = [cls for cls in class_names if cls in '0123456789']
    operators = ['add', 'div', 'mul', 'sub']

    first_imgs = []
    first_expected = []
    first_preds = []
    for _ in range(2):
        digit = random.choice(digits)
        img = random.choice(images_by_class[digit]) if images_by_class[digit] else None
        if img:
            first_imgs.append(img)
            first_expected.append(digit)
            first_preds.append(predict_symbol(os.path.join(folder_path, img), model, class_names))

    # Choose operator
    op = random.choice(operators)
    op_img = random.choice(images_by_class[op]) if images_by_class[op] else None
    op_expected = op

    second_imgs = []
    second_expected = []
    second_preds = []
    valid_digits = [d for d in digits if d != '0'] if op == 'div' else digits

    for _ in range(2):
        digit = random.choice(valid_digits) if op == 'div' and len(second_expected) == 0 else random.choice(digits)
        img = random.choice(images_by_class[digit]) if images_by_class[digit] else None
        if img:
            second_imgs.append(img)
            second_expected.append(digit)
            second_preds.append(predict_symbol(os.path.join(folder_path, img), model, class_names))

    if not (op_img and first_imgs and second_imgs):
        return None, None, None, None

    # Form expected equation
    expected_classes = first_expected + [op_expected] + second_expected
    expected_equation = ''
    for i, cls in enumerate(expected_classes):
        if cls in operator_mapping:
            expected_equation += operator_mapping[cls]
        else:
            expected_equation += cls
        if i == 1 or i == 2:
            expected_equation += ' '
    expected_equation = expected_equation.strip()

    # Form predicted equation
    image_files = first_imgs + [op_img] + second_imgs
    predictions = first_preds + [
        predict_symbol(os.path.join(folder_path, op_img), model, class_names)] + second_preds
    predicted_equation = ''
    for i, pred in enumerate(predictions):
        if pred in operator_mapping:
            predicted_equation += operator_mapping[pred]
        else:
            predicted_equation += pred
        if i == 1 or i == 2:
            predicted_equation += ' '
    predicted_equation = predicted_equation.strip()

    is_correct = (expected_equation == predicted_equation)

    print(f"\nDouble-Digit Equation Classification:")
    print(f"Images: {', '.join(image_files)}")
    print(f"Expected equation: {expected_equation}")
    print(f"Predicted equation: {predicted_equation}")
    print(f"{'Correct!' if is_correct else 'Incorrect...'}")
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    display_equation_images(image_paths, f"Expected: {expected_equation}\nPredicted: {predicted_equation}"
                                         f"\n{'Correct!' if is_correct else 'Incorrect...'}")

    return image_files, expected_equation, predicted_equation, is_correct


# Main interactive loop
def main():
    output_file = "interactive_classification_results.txt"
    results = []
    total_count = 0
    correct_count = 0
    incorrect_count = 0

    while True:
        print("\nChoose classification type:")
        print("1. Individual Symbol Classification")
        print("2. Equation Classification")
        print("3. Exit")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == '1':
            image_file, expected_class, predicted_class, is_correct = classify_single_symbol(
                handwritten_folder, model, class_names
            )
            if image_file:
                results.append({
                    'type': 'Individual Symbol',
                    'image': image_file,
                    'expected': expected_class,
                    'predicted': predicted_class,
                    'is_correct': is_correct
                })
                total_count += 1
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1
            else:
                print("No images available for classification.")

        elif choice == '2':
            print("\nChoose equation type:")
            print("1. Single Digit (e.g., 1 + 2)")
            print("2. Multi-Digit (e.g., 1 + 23, 12 + 3, 12 + 34)")
            print("3. Double-Digit (e.g., 12 + 34)")
            eq_choice = input("Enter 1, 2 or 3: ").strip()

            if eq_choice == '1':
                image_files, expected_equation, predicted_equation, is_correct = generate_single_digit_equation(
                    handwritten_folder, model, class_names
                )
                if image_files:
                    results.append({
                        'type': 'Single-Digit Equation',
                        'images': image_files,
                        'expected': expected_equation,
                        'predicted': predicted_equation,
                        'is_correct': is_correct
                    })
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                else:
                    print("Failed to generate equation (missing images).")

            elif eq_choice == '2':
                image_files, expected_equation, predicted_equation, is_correct = generate_multi_digit_equation(
                    handwritten_folder, model, class_names
                )
                if image_files:
                    results.append({
                        'type': 'Multi-Digit Equation',
                        'images': image_files,
                        'expected': expected_equation,
                        'predicted': predicted_equation,
                        'is_correct': is_correct
                    })
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                else:
                    print("Failed to generate equation (missing images).")

            elif eq_choice == '3':
                image_files, expected_equation, predicted_equation, is_correct = generate_double_digit_equation(
                    handwritten_folder, model, class_names
                )
                if image_files:
                    results.append({
                        'type': 'Double-Digit Equation',
                        'images': image_files,
                        'expected': expected_equation,
                        'predicted': predicted_equation,
                        'is_correct': is_correct
                    })
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                else:
                    print("Failed to generate equation (missing images).")

            else:
                print("Invalid choice. Please enter 1, 2, or 3..")

        elif choice == '3':
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Calculate and display accuracy
    accuracy_percentage = (correct_count / total_count * 100) if total_count > 0 else 0
    print(f"\nSession Summary:")
    print(f"Total classifications: {total_count}")
    print(f"Correct classifications: {correct_count}")
    print(f"Incorrect classifications: {incorrect_count}")
    print(f"Accuracy: {accuracy_percentage:.2f}%")

    # Save results to file
    with open(output_file, 'w') as f:
        for i, result in enumerate(results):
            f.write(f"Result {i + 1}:\n")
            if result['type'] == 'Individual Symbol':
                f.write(f"Type: Individual Symbol Classification\n")
                f.write(f"Image: {result['image']}\n")
                f.write(f"Expected classification: {result['expected']}\n")
                f.write(f"Predicted classification: {result['predicted']}\n")
                f.write(f"{'Correct!' if result['is_correct'] else 'Incorrect...'}\n")
            else:
                f.write(f"Type: {result['type']}\n")
                f.write(f"Images: {', '.join(result['images'])}\n")
                f.write(f"Expected equation: {result['expected']}\n")
                f.write(f"Predicted equation: {result['predicted']}\n")
                f.write(f"{'Correct!' if result['is_correct'] else 'Incorrect...'}\n")
            f.write("\n")
        f.write(f"Session Summary:")
        f.write(f"Total classifications: {total_count}")
        f.write(f"Correct classifications: {correct_count}")
        f.write(f"Incorrect classifications: {incorrect_count}")
        f.write(f"Accuracy: {accuracy_percentage:.2f}%")

    print(f"\nSession ended. Results saved to {output_file}")


if __name__ == "__main__":
    main()