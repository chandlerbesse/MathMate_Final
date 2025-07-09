import os
import random

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


# Group images by class
def group_images_by_class(folder_path):
    images_by_class = {cls: [] for cls in class_names}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            class_raw = filename.split("_")[0]
            class_name = digit_mapping.get(class_raw, class_raw)
            if class_name in class_names:
                images_by_class[class_name].append(filename)
    return images_by_class


# Generate equations with single-digit numbers
def generate_single_digit_equations(num_equations, folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    equations = []
    correct_equations = 0
    digits = [cls for cls in class_names if cls in "0123456789"]
    operators = ["add", "div", "mul", "sub"]

    for i in range(num_equations):
        print(f"\nGenerating single-digit equation {i + 1}:")
        # Choose first digit
        digit1 = random.choice(digits)
        digit1_img = (
            random.choice(images_by_class[digit1]) if images_by_class[digit1] else None
        )
        digit1_expected = digit1

        # Choose operator
        op = random.choice(operators)
        op_img = random.choice(images_by_class[op]) if images_by_class[op] else None
        op_expected = op

        # Choose second digit (avoid 0 for div)
        digit2 = (
            random.choice([d for d in digits if d != "0"])
            if op == "div"
            else random.choice(digits)
        )
        digit2_img = (
            random.choice(images_by_class[digit2]) if images_by_class[digit2] else None
        )
        digit2_expected = digit2

        # Check if all images are available
        if not (digit1_img and op_img and digit2_img):
            print("Failed to generate equation (missing images for some classes).")
            continue

        # Classify images
        image_files = [digit1_img, op_img, digit2_img]
        image_paths = [os.path.join(folder_path, f) for f in image_files]
        predictions = [predict_symbol(path, model, class_names) for path in image_paths]

        # Form expected equation
        expected_classes = [digit1_expected, op_expected, digit2_expected]
        expected_equation = ""
        for i, cls in enumerate(expected_classes):
            if cls in operator_mapping:
                expected_equation += operator_mapping[cls]
            else:
                expected_equation += cls
            if i == 0 or i == 1:
                expected_equation += " "
        expected_equation = expected_equation.strip()

        # Form predicted equation
        predicted_equation = ""
        for i, pred in enumerate(predictions):
            if pred in operator_mapping:
                predicted_equation += operator_mapping[pred]
            else:
                predicted_equation += pred
            if i == 0 or i == 1:
                predicted_equation += " "
        predicted_equation = predicted_equation.strip()

        # Track correctness
        equations.append((image_files, expected_equation, predicted_equation))
        if expected_equation == predicted_equation:
            correct_equations += 1

        print("Selected images:", image_files)
        print("Expected equation:", expected_equation)
        print("Predicted equation:", predicted_equation)

    # Calculate and display accuracy
    total_equations = len(equations)
    accuracy_percentage = (
        (correct_equations / total_equations * 100) if total_equations > 0 else 0
    )
    print(f"\nSingle-Digit Classification Results:")
    print(f"Correctly classified equations: {correct_equations}/{total_equations}")
    print(f"Percentage correct: {accuracy_percentage:.2f}%")

    return equations, correct_equations, total_equations


# Generate equations with single and/or double-digit numbers
def generate_mixed_digit_equations(num_equations, folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    equations = []
    correct_equations = 0
    digits = [cls for cls in class_names if cls in "0123456789"]
    operators = ["add", "div", "mul", "sub"]

    for i in range(num_equations):
        # print(f"\nGenerating mixed-digit equation {i + 1}:")
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
                img = (
                    random.choice(images_by_class[digit])
                    if images_by_class[digit]
                    else None
                )
                if img:
                    first_imgs.append(img)
                    first_expected.append(digit)
                    first_preds.append(
                        predict_symbol(
                            os.path.join(folder_path, img), model, class_names
                        )
                    )
        else:
            digit = random.choice(digits)
            img = (
                random.choice(images_by_class[digit])
                if images_by_class[digit]
                else None
            )
            if img:
                first_imgs.append(img)
                first_expected.append(digit)
                first_preds.append(
                    predict_symbol(os.path.join(folder_path, img), model, class_names)
                )

        # Choose second number (single or double-digit after operator)
        is_double_second = random.choice([True, False])
        second_imgs = []
        second_expected = []
        second_preds = []
        valid_digits = [d for d in digits if d != "0"] if op == "div" else digits
        if is_double_second:
            if op == "div":
                first_digit = random.choice(valid_digits)
                second_digit = random.choice(digits)
                imgs = [
                    (
                        random.choice(images_by_class[first_digit])
                        if images_by_class[first_digit]
                        else None
                    ),
                    (
                        random.choice(images_by_class[second_digit])
                        if images_by_class[second_digit]
                        else None
                    ),
                ]
                if all(imgs):
                    second_imgs.extend(imgs)
                    second_expected.extend([first_digit, second_digit])
                    second_preds.extend(
                        [
                            predict_symbol(
                                os.path.join(folder_path, imgs[0]), model, class_names
                            ),
                            predict_symbol(
                                os.path.join(folder_path, imgs[1]), model, class_names
                            ),
                        ]
                    )
            else:
                for _ in range(2):
                    digit = random.choice(digits)
                    img = (
                        random.choice(images_by_class[digit])
                        if images_by_class[digit]
                        else None
                    )
                    if img:
                        second_imgs.append(img)
                        second_expected.append(digit)
                        second_preds.append(
                            predict_symbol(
                                os.path.join(folder_path, img), model, class_names
                            )
                        )
        else:
            digit = (
                random.choice(valid_digits) if op == "div" else random.choice(digits)
            )
            img = (
                random.choice(images_by_class[digit])
                if images_by_class[digit]
                else None
            )
            if img:
                second_imgs.append(img)
                second_expected.append(digit)
                second_preds.append(
                    predict_symbol(os.path.join(folder_path, img), model, class_names)
                )

        # Check if all images are available
        if not (op_img and first_imgs and second_imgs):
            print("Failed to generate equation (missing images for some classes).")
            continue

        # Form expected equation
        expected_classes = first_expected + [op_expected] + second_expected
        expected_equation = ""
        for i, cls in enumerate(expected_classes):
            if cls in operator_mapping:
                expected_equation += operator_mapping[cls]
            else:
                expected_equation += cls
            if (
                i == len(first_expected) - 1
                or i == len(first_expected)
                or i == len(first_expected) + len(second_expected)
            ):
                expected_equation += " "
        expected_equation = expected_equation.strip()

        # Form predicted equation
        image_files = first_imgs + [op_img] + second_imgs
        predictions = (
            first_preds
            + [predict_symbol(os.path.join(folder_path, op_img), model, class_names)]
            + second_preds
        )
        predicted_equation = ""
        for i, pred in enumerate(predictions):
            if pred in operator_mapping:
                predicted_equation += operator_mapping[pred]
            else:
                predicted_equation += pred
            if (
                i == len(first_preds) - 1
                or i == len(first_preds)
                or i == len(first_preds) + len(second_preds)
            ):
                predicted_equation += " "
        predicted_equation = predicted_equation.strip()

        # Track correctness
        equations.append((image_files, expected_equation, predicted_equation))
        if expected_equation == predicted_equation:
            correct_equations += 1

        print("Selected images:", image_files)
        print("Expected equation:", expected_equation)
        print("Predicted equation:", predicted_equation)

    # Calculate and display accuracy
    total_equations = len(equations)
    accuracy_percentage = (
        (correct_equations / total_equations * 100) if total_equations > 0 else 0
    )
    print(f"\nMixed-Digit Classification Results:")
    print(f"Correctly classified equations: {correct_equations}/{total_equations}")
    print(f"Percentage correct: {accuracy_percentage:.2f}%")

    return equations, correct_equations, total_equations


# Generate equations with single and/or double-digit numbers
def generate_double_digit_equations(num_equations, folder_path, model, class_names):
    images_by_class = group_images_by_class(folder_path)
    equations = []
    correct_equations = 0
    digits = [cls for cls in class_names if cls in "0123456789"]
    operators = ["add", "div", "mul", "sub"]

    for i in range(num_equations):
        print(f"\nGenerating double-digit equation {i + 1}:")
        # Choose first number (double-digit)
        first_imgs = []
        first_expected = []
        first_preds = []
        for _ in range(2):
            digit = random.choice(digits)
            img = (
                random.choice(images_by_class[digit])
                if images_by_class[digit]
                else None
            )
            if img:
                first_imgs.append(img)
                first_expected.append(digit)
                first_preds.append(
                    predict_symbol(os.path.join(folder_path, img), model, class_names)
                )

        # Choose operator
        op = random.choice(operators)
        op_img = random.choice(images_by_class[op]) if images_by_class[op] else None
        op_expected = op

        # Choose second number (double-digit, avoid 0 for div)
        second_imgs = []
        second_expected = []
        second_preds = []
        valid_digits = [d for d in digits if d != "0"] if op == "div" else digits
        for _ in range(2):
            digit = (
                random.choice(valid_digits)
                if op == "div" and len(second_expected) == 0
                else random.choice(digits)
            )
            img = (
                random.choice(images_by_class[digit])
                if images_by_class[digit]
                else None
            )
            if img:
                second_imgs.append(img)
                second_expected.append(digit)
                second_preds.append(
                    predict_symbol(os.path.join(folder_path, img), model, class_names)
                )

        # Check if all images are available
        if not (op_img and len(first_imgs) == 2 and len(second_imgs) == 2):
            print("Failed to generate equation (missing images for some classes).")
            continue

        # Form expected equation
        expected_classes = first_expected + [op_expected] + second_expected
        expected_equation = ""
        for i, cls in enumerate(expected_classes):
            if cls in operator_mapping:
                expected_equation += operator_mapping[cls]
            else:
                expected_equation += cls
            if i == 1 or i == 2 or i == 4:
                expected_equation += " "
        expected_equation = expected_equation.strip()

        # Form predicted equation
        image_files = first_imgs + [op_img] + second_imgs
        predictions = (
            first_preds
            + [predict_symbol(os.path.join(folder_path, op_img), model, class_names)]
            + second_preds
        )
        predicted_equation = ""
        for i, pred in enumerate(predictions):
            if pred in operator_mapping:
                predicted_equation += operator_mapping[pred]
            else:
                predicted_equation += pred
            if i == 1 or i == 2 or i == 4:
                predicted_equation += " "
        predicted_equation = predicted_equation.strip()

        # Track correctness
        equations.append((image_files, expected_equation, predicted_equation))
        if expected_equation == predicted_equation:
            correct_equations += 1

        print("Selected images:", image_files)
        print("Expected equation:", expected_equation)
        print("Predicted equation:", predicted_equation)

    # Calculate and display accuracy
    total_equations = len(equations)
    accuracy_percentage = (
        (correct_equations / total_equations * 100) if total_equations > 0 else 0
    )
    print(f"\nDouble-Digit Classification Results:")
    print(f"Correctly classified equations: {correct_equations}/{total_equations}")
    print(f"Percentage correct: {accuracy_percentage:.2f}%")

    return equations, correct_equations, total_equations


# Main function to test all equation generators
def main():
    num_equations = 5  # Number of equations generated per equation type
    output_file = "generated_equations_test.txt"

    # Generate random single-digit equations
    print("\n=== Testing Single-Digit Equations ===")
    single_equations, single_correct, single_total = generate_single_digit_equations(
        num_equations, handwritten_folder, model, class_names
    )

    # Generate random mixed-digit equations
    print("\n=== Testing Mixed-Digit Equations ===")
    mixed_equations, mixed_correct, mixed_total = generate_mixed_digit_equations(
        num_equations, handwritten_folder, model, class_names
    )

    # Generate random double-digit equations
    print("\n=== Testing Double-Digit Equations ===")
    double_equations, double_correct, double_total = generate_double_digit_equations(
        num_equations, handwritten_folder, model, class_names
    )

    # Save all results to output_file
    with open(output_file, "w") as f:
        f.write("=== Single-Digit Equations ===\n")
        for i, (image_files, expected_equation, predicted_equation) in enumerate(
            single_equations
        ):
            f.write(f"Equation {i + 1}:\n")
            f.write(f"Images: {', '.join(image_files)}\n")
            f.write(f"Expected equation: {expected_equation}\n")
            f.write(f"Predicted equation: {predicted_equation}\n\n")
        f.write(f"Classification Results:\n")
        f.write(f"Correctly classified equations: {single_correct}/{single_total}\n")
        f.write(
            f"Percentage correct: {(single_correct / single_total * 100) if single_total > 0 else 0:.2f}%\n\n"
        )

        f.write("=== Mixed-Digit Equations ===\n")
        for i, (image_files, expected_equation, predicted_equation) in enumerate(
            mixed_equations
        ):
            f.write(f"Equation {i + 1}:\n")
            f.write(f"Images: {', '.join(image_files)}\n")
            f.write(f"Expected equation: {expected_equation}\n")
            f.write(f"Predicted equation: {predicted_equation}\n\n")
        f.write(f"Classification Results:\n")
        f.write(f"Correctly classified equations: {mixed_correct}/{mixed_total}\n")
        f.write(
            f"Percentage correct: {(mixed_correct / mixed_total * 100) if mixed_total > 0 else 0:.2f}%\n\n"
        )

        f.write("=== Double-Digit Equations ===\n")
        for i, (image_files, expected_equation, predicted_equation) in enumerate(
            double_equations
        ):
            f.write(f"Equation {i + 1}:\n")
            f.write(f"Images: {', '.join(image_files)}\n")
            f.write(f"Expected equation: {expected_equation}\n")
            f.write(f"Predicted equation: {predicted_equation}\n\n")
        f.write(f"Classification Results:\n")
        f.write(f"Correctly classified equations: {double_correct}/{double_total}\n")
        f.write(
            f"Percentage correct: {(double_correct / double_total * 100) if double_total > 0 else 0:.2f}%\n"
        )

    print(f"\nAll results saved to {output_file}")


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
print("Class names:", class_names)

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

# Mapping for operator classes to symbols
operator_mapping = {"add": "+", "div": "/", "eq": "=", "mul": "*", "sub": "-"}

if __name__ == "__main__":
    main()
