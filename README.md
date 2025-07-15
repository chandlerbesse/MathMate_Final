# MathMate

## Description
MathMate is designed to classify images of digits (0-9) and operators (+,-,*,/,=) to generate and solve basic arithmetic equations. Models are trained
using a CNN architecture:
- 4 augmentation layers 
- 2 convolution layers with 32 filters, (3, 3) kernel
- 1 max pooling layer (2, 2)
- 2 convolution layers with 64 filters, (3, 3) kernel
- 1 max pooling layer (2, 2)
- 1 convolution layer with 128 filters, (3, 3) kernel
- 1 convolution layer with 128 filters, (1, 1) kernel
- 1 max pooling layer (2, 2)
- Flatten
- 1 dense layer with 256 neurons
- Dropout 0.5
- 1 dense layer with 15 neurons
## How to run:
- Run the command `pip install -r requirements.txt`
- To run the interactive program, run the command `python mathmate_interactive.py` then enter the commands as prompted on screen. 
  - Based on your response(s), the program will output both the expected/predicted classifications and provide a visual of the handwritten digit or equation.
- If you would like to train a new model:
  - Open `mathmate_arithmetic_training.py` and locate the line `cnn_model.save("mathmate_basic_arithmetic_model8.keras")` by using `ctrl+f`.
  - Enter a new name to save the model as (otherwise it will rewrite the current model).
  - Tweak the parameters as necessary.
  - Run `mathmate_arithmetic_training.py` to train the new model.
  - NOTE: To assess this new model, you will need to use `ctrl+f` in the other scripts to find `model_file = "mathmate_basic_arithmetic_model8.keras"` and change it to the new model's name.
   
