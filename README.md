# AI Digit Classifier using Pytorch

# Installation
```
git clone https://github.com/AshDaGrat/Election-software
```
```
pip install -r requirements.txt
```
<br>

# Training your own model

To train your own model, run NN_training.py

edit line 123 with the path and name of your model

<img src = "https://cdn.discordapp.com/attachments/1073895599910436915/1134545715796922448/image.png">

NN_training.py trains a digit classifier using the MNIST dataset and data augmentation. It defines a neural network with three layers, applies affine transformations to the images, and optimizes the model using backpropagation. The training loop iterates through the data loader and prints the training loss after each epoch.

<br>

# Evaluating your model

To evaluate your model run eval.py 

edit line 7 with the path to your model 

<img src = "https://cdn.discordapp.com/attachments/1073895599910436915/1134547002890399795/image.png">

eval.py evaluates the accuracy of a pre-trained digit classifier model on the MNIST validation dataset. It loads the model, transforms the images, and uses a data loader to iterate through the validation dataset. The code calculates the accuracy by comparing the predicted labels with the true labels and prints the number of images tested and the model's accuracy.

<br>

# Running 

to run the program, run main.py after editing line 10 with the path to your model

<img src = "https://cdn.discordapp.com/attachments/1073895599910436915/1134545361483075705/image.png">

This creates a simple GUI using Tkinter to draw digits and then uses a pre-trained model to classify the drawn digit. The user can draw a digit on the canvas, and by clicking the "find" button, the model predicts the digit. The "reset" button clears the canvas.

<img src = "https://cdn.discordapp.com/attachments/1073895599910436915/1134548715437293640/image.png">

After clicking the "find" button, the neural netowork's prediction is outputed in the console. 

# TODO
1. Output in GUI and not in console
2. Live guess updation as user draws digit
3. Output confidence of guess
4. Convolutional Neural Networks for better accuracy