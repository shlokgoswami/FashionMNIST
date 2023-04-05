# FashionMNIST
# Problem Statement:
The problem statement for this code is to develop a convolutional neural network (CNN) model using only Numpy and Scipy to classify the FashionMNIST dataset. The goal is to achieve high accuracy in predicting the labels of the images in the test dataset.



# Solution
FashionMNIST_CNN_MODEL.py conrains rhe soltuion, the solution is implemented in Python and uses the NumPy and SciPy libraries to efficiently perform matrix operations and convolution calculations for the deep learning model. The code is organized into classes for each layer of the CNN model, including a fully connected layer, activation layer, reshape layer, and convolutional layer.

The code defines a function to calculate mean squared error (MSE) and its derivative, which are used as the loss function and loss derivative function for backpropagation during the training of the CNN model. The code also includes functions to predict the labels of the test dataset and to train the CNN model using the backpropagation algorithm.

The CNN model architecture includes several layers, such as convolutional layers with different kernel sizes, activation layers with different activation functions, fully connected layers with different output sizes, and reshape layers to reshape the output of the convolutional layers. The network is trained on the FashionMNIST dataset using the training dataset, and the accuracy of the model is evaluated on the test dataset.
