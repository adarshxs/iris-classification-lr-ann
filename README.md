# iris-classification-lr-ann
Iris Classification Model using Logistic Regression and Artificial Neural Network

In this project, I used logistic regression and artificial neural network to classify the iris dataset on 4 features: sepal length, sepal width, petal length, and petal width. The iris dataset was loaded using scikit-learn and the model was trained using PyTorch.

First, I loaded the iris dataset using scikit-learn’s load_iris function. Then, I split the data into training and test sets using train_test_split function in the ratio of 80:20. After that, I trained a logistic regression model and an artificial neural network model on the training data. Finally, I evaluated the performance of both models on the test data by storing the loss and accuracy history and then plotting them later on.

To prepare the data for training, I first split it into training and test sets using `train_test_split` function from scikit-learn. Then, I standardized the data using `StandardScaler` and converted the data into PyTorch tensors. After that, I created data loaders for training and test data using `DataLoader` class from PyTorch.

I also defined two models: a logistic regression model and a neural network model. The logistic regression model has one linear layer with 4 input features and 3 output classes. The neural network model has two fully connected layers with a `ReLU` activation function in between(I did look out for other activation functions but ReLU worked the best imo).

To train the models, I defined a train function that takes in a model, a loss function (criterion), an optimizer, and the number of epochs as inputs(2000 epochs worked best for me). Inside the function, I looped over the number of epochs and for each epoch, I looped over the batches of data in the training data loader. For each batch, I performed a forward pass to compute the predicted outputs (y_pred) and the loss. Then, I performed a backward pass to compute the gradients and update the model parameters using the optimizer.

After each epoch, I computed the loss and accuracy on the test data and appended them to loss_history and accuracy_history lists respectively.

I trained both the logistic regression model and the neural network model using this train function with `CrossEntropyLoss` as the loss function and `SGD` as the optimizer.

After training was complete, I plotted the loss and accuracy curves for both models using `matplotlib`. The loss curve showed how the loss changed over the epochs during training, while the accuracy curve showed how the accuracy on the test data changed over the epochs.

## Results are as follows:
For data split 80:20 and batch size = 8 and random state = 1

![image](https://user-images.githubusercontent.com/114558126/235588414-119812b0-1763-43b8-800b-d452e80f537a.png)
![image](https://user-images.githubusercontent.com/114558126/235588429-2384487d-c788-4a77-9ec8-0175e66f0de5.png)

---
For data split 80:20 and batch size = 16 and random state = 0

![image](https://user-images.githubusercontent.com/114558126/235588902-60d055ad-c742-4b38-b3b7-a92c0bfce9eb.png)
![image](https://user-images.githubusercontent.com/114558126/235588901-4b217d24-3524-476d-bb6c-7a5eb9b002ac.png)



