import numpy as np
import math

# Log in to your W&B account
import wandb
import os
os.environ['WAND_NOTEBOOK_NAME']='train'
# !wandb login 8f26d3215193b9c0e8e37007dfbb313be26db111

# argparse is used to get inputs from the user on the command line interface
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("-wp",     "--wandb_project",   help="project_name",       type=str,                                                                  default="Assignment 1")
parser.add_argument("-we",     "--wandb_entity",    help="entity",             type=str,                                                                  default="cs22m70")
parser.add_argument("-d",      "--dataset",         help="dataset_name",       type=str,   choices=["fashion_mnist","mnist"],                             default="fashion_mnist")
parser.add_argument("-m",      "--momentum",        help="m",                  type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("-beta",   "--beta",            help="beta",               type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("-beta1",  "--beta1",           help="beta1",              type=float, choices=[0.5,0.9],                                             default=0.9)
parser.add_argument("-beta2",  "--beta2",           help="beta2",              type=float, choices=[0.999,0.5],                                           default=0.999)
parser.add_argument("-eps",    "--epsilon",         help="epsilon",            type=float, choices=[1e-3,1e-4],                                           default=1e-3)
parser.add_argument("-w_d",    "--weight_decay",    help="alpha",              type=float, choices=[0,0.0005,0.5],                                        default=0)
parser.add_argument("-o",      "--optimizer",       help="loss_function",      type=str,   choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],default="adam")
parser.add_argument("-lr",     "--learning_rate",   help="lr",                 type=float, choices=[1e-4,1e-3],                                           default=1e-3)
parser.add_argument("-e",      "--epochs",          help="epochs",             type=int,   choices=[5,10],                                                default=10)
parser.add_argument("-b",      "--batch_size",      help="batch_size",         type=int,   choices=[1,16,32,64],                                          default=32)
parser.add_argument("-nhl",    "--num_layers",      help="hidden_layer",       type=int,   choices=[3,4,5],                                               default=5)
parser.add_argument("-w_i",    "--weight_init",     help="weight_init",        type=str,   choices=["random","Xavier"],                                   default="Xavier")
parser.add_argument("-a",      "--activation",      help="activation_function",type=str,   choices=["ReLU","tanh","sigmoid"],                             default="ReLU")
parser.add_argument("-sz",     "--hidden_size",     help="hidden_layer_size",  type=int,   choices=[32,64,128],                                           default=128)
parser.add_argument("-l",      "--loss",            help="loss_function",      type=str,   choices=["mean_squared_error", "cross_entropy"],               default="cross_entropy")
args=parser.parse_args()

# setting the values of parameters that are taken from argparse to the variable names used in the code
project_name=args.wandb_project
entity=args.wandb_entity
dataset_name=args.dataset
m=args.momentum
epsilon=args.epsilon
alpha=args.weight_decay
hidden_layer=args.num_layers
activation_function=args.activation
hidden_layer_size=args.hidden_size
loss_function=args.loss
beta=args.beta
beta1=args.beta1
beta2=args.beta2
lr=args.lr
epochs=args.epochs
optimizer=args.optimizer
weight_init=args.weight_init
loss_function=args.loss
batch_size=args.batch_size

# default paramenters used while running a sweep on wandb
# default_parameters=dict(
#     optimizer="adam",
#     lr=1e-3,
#     epochs=10,
#     hidden_layer_size=128,
#     activation_function="ReLU",
#     weight_init="Xavier",
#     hidden_layer=5,
#     batch_size=32,
#     alpha=0
# )
# config=default_parameters
run=wandb.init(project=project_name,entity=entity,name="train",reinit ='True')
# config=wandb.config

classes=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# The function reinit_labels converts a label vector into a matrix of one-hot vectors.
def reinit_labels(label_vector):
    num_classes = label_vector.max() + 1  # Determine the number of classes.
    one_hot_labels = []
    for label in label_vector:
        one_hot = [0] * num_classes  # Initialize a one-hot vector for each label.
        one_hot[label] = 1  # Set the corresponding index to 1.
        one_hot_labels.append(one_hot)
    return np.array(one_hot_labels)

# Importing the training and test datasets based on the specified dataset name.
if dataset_name == "fashion_mnist":
    from keras.datasets import fashion_mnist
    (X_Train, Y_Train), (X_Test, Y_Test) = fashion_mnist.load_data()
elif dataset_name == "mnist":
    from keras.datasets import mnist
    (X_Train, Y_Train), (X_Test, Y_Test) = mnist.load_data()
else:
    print("Invalid input")
    exit()


# Reshaping and preprocessing the input data
train_samples = X_Train.shape[0]
image_size = X_Train.shape[1] * X_Train.shape[1]
test_samples = X_Test.shape[0]

# Reshape and normalize the training and test data
x_train = X_Train.reshape(train_samples, image_size) / 255
x_test = X_Test.reshape(test_samples, image_size) / 255

# Convert labels to one-hot encoded matrices
y_train = reinit_labels(Y_Train)
y_test = reinit_labels(Y_Test)

# Determine the number of output classes
output_size = y_train.shape[1]

# Split the input data into training and validation sets (90% training, 10% validation)
validation_split = int(len(x_train) * 0.9)
val_x = x_train[validation_split:, :]
val_y = y_train[validation_split:, :]

x_train = x_train[:validation_split, :]
y_train = y_train[:validation_split, :]

# Define the output function as softmax
output_function = "softmax"


# Define the architecture of the neural network based on model parameters
def layer_init(input_size, output_size, hidden_layer_size, hidden_layers, activation_function, output_function):
    layers = []
    # Define the input layer with its size and activation function
    layers.append([input_size, hidden_layer_size, activation_function])
    # Define hidden layers with their size and activation function
    for _ in range(hidden_layers - 1):
        layers.append([hidden_layer_size, hidden_layer_size, activation_function])
    # Define the output layer with its size and output function
    layers.append([hidden_layer_size, output_size, output_function])
    return layers

# Initialize weights and biases for the network
def start_weights_and_bias(layers, weight_init):
    initial_weights = []
    initial_bias = []
  
    # Initialize weights and biases based on the chosen initialization method
    for layer in layers:
        if weight_init == "random":
            # Initialize weights randomly between -1 and 1
            weights = np.random.uniform(-1, 1, (layer[1], layer[0]))
            # Initialize biases randomly between 0 and 1
            biases = np.random.rand(1, layer[1])
        elif weight_init == "Xavier":
            # Calculate scaling factor using Xavier initialization formula
            scale = np.sqrt(6 / (layer[1] + layer[0]))
            # Initialize weights and biases using Xavier initialization
            weights = np.random.uniform(-scale, scale, (layer[1], layer[0]))
            biases = np.random.uniform(-scale, scale, (1, layer[1]))
        else:
            raise ValueError("Invalid weight initialization method.")

        initial_weights.append(weights)
        initial_bias.append(biases)

    return initial_weights, initial_bias

# Define the activation function to compute the output based on the given activation function
def activation(z, activation_function):
    # Compute the activation output based on the specified activation function
    
    if activation_function == "sigmoid":
        # Sigmoid activation function
        return 1.0 / (1.0 + np.exp(-z))
    
    elif activation_function == "softmax":
        # Softmax activation function
        output = []
        for row in z:
            # Normalize by subtracting the maximum value for numerical stability
            max_val = np.max(row)
            row = row - max_val
            # Compute softmax
            exp_vals = np.exp(row)
            softmax_vals = exp_vals / np.sum(exp_vals)
            output.append(softmax_vals)
        return np.array(output)
    
    elif activation_function == "tanh":
        # Hyperbolic tangent (tanh) activation function
        return np.tanh(z)
    
    elif activation_function == "ReLU":
        # Rectified Linear Unit (ReLU) activation function
        return np.maximum(0, z)
    
    else:
        # Handle invalid activation function
        raise ValueError("Invalid activation function.")



# Define the derivative of the activation function for a given input
def activation_derivative(z, activation_function):
    # Compute the derivative of the activation function based on the specified activation function
    
    if activation_function == "sigmoid":
        # Derivative of the sigmoid activation function
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return sigmoid * (1.0 - sigmoid)
    
    elif activation_function == "tanh":
        # Derivative of the hyperbolic tangent (tanh) activation function
        return 1 - np.tanh(z)**2
    
    elif activation_function == "ReLU":
        # Derivative of the Rectified Linear Unit (ReLU) activation function
        relu = np.maximum(0, z)
        relu[relu > 0] = 1
        return relu
    
    else:
        # Handle invalid activation function
        raise ValueError("Invalid activation function.")

# Define the loss function to calculate the loss based on the specified loss function
def loss(y_pred, y_batch, loss_function, weights, alpha):
    # Define a small epsilon to avoid division by zero
    epsilon = 1e-5
    
    if loss_function == "cross_entropy":
        # Cross-entropy loss function
        loss = -(np.multiply(y_batch, np.log(y_pred + epsilon))).sum() / len(y_pred)
        
        # Regularization term
        regularization = 0
        for weight in weights:
            regularization += np.sum(weight**2)
        regularization = (alpha / 2) * regularization / len(y_pred)
        
        return loss + regularization
    
    elif loss_function == "mean_squared_error":
        # Mean squared error loss function
        numerator = np.sum((y_pred - y_batch)**2)
        denominator = 2 * len(y_pred)
        loss = numerator / denominator
        
        return loss
    
    else:
        # Handle invalid loss function
        raise ValueError("Invalid loss function.")


# Define the accuracy function to calculate the accuracy during model training
def accuracy(y_pred, y_batch):
    # Initialize the count for correct predictions
    correct_predictions = 0
    
    # Iterate through each prediction and true label pair
    for pred, true_label in zip(y_pred, y_batch):
        # Get the index of the maximum value in the prediction and true label arrays
        pred_label = np.argmax(pred)
        true_label = np.argmax(true_label)
        
        # If the predicted label matches the true label, increment the count of correct predictions
        if pred_label == true_label:
            correct_predictions += 1
    
    # Return the accuracy as the ratio of correct predictions to the total number of predictions
    return correct_predictions

# Define the forward propagation function to compute the predicted values of the labels for a given input and network configuration
def forward_propagation(x_batch, weights, biases, n_layers, activation_function, output_function):
    # Initialize lists to store pre-activation (a) and activation (h) values at each layer in the network
    pre_activations, activations = [], []
    
    # Compute the pre-activation and activation values for the first layer
    pre_activation_1 = np.matmul(x_batch, weights[0].T) + biases[0]
    pre_activations.append(pre_activation_1)
    
    # Normalize pre-activation values
    for i in range(len(pre_activation_1)):
        pre_activation_1[i] = pre_activation_1[i] / pre_activation_1[i][np.argmax(pre_activation_1[i])]
    
    activation_1 = activation(pre_activation_1, activation_function)
    activations.append(activation_1)
    
    # Compute pre-activation and activation values for subsequent layers
    for j in range(len(n_layers) - 2):
        pre_activation_next = np.matmul(activations[j], weights[j + 1].T) + biases[j + 1]
        pre_activations.append(pre_activation_next)
        
        activation_next = activation(pre_activation_next, activation_function)
        activations.append(activation_next)
    
    # Compute pre-activation and activation values for the output layer
    final_pre_activation = np.matmul(activations[-1], weights[-1].T) + biases[-1]
    pre_activations.append(final_pre_activation)
    
    final_activation = activation(final_pre_activation, output_function)
    activations.append(final_activation)
    
    return pre_activations, activations



# Define the backward propagation function to calculate the derivatives with respect to weights and biases while moving from output to input in the network
def backward_propagation(x_batch, y_pred, y_batch, weights, a, h, n_layers, activation_function):
    # Initialize dictionaries to store derivatives with respect to weights (dw) and biases (db) at each layer in the network
    dw, db = {}, {}
    m = len(y_batch)
    
    # Compute initial derivative with respect to activations
    da_prev = y_pred - y_batch
    i = len(n_layers) - 1
    
    # Backpropagate through each layer in the network
    while i >= 1:
        # Compute derivative with respect to weights at current layer
        da = da_prev
        d = []
        z = (np.array(np.matmul(h[i - 1].T, da)).T) / m
        dw[i + 1] = z
        
        # Compute derivative with respect to biases at current layer
        for k in range(len(da[0])):
            sum = 0
            for j in range(len(da)):
                sum += da[j][k]
            d.append(sum / m)
        db[i + 1] = np.array(d)
        
        # Compute derivative with respect to activations at previous layer
        dh_prev = np.matmul(da, weights[i])
        a_new = activation_derivative(a[i - 1], activation_function)
        da_prev = np.multiply(dh_prev, a_new)
        
        i -= 1
    
    # Compute derivative with respect to weights and biases at the input layer
    d = []
    z = (np.array(np.matmul(x_batch.T, da_prev)).T) / m
    dw[1] = z
    
    for k in range(len(da_prev[0])):
        sum = 0
        for j in range(len(da_prev)):
            sum += da_prev[j][k]
        d.append(sum / m)
    db[1] = np.array(d)
    
    return dw, db



# Define the stochastic gradient descent function to train the model with a batch size of 1 for a given network configuration
def stochastic_gradient_descent(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha):
    # Initialize the network layers
    n_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

    # Initialize weights and biases based on the specified weight initialization method
    weight, bias = start_weights_and_bias(n_layers, weight_init)

    # Split the training data into the specified number of batches
    x_batch = np.array(np.array_split(x_train, batches))
    y_batch = np.array(np.array_split(y_train, batches))

    # Initialize lists to store training and validation errors and accuracies
    train_error, train_accuracy, val_error, val_accuracy = [], [], [], []

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        train_loss, train_count = 0, 0

        # Iterate through each batch
        for i in range(len(x_batch)):
            # Perform forward propagation to compute pre-activation and activation values
            a, h = forward_propagation(x_batch[i], weight, bias, n_layers, activation_function, output_function)
            y_pred = h[-1]

            # Perform backward propagation to compute derivatives w.r.t. weights and biases
            dw, db = backward_propagation(x_batch[i], y_pred, y_batch[i], weight, a, h, n_layers, activation_function)

            # Update weights and biases using the stochastic gradient descent update rule
            for j in range(len(weight)):
                weight[j] -= lr * dw[j + 1]
                bias[j] -= lr * db[j + 1]

        # Calculate regularized loss and accuracy on the training set
        a, h = forward_propagation(x_train, weight, bias, n_layers, activation_function, output_function)
        y_pred = h[-1]
        train_loss = loss(y_pred, y_train, loss_function, weight, alpha)
        train_count = accuracy(y_pred, y_train)
        train_error.append(train_loss)
        train_accuracy.append(train_count / len(x_train))

        # Calculate regularized loss and accuracy on the validation set
        a, h = forward_propagation(val_x, weight, bias, n_layers, activation_function, output_function)
        y_valpred = h[-1]
        val_loss = loss(y_valpred, val_y, loss_function, weight, alpha)
        val_count = accuracy(y_valpred, val_y)
        val_error.append(val_loss)
        val_accuracy.append(val_count / len(val_y))

        # Log values to wandb
        wandb.log({"train_accuracy": (train_count / len(x_train)), "train_error": train_loss, "val_accuracy": (val_count / len(val_y)), "val_error": val_loss})

    return weight, bias, train_error, train_accuracy, val_error, val_accuracy


# Define the momentum gradient descent function to train the model with a batch size of 1 for a given network configuration
def momentum_gd(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha):
    # Initialize the network layers
    n_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

    # Initialize weights and biases based on the specified weight initialization method
    weight, bias = start_weights_and_bias(n_layers, weight_init)

    # Split the training data into the specified number of batches
    x_batch = np.array(np.array_split(x_train, batches))
    y_batch = np.array(np.array_split(y_train, batches))

    # Initialize lists to store training and validation errors and accuracies
    train_error, train_accuracy, val_error, val_accuracy = [], [], [], []

    # Initialize dictionaries to store the history of weight and bias updates
    history_weight = {}
    history_bias = {}

    # Initialize the history for each round of updates for weights and biases
    for i in range(len(n_layers)):
        history_weight[i + 1] = np.zeros(weight[i].shape)
        history_bias[(i + 1)] = np.zeros(bias[i].shape)

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        train_loss, train_count = 0, 0

        # Iterate through each batch
        for i in range(len(x_batch)):
            # Perform forward propagation to compute pre-activation and activation values
            a, h = forward_propagation(x_batch[i], weight, bias, n_layers, activation_function, output_function)
            y_pred = h[-1]

            # Perform backward propagation to compute derivatives w.r.t. weights and biases
            dw, db = backward_propagation(x_batch[i], y_pred, y_batch[i], weight, a, h, n_layers, activation_function)

            # Update weights and biases using momentum gradient descent update rule
            for j in range(len(n_layers)):
                history_weight[j + 1] = m * history_weight[j + 1] + lr * dw[j + 1]
                history_bias[j + 1] = m * history_bias[j + 1] + lr * db[j + 1]

                weight[j] -= history_weight[j + 1]
                bias[j] -= history_bias[j + 1]

        # Calculate regularized loss and accuracy on the training set
        a, h = forward_propagation(x_train, weight, bias, n_layers, activation_function, output_function)
        y_pred = h[-1]
        train_loss = loss(y_pred, y_train, loss_function, weight, alpha)
        train_count = accuracy(y_pred, y_train)
        train_error.append(train_loss)
        train_accuracy.append(train_count / len(x_train))

        # Calculate regularized loss and accuracy on the validation set
        a, h = forward_propagation(val_x, weight, bias, n_layers, activation_function, output_function)
        y_valpred = h[-1]
        val_loss = loss(y_valpred, val_y, loss_function, weight, alpha)
        val_count = accuracy(y_valpred, val_y)
        val_error.append(val_loss)
        val_accuracy.append(val_count / len(val_y))

        # Log values to wandb
        wandb.log({"train_accuracy": (train_count / len(x_train)), "train_error": train_loss, "val_accuracy": (val_count / len(val_y)), "val_error": val_loss})

    return weight, bias, train_error, train_accuracy, val_error, val_accuracy

# Define the Nesterov accelerated gradient descent function to train the model with a batch size of 1 for a given network configuration
def nesterov_gd(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha):
    # Initialize the network layers
    n_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

    # Initialize weights and biases using the specified weight initialization method
    weight, bias = start_weights_and_bias(n_layers, weight_init)

    # Split the training data into the specified number of batches
    x_batch = np.array(np.array_split(x_train, batches))
    y_batch = np.array(np.array_split(y_train, batches))

    # Initialize lists to store training and validation errors and accuracies
    train_error, train_accuracy, val_error, val_accuracy = [], [], [], []

    # Initialize dictionaries to store the history of weight and bias updates
    history_weight = {}
    history_bias = {}

    # Initialize the history for each round of updates for weights and biases
    for i in range(len(n_layers)):
        history_weight[i + 1] = np.zeros(weight[i].shape)
        history_bias[i + 1] = np.zeros(bias[i].shape)

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        train_loss, train_count = 0, 0

        # Iterate through each batch
        for i in range(len(x_batch)):
            # Initialize lookahead weights and biases for the next round
            lookahead_weight, lookahead_bias = [], []
            for j in range(len(n_layers)):
                lookahead_weight.append(weight[j] - m * history_weight[j + 1])
                lookahead_bias.append(bias[j] - m * history_bias[j + 1])

            # Perform forward propagation to compute pre-activation and activation values using lookahead weights and biases
            a, h = forward_propagation(x_batch[i], lookahead_weight, lookahead_bias, n_layers, activation_function, output_function)
            y_pred = h[-1]

            # Perform backward propagation to compute derivatives w.r.t. weights and biases
            dw, db = backward_propagation(x_batch[i], y_pred, y_batch[i], lookahead_weight, a, h, n_layers, activation_function)

            # Update weights and biases using Nesterov accelerated gradient descent update rule
            for j in range(len(n_layers)):
                history_weight[j + 1] = m * history_weight[j + 1] + lr * dw[j + 1]
                history_bias[j + 1] = m * history_bias[j + 1] + lr * db[j + 1]

                weight[j] -= history_weight[j + 1]
                bias[j] -= history_bias[j + 1]

        # Calculate regularized loss and accuracy on the training set
        a, h = forward_propagation(x_train, weight, bias, n_layers, activation_function, output_function)
        y_pred = h[-1]
        train_loss = loss(y_pred, y_train, loss_function, weight, alpha)
        train_count = accuracy(y_pred, y_train)
        train_error.append(train_loss)
        train_accuracy.append(train_count / len(x_train))

        # Calculate regularized loss and accuracy on the validation set
        a, h = forward_propagation(val_x, weight, bias, n_layers, activation_function, output_function)
        y_valpred = h[-1]
        val_loss = loss(y_valpred, val_y, loss_function, weight, alpha)
        val_count = accuracy(y_valpred, val_y)
        val_error.append(val_loss)
        val_accuracy.append(val_count / len(val_y))

        # Log values to wandb
        wandb.log({"train_accuracy": (train_count / len(x_train)), "train_error": train_loss, "val_accuracy": (val_count / len(val_y)), "val_error": val_loss})

    return weight, bias, train_error, train_accuracy, val_error, val_accuracy

# Define the RMSprop gradient descent function to train the model for a batch size of 1 for a particular network configuration
def rms_prop(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha):
    # Initialize the network layers
    n_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

    # Initialize weights and biases using the specified weight initialization method
    weight, bias = start_weights_and_bias(n_layers, weight_init)

    # Split the training data into the specified number of batches
    x_batch = np.array(np.array_split(x_train, batches))
    y_batch = np.array(np.array_split(y_train, batches))

    # Initialize lists to store training and validation errors and accuracies
    train_error, train_accuracy, val_error, val_accuracy = [], [], [], []

    # Initialize dictionaries to store the history of weight and bias updates
    history_weight = {}
    history_bias = {}

    # Initialize the history for each round of updates for weights and biases
    for i in range(len(n_layers)):
        history_weight[i + 1] = np.zeros(weight[i].shape)
        history_bias[(i + 1)] = np.zeros(bias[i].shape)

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        train_loss, train_count = 0, 0

        for i in range(len(x_batch)):
            # Perform forward propagation to compute pre-activation and activation values at each layer
            a, h = forward_propogation(x_batch[i], weight, bias, n_layers, activation_function, output_function)
            y_pred = h[-1]

            # Perform backward propagation to compute derivatives w.r.t. weights and biases
            dw, db = backward_propagation(x_batch[i], y_pred, y_batch[i], weight, a, h, n_layers, activation_function)

            # Update weights and biases using RMSprop update rule
            for j in range(len(n_layers)):
                history_weight[j + 1] = beta * history_weight[j + 1] + (1 - beta) * (dw[j + 1]) ** 2
                history_bias[j + 1] = beta * history_bias[j + 1] + (1 - beta) * (db[j + 1]) ** 2

                weight[j] -= lr * np.divide(dw[j + 1], np.sqrt(history_weight[j + 1] + epsilon))
                bias[j] -= lr * np.divide(db[j + 1], np.sqrt(history_bias[j + 1] + epsilon))

        # Calculate regularized loss and accuracy on the training set
        a, h = forward_propogation(x_train, weight, bias, n_layers, activation_function, output_function)
        y_pred = h[-1]
        train_loss = loss(y_pred, y_train, loss_function, weight, alpha)
        train_count = accuracy(y_pred, y_train)
        train_error.append(train_loss)
        train_accuracy.append(train_count / len(x_train))

        # Calculate regularized loss and accuracy on the validation set
        a, h = forward_propogation(val_x, weight, bias, n_layers, activation_function, output_function)
        y_valpred = h[-1]
        val_loss = loss(y_valpred, val_y, loss_function, weight, alpha)
        val_count = accuracy(y_valpred, val_y)
        val_error.append(val_loss)
        val_accuracy.append(val_count / len(val_y))

        # Log values to wandb
        wandb.log({"train_accuracy": (train_count / len(x_train)), "train_error": train_loss, "val_accuracy": (val_count / len(val_y)), "val_error": val_loss})

    return weight, bias, train_error, train_accuracy, val_error, val_accuracy


# Define the Adam gradient descent function to train the model for a batch size of 1 with a specific network configuration
def adam_gradient_descent(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha):
    # Initialize the network layers
    n_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

    # Initialize weights and biases using the specified weight initialization method
    weight, bias = start_weights_and_bias(n_layers, weight_init)

    # Split the training data into the specified number of batches
    x_batch = np.array(np.array_split(x_train, batches))
    y_batch = np.array(np.array_split(y_train, batches))

    # Initialize lists to store training and validation errors and accuracies
    train_error, train_accuracy, val_error, val_accuracy = [], [], [], []

    # Initialize dictionaries to store the history of weight and bias updates
    v_weight = {}
    v_bias = {}
    m_weight = {}
    m_bias = {}
    v_hatw = {}
    v_hatb = {}
    m_hatw = {}
    m_hatb = {}

    # Initialize the history for each round of updates for weights and biases
    for i in range(len(n_layers)):
        v_weight[i + 1] = np.zeros(weight[i].shape)
        v_bias[(i + 1)] = np.zeros(bias[i].shape)
        m_weight[i + 1] = np.zeros(weight[i].shape)
        m_bias[(i + 1)] = np.zeros(bias[i].shape)

    t = 0

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        train_loss, train_count = 0, 0

        for i in range(len(x_batch)):
            t += 1

            # Perform forward propagation to compute pre-activation and activation values at each layer
            a, h = forward_propogation(x_batch[i], weight, bias, n_layers, activation_function, output_function)
            y_pred = h[-1]

            # Perform backward propagation to compute derivatives w.r.t. weights and biases
            dw, db = backward_propagation(x_batch[i], y_pred, y_batch[i], weight, a, h, n_layers, activation_function)

            # Update weights and biases using Adam update rule
            for j in range(len(n_layers)):
                v_weight[j + 1] = beta2 * v_weight[j + 1] + (1 - beta2) * (dw[j + 1]) ** 2
                v_bias[j + 1] = beta2 * v_bias[j + 1] + (1 - beta2) * (db[j + 1]) ** 2
                m_weight[j + 1] = beta1 * m_weight[j + 1] + (1 - beta1) * dw[j + 1]
                m_bias[j + 1] = beta1 * m_bias[j + 1] + (1 - beta1) * db[j + 1]
                v_hatw[j + 1] = np.divide(v_weight[j + 1], (1 - beta2 ** t))
                v_hatb[j + 1] = np.divide(v_bias[j + 1], (1 - beta2 ** t))
                m_hatw[j + 1] = np.divide(m_weight[j + 1], (1 - beta1 ** t))
                m_hatb[j + 1] = np.divide(m_bias[j + 1], (1 - beta1 ** t))
                weight[j] -= lr * np.divide(m_hatw[j + 1], np.sqrt(v_hatw[j + 1] + epsilon))
                bias[j] -= lr * np.divide(m_hatb[j + 1], np.sqrt(v_hatb[j + 1] + epsilon))

        # Calculate regularized loss and accuracy on the training set
        a, h = forward_propogation(x_train, weight, bias, n_layers, activation_function, output_function)
        y_pred = h[-1]
        train_loss = loss(y_pred, y_train, loss_function, weight, alpha)
        train_count = accuracy(y_pred, y_train)
        train_error.append(train_loss)
        train_accuracy.append(train_count / len(x_train))

        # Calculate regularized loss and accuracy on the validation set
        a, h = forward_propogation(val_x, weight, bias, n_layers, activation_function, output_function)
        y_valpred = h[-1]
        val_loss = loss(y_valpred, val_y, loss_function, weight, alpha)
        val_count = accuracy(y_valpred, val_y)
        val_error.append(val_loss)
        val_accuracy.append(val_count / len(val_y))

        # Log values to wandb


    # logging values into wandb
    wandb.log({"train_accuracy":(counttrain/len(x_train)),"train_error":l,"val_accuracy": (countval/len(val_y)),"val_error":l_val})

  return weight,bias,train_error,train_accuracy,val_error,val_accuracy


# nadam_gradient_descent is used to train the model for a batch_size of 1 for a particular network configuration
def nadam_gradient_descent(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha):
    # Initialize the network layers
    n_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

    # Initialize weights and biases using the specified weight initialization method
    weight, bias = start_weights_and_bias(n_layers, weight_init)

    # Split the training data into the specified number of batches
    x_batch = np.array(np.array_split(x_train, batches))
    y_batch = np.array(np.array_split(y_train, batches))

    # Initialize lists to store training and validation errors and accuracies
    train_error, train_accuracy, val_error, val_accuracy = [], [], [], []

    # Initialize dictionaries to store the history of weight and bias updates
    v_weight = {}
    v_bias = {}
    m_weight = {}
    m_bias = {}

    v_hatw = {}
    v_hatb = {}
    m_hatw = {}
    m_hatb = {}

    # Initialize momentum and history terms for weights and biases
    for i in range(len(n_layers)):
        v_weight[i + 1] = np.zeros(weight[i].shape)
        v_bias[i + 1] = np.zeros(bias[i].shape)
        m_weight[i + 1] = np.zeros(weight[i].shape)
        m_bias[i + 1] = np.zeros(bias[i].shape)

    t = 0

    # Train the model for the specified number of epochs
    for epoch in range(epochs):
        train_loss, train_count = 0, 0

        for i in range(len(x_batch)):
            # Update the iteration count
            t += 1

            # Initialize lookahead terms for weights and biases
            lookahead_w = []
            lookahead_b = []
            lookahead_mhatw = []
            lookahead_mhatb = []
            lookahead_vhatw = []
            lookahead_vhatb = []

            for j in range(len(n_layers)):
                lookahead_vhatw.append(np.divide(beta2 * v_weight[j + 1], (1 - beta2 ** t)))
                lookahead_vhatb.append(np.divide(beta2 * v_bias[j + 1], (1 - beta2 ** t)))

                lookahead_mhatw.append(np.divide(beta1 * m_weight[j + 1], (1 - beta1 ** t)))
                lookahead_mhatb.append(np.divide(beta1 * m_bias[j + 1], (1 - beta1 ** t)))

                lookahead_w.append(weight[j] - lr * np.divide(lookahead_mhatw[j], np.sqrt(lookahead_vhatw[j] + epsilon)))
                lookahead_b.append(bias[j] - lr * np.divide(lookahead_mhatb[j], np.sqrt(lookahead_vhatb[j] + epsilon)))

            # Perform forward propagation to compute pre-activation and activation values
            a, h = forward_propogation(x_batch[i], lookahead_w, lookahead_b, n_layers, activation_function, output_function)
            y_pred = h[-1]

            # Perform backward propagation to compute derivatives w.r.t. weights and biases
            dw, db = backward_propagation(x_batch[i], y_pred, y_batch[i], lookahead_w, a, h, n_layers, activation_function)

            # Update rules for NAdam
            for j in range(len(n_layers)):
                v_weight[j + 1] = beta2 * v_weight[j + 1] + (1 - beta2) * (dw[j + 1]) ** 2
                v_bias[j + 1] = beta2 * v_bias[j + 1] + (1 - beta2) * (db[j + 1]) ** 2

                m_weight[j + 1] = beta1 * m_weight[j + 1] + (1 - beta1) * dw[j + 1]
                m_bias[j + 1] = beta1 * m_bias[j + 1] + (1 - beta1) * db[j + 1]

                v_hatw[j + 1] = np.divide(v_weight[j + 1], (1 - beta2 ** t))
                v_hatb[j + 1] = np.divide(v_bias[j + 1], (1 - beta2 ** t))

                m_hatw[j + 1] = np.divide(m_weight[j + 1], (1 - beta1 ** t))
                m_hatb[j + 1] = np.divide(m_bias[j + 1], (1 - beta1 ** t))

                weight[j] -= lr * np.divide(m_hatw[j + 1], np.sqrt(v_hatw[j + 1] + epsilon))
                bias[j] -= lr * np.divide(m_hatb[j + 1], np.sqrt(v_hatb[j + 1] + epsilon)))

        # Calculate regularized loss and accuracy on the training set
        a, h = forward_propogation(x_train, weight, bias, n_layers, activation_function, output_function)
        y_pred = h[-1]
        train_loss = loss(y_pred, y_train, loss_function, weight, alpha)
        train_count = accuracy(y_pred, y_train)
        train_error.append(train_loss)
        train_accuracy.append(train_count / len(x_train))

        # Calculate regularized loss and accuracy on the validation set
        a, h = forward_propogation(val_x, weight, bias, n_layers, activation_function, output_function)
        y_valpred = h[-1]
        val_loss = loss(y_valpred, val_y, loss_function, weight, alpha)
        val_count = accuracy(y_valpred, val_y)
        val_error.append(val_loss)
        val_accuracy.append(val_count / len(val_y))

        # Log values to wandb
        wandb.log({"train_accuracy": (train_count / len(x_train)), "train_error": train_loss, "val_accuracy": (val_count / len(val_y)), "val_error": val_loss})

    return weight, bias, train_error, train_accuracy


# the train function contains calls to various optimization function based on the input given
def train_model(x_train, y_train, batch_size, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, optimizer, loss_function, alpha):
  
  # Calculate the number of batches
  batches = math.ceil(len(x_train) / batch_size)

  if optimizer == "sgd":
    weight, bias, train_error, train_accuracy, val_error, val_accuracy = stochastic_gradient_descent(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha)
  
  elif optimizer == "momentum":
    weight, bias, train_error, train_accuracy, val_error, val_accuracy = momentum_gd(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha)
  
  elif optimizer == "nag":
    weight, bias, train_error, train_accuracy, val_error, val_accuracy = nesterov_gd(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha)
  
  elif optimizer == "rmsprop":
    weight, bias, train_error, train_accuracy, val_error, val_accuracy = rms_prop(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha)
  
  elif optimizer == "adam":
    weight, bias, train_error, train_accuracy, val_error, val_accuracy = adam(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha)
  
  elif optimizer == "nadam":
    weight, bias, train_error, train_accuracy, val_error, val_accuracy = nadam(x_train, y_train, batches, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, loss_function, alpha)
    
  return weight, bias, train_error, train_accuracy, val_error, val_accuracy


# wandb config used to execute a sweep in wandb
# lr = config.lr
# epochs = config.epochs
# batch_size = config.batch_size
# optimizer = config.optimizer
# weight_init = config.weight_init
# hidden_layer_size=config.hidden_layer_size
# activation_function=config.activation_function
# hidden_layer=config.hidden_layer
# alpha=config.alpha
# run.name='hl_'+str(hidden_layer)+'_bs_'+str(batch_size)+'_ac_'+activation_function

# call to train function
# Train the model
trained_weight, trained_bias, train_error, train_accuracy, val_error, val_accuracy = train_model(x_train, y_train, batch_size, hidden_layer, hidden_layer_size, lr, weight_init, epochs, activation_function, output_function, optimizer, loss_function, alpha)

# Initialize network layers
network_layers = layer_init(dim2, output_size, hidden_layer_size, hidden_layer, activation_function, output_function)

# Print the initialized layers
for layer in network_layers:
    print(layer)

# Perform forward propagation on test data
activations, outputs = forward_propagation(x_test, trained_weight, trained_bias, network_layers, activation_function, output_function)
predicted_labels = outputs[-1]

# Convert predicted labels to their corresponding classes
predicted_classes = []
for pred_label in predicted_labels:
    predicted_classes.append(np.argmax(pred_label))

# Calculate test loss and accuracy
test_loss = loss(predicted_labels, y_test, loss_function, trained_weight, alpha)
test_accuracy = accuracy(predicted_labels, y_test)

# Print test accuracy and loss
print("Test accuracy after training the model:", test_accuracy / len(y_test))
print("Test loss after training the model:", test_loss)

# the following the the code for making the confusion matrix on test data as shown in the report

# cm=wandb.plot.confusion_matrix(
#   y_true=Y_Test,
#   preds=y,
#   class_names= classes
# )
# print('Test Confusion Matrix\n')
# wandb.log({"conf_mat": cm})
