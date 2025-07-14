"""
5.1.2 Stochastic gradient descent (SGD)
"""
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
      
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 

"""
5.1.4 Momentum SGD
"""
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]

"""
5.1.5 AdaGrad
"""
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

"""
5.1.6 Adam (http://arxiv.org/abs/1412.6980v8)
"""
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

"""
5.1.7 Which update method should we use?
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

# Objective function to minimize (a simple bowl-shaped function)
def f(x, y):
    return x**2 / 20.0 + y**2

# Gradient of the function f(x, y)
def df(x, y):
    return x / 10.0, 2.0 * y

# Initial starting point (x, y)
init_pos = (-7.0, 2.0)

# Parameters and gradients dictionaries
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

# Create different optimizers with specific learning rates
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)             # Stochastic Gradient Descent
optimizers["Momentum"] = Momentum(lr=0.1)     # Momentum-based update
optimizers["AdaGrad"] = AdaGrad(lr=1.5)       # Adaptive Gradient
optimizers["Adam"] = Adam(lr=0.3)             # Adaptive Moment Estimation

# Set up the plot window
plt.figure(figsize=(12, 8))

idx = 1  # Subplot index

# Loop through each optimizer and simulate parameter updates
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []  # To store x positions during optimization
    y_history = []  # To store y positions during optimization

    # Reset parameters to initial position
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    # Perform 30 steps of optimization
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        # Calculate gradient at current position
        grads['x'], grads['y'] = df(params['x'], params['y'])

        # Update parameters using the selected optimizer
        optimizer.update(params, grads)

    # Create a grid to visualize the contour of the function
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Limit the height of the contour for clearer visualization
    mask = Z > 7
    Z[mask] = 0

    # Draw the optimization path and function contour
    plt.subplot(2, 2, idx)  # 2x2 grid layout
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")  # Path taken by optimizer
    plt.contour(X, Y, Z)  # Contour of the function
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')  # Mark the minimum point at the origin
    plt.title(key)  # Title with optimizer name
    plt.xlabel("x")
    plt.ylabel("y")

# Adjust layout and show the plot
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()

"""
5.1.8 Comparison of optimization methods using the MNIST dataset
"""
import sys, os
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *

# Load MNIST data (with normalization to scale input values between 0 and 1)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]  # Number of training samples
batch_size = 128               # Number of samples per mini-batch
max_iterations = 2000          # Total number of training iterations

# Set up different optimization algorithms for comparison
optimizers = {}
optimizers['SGD'] = SGD()             # Stochastic Gradient Descent
optimizers['Momentum'] = Momentum()   # SGD with momentum
optimizers['AdaGrad'] = AdaGrad()     # Adaptive Gradient
optimizers['Adam'] = Adam()           # Adaptive Moment Estimation
# optimizers['RMSprop'] = RMSprop()   # (Optional) RMSprop optimizer

# Initialize a neural network and loss tracker for each optimizer
networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784,                 # Input dimension (28x28 images)
        hidden_size_list=[100, 100, 100, 100],  # 4 hidden layers with 100 neurons each
        output_size=10)                # Output layer (10 classes for digits 0–9)
    train_loss[key] = []               # To store training loss for each optimizer

# Start training loop
for i in range(max_iterations):
    # Randomly select a batch of training data
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # Perform training step for each optimizer
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)           # Compute gradients
        optimizers[key].update(networks[key].params, grads)        # Update parameters
        
        loss = networks[key].loss(x_batch, t_batch)                # Compute training loss
        train_loss[key].append(loss)                               # Record the loss
    
    # Print loss every 100 iterations
    if i % 100 == 0:
        print("===========" + " iteration: " + str(i) + " ===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ": " + str(loss))

# Plot the training loss curves for each optimizer
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key],
             markevery=100, label=key)  # markevery: show marker every 100 steps

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

"""
5.2.2 Distribution of activation values in hidden layers
"""
import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Generate input data: 1000 samples, each with 100 features
input_data = np.random.randn(1000, 100)

node_num = 100             # Number of neurons in each hidden layer
hidden_layer_size = 5      # Total number of hidden layers
activations = {}           # Dictionary to store activation outputs

x = input_data

# Forward pass through each hidden layer
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # Experiment with different weight initialization strategies
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)

    # Experiment with different activation functions
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z  # Store the activation output

# Plot histograms of activation distributions for each layer
for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    if i != 0:
        plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()

"""
5.2.3 Weight initialization when using ReLU
"""
import numpy as np
import matplotlib.pyplot as plt

# Activation function: ReLU
def ReLU(x):
    return np.maximum(0, x)

# Experiment settings
input_data = np.random.randn(1000, 100)  # 1000 samples, 100 features
node_num = 100                           # Number of neurons per layer
hidden_layer_size = 5                    # Number of hidden layers
activation = ReLU                        # Activation function

# Different weight initialization methods
def init_weight(method, node_num):
    if method == "std=0.01":
        return np.random.randn(node_num, node_num) * 0.01
    elif method == "Xavier":
        return np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    elif method == "He":
        return np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
    else:
        raise ValueError("Unknown method")

methods = ["std=0.01", "Xavier", "He"]

for method in methods:
    print(f"Weight Init Method: {method}")
    x = input_data
    activations = {}

    for i in range(hidden_layer_size):
        w = init_weight(method, node_num)
        a = np.dot(x, w)
        z = activation(a)
        activations[i] = z
        x = z  # pass to next layer

    # Plot histograms
    plt.figure(figsize=(10, 2))
    for i, z in activations.items():
        plt.subplot(1, hidden_layer_size, i + 1)
        plt.title(f"{i+1}-layer")
        plt.hist(z.flatten(), bins=30, range=(0, 1))
        plt.yticks([])
        if i != 0:
            plt.xticks([])
    plt.suptitle(f"{method} initialization")
    plt.tight_layout()
    plt.show()

"""
5.2.4 Comparison of Weight Initialization Methods on the MNIST Dataset
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

# Load the MNIST dataset (normalized)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]  # Total number of training samples
batch_size = 128               # Mini-batch size
max_iterations = 2000          # Total training iterations

# Experiment settings
# Test three different weight initialization methods
weight_init_types = {
    'std=0.01': 0.01,          # Small standard deviation
    'Xavier': 'sigmoid',       # Xavier initialization (recommended for sigmoid/tanh)
    'He': 'relu'               # He initialization (recommended for ReLU)
}

optimizer = SGD(lr=0.01)       # Use SGD optimizer for all experiments

networks = {}      # Store networks for each init method
train_loss = {}    # Store training losses

# Initialize a neural network for each weight initialization method
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],  # 4 hidden layers with 100 neurons each
        output_size=10,
        weight_init_std=weight_type
    )
    train_loss[key] = []

# Start training
for i in range(max_iterations):
    # Randomly select a mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # Train each network with the same mini-batch
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)              # Compute gradients
        optimizer.update(networks[key].params, grads)                 # Update weights
        loss = networks[key].loss(x_batch, t_batch)                   # Compute loss
        train_loss[key].append(loss)                                  # Record loss

    # Print losses every 100 iterations
    if i % 100 == 0:
        print("===========" + "iteration: " + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ": " + str(loss))

# Plot training loss curves
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)

for key in weight_init_types.keys():
    plt.plot(
        x, 
        smooth_curve(train_loss[key]), 
        marker=markers[key], 
        markevery=100, 
        label=key
    )

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()

"""
5.3.2 Effect of batch normalization
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce the training data
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
    # Network with Batch Normalization
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                     weight_init_std=weight_init_std, use_batchnorm=True)
    
    # Network without Batch Normalization
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                  weight_init_std=weight_init_std)
    
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list

# Plotting the graph
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

train_acc_list, bn_train_acc_list = __train(weight_scale_list[4])
    
plt.title("Training Accuracy")
plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
plt.plot(x, train_acc_list, linestyle = "--", label='Normal (without BatchNorm)', markevery=2)
plt.ylim(0, 1.0)
plt.xlim(0, max_epochs)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(loc='lower right')
    
plt.show()

"""
5.4.1 overfitting
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce the number of training samples to reproduce overfitting
x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0.1  # Weight decay setting

network = MultiLayerNet(input_size=784,
                        hidden_size_list=[100, 100, 100, 100, 100, 100],
                        output_size=10,
                        weight_decay_lambda=weight_decay_lambda)

optimizer = SGD(lr=0.01)  # Use SGD with learning rate 0.01

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) +
              ", train acc:" + str(train_acc) +
              ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# Plotting the graph
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""
5.4.2 Weight decay
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce the training data to reproduce overfitting
x_train = x_train[:300]
t_train = t_train[:300]

weight_decay_lambda = 0.1  # Set weight decay

network = MultiLayerNet(input_size=784,
                        hidden_size_list=[100, 100, 100, 100, 100, 100],
                        output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)  # Use SGD with learning rate of 0.01 to update parameters

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) +
              ", train acc:" + str(train_acc) +
              ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

# Plot the graph
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""
5.4.3 Dropout
"""
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce the number of training samples to reproduce overfitting
x_train = x_train[:300]
t_train = t_train[:300]

# Set whether to use Dropout and the dropout ratio
use_dropout = True  # Set to False if you don't want to use dropout
dropout_ratio = 0.2

network = MultiLayerNetExtend(input_size=784,
                              hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10,
                              use_dropout=use_dropout,
                              dropout_ratio=dropout_ratio)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01},
                  verbose=True)

trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

# Plotting the graph
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

"""
5.5.1 Validation data
"""
(x_train, t_train), (x_test, t_test) = load_mnist()

# Shuffle the training data
x_train, t_train = shuffle_dataset(x_train, t_train)

# Split approximately 20% as validation data
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

"""
5.5.2 Optimize hyperparameter
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# Reduce training data size for quicker results
x_train = x_train[:500]
t_train = t_train[:500]

# Split 20% as validation data
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list

# Random search for hyperparameter optimization
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # Define the range for hyperparameter search
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# Plotting the results
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

plt.figure(figsize=(10, 8))

for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.subplots_adjust(wspace=0.2, hspace=0.3)    
plt.show()
