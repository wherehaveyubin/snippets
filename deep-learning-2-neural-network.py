import numpy as np
import matplotlib.pylab as plt

"""
2.2.1 Step functions
"""
# Step function (1)
# x must be a real number (float)
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0

# Step function (2)
# supports NumPy arrays
def step_function(x):
  y = x > 0
  return y.astype(np.int)

x = np.array([-1.0, 1.0, 2.0])
y = x > 0
y # array([False,  True,  True])
y = y.astype(int)
y # array([0, 1, 1])

# Step function graph
def step_function(x):
    return np.array(x > 0, dtype=int)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1) # Set y-axis range
plt.show()

"""
Sigmoid function
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x) # array([0.26894142, 0.73105858, 0.88079708])

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(X)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# Compare functions
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1)
plt.show()

"""
ReLU function
"""
def relu(x):
    return np.maximum(0, x) # Return the greater of the two

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()

      
"""
Multi-dimension
"""
# Understanding matrix
A = np.array([1, 2, 3, 4])
print(A)
# [1 2 3 4]
np.ndim(A) # Dimension 
# 1
A.shape # Returns a tuple 
# (4,) 
A.shape[0] 
# 4

B = np.array([[1,2], [3,4],[5,6]])
print(B)
# [[1 2]
# [3 4]
# [5 6]]
np.ndim(B)
# 2
B.shape
# (3, 2)      

# Multiply matrix
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
np.dot(A, B)

A = np.array([[1,2,3], [4,5,6]])
B = np.array([[1,2], [3,4], [5,6]])
np.dot(A, B)

A = np.array([[1,2], [3,4], [5,6]])
B = np.array([7,8])
np.dot(A, B)

# Multyply neural network       
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])
Y = np.dot(X, W)
print(Y) # [ 5 11 17]

      
"""
Three-layer neural network
"""
# Zero layer to First layer
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)

# First layer to Second layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# Second layer to Third layer
def identity_function(x):
  return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # or Y = A3

# Sum up
def init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network

def identity_function(x):
  return x

def forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = identity_function(a3)

  return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708 0.69627909]

    
"""
Softmax function
"""
# Basic softmax function
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

# Softmax function with improved overflow handling
def softmax(a):
  c = np.max(a)
  exp_a = np.exp(a - c) # Overflow prevention
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y
