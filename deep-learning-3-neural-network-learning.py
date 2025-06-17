import numpy as np
import matplotlib.pylab as plt

## Loss function
# Sum of squares for error, SSE
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]) # Probability for each element being 0, 1, ..., 9
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # Correct answer is the number 2

def sum_squared_error(y, t):
	return 0.5 * np.sum((y - t) ** 2)

# If the probability of number 2 is estimated to be the highest
sum_squared_error(np.array(y), np.array(t)) # 0.0975

# If the probability of number 7 is estimated to be the highest
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
sum_squared_error(np.array(y2), np.array(t)) # 0.5975

# Cross entropy error, CEE
"""
If 0 is passed to np.log(), it becomes negative infinity (-inf),
which prevents further computation, so a very small delta value is added to avoid this.
"""
def cross_entropy_error(y, t):
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))
  
# If the probability of number 2 is estimated to be the highest
print(cross_entropy_error(y, t)) # 0.510825457099338

# If the probability of number 7 is estimated to be the highest
y1 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(cross_entropy_error(y1, t)) # 2.302584092994546


## Mini-batch
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# Randomly select 10 samples
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


## Cross entropy error for minibatch
# When the true labels are one-hot encoded
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        """
        # print(t) # [7 2 1 ... 4 5 6]
        # print(t.reshape(1, t.size)) # [[7 2 1 ... 4 5 6]]
        """
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# When the true labels are numerical values like 2, 7
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


## Numerical differentiation
# Bad example
def numerical_diff(f, x):
    h = 1e-50  # Very small value for h
    return (f(x + h) - f(x)) / h
"""
1. Rounding error problem.
   By assigning h as 1e-50 (a number with 50 zeros after the decimal point),
   rounding errors occur. Small values (like those with less than 8 decimal places)
   are omitted, causing errors in the final calculation.
   e.g. np.float32(1e-50) # 0.0

2. Issue with the difference of the function (difference between values of the function at two arbitrary points).
   This code calculates the slope (the true derivative or the true tangent) at x
   but instead calculates the slope between x+h and x (numerical derivative, an approximation).
"""

# Good example
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

# Visualize numerical differentiation 
def function_1(x):
	return 0.01*x**2 + 0.1*x

# Create an array from 0 to 20 with a step of 0.1 (20 is not included)
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
ply.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# Numerical derivative at x = 5 and x = 10
numerical_diff(function_1, 5) # 0.1999999999990898
numerical_diff(function_1, 10) # 0.2999999999986347


## Partial differentiation
def function_2(x):
	return x[0]**2 + x[1]**2
    # or return np.sum(x**2)

# Partial derivative with respect to x_0 at x_0 = 3
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0
numerical_diff(function_tmp1, 3.0)  # 6.00000000000378

# Partial derivative with respect to x_1 at x_1 = 4
def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1
numerical_diff(function_tmp2, 4.0)  # 7.999999999999119


## Gradient
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # Create an array of the same shape as x, with all elements initialized to 0
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # Calculate f(x + h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # Calculate f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        # Compute the gradient (numerical derivative) at x[idx]
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        
        # Restore the original value of x[idx]
        x[idx] = tmp_val
    
    return grad

# Example usage with function_2 and different values of x
numerical_gradient(function_2, np.array([3.0, 4.0]))  # array([6., 8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))  # array([0., 4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))  # array([6., 0.])
