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
