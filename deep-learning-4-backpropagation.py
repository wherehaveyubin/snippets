"""
4.4.1 Multiplication layer
"""
# Multiply propagation
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy

# Example 
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# Forward
apple_price = mul_apple_layer.forward(apple, apple_num) # 200
price = mul_tax_layer.forward(apple_price, tax) # 220.00000000000003

# Backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice) # (1.1, 200)
dapple, dapple_num = mul_apple_layer.backward(dapple_price) # (2.2, 110.00000000000001)

print("price:", int(price)) # 220
print("dApple:", dapple) # 2.2
print("dApple_num:", int(dapple_num)) # 110
print("dTax:", dtax) # 200


"""
4.4.2 Addition layer
"""
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

# Example
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# Forward
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)

# Backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print("price:", int(price)) # 715
print("dApple:", dapple) # 2.2
print("dApple_num:", int(dapple_num)) # 110
print("dOrange:", dorange) # 3.3000000000000003
print("dOrange_num:", int(dorange_num)) # 165
print("dTax:", dtax) # 650


"""
4.5.1 ReLU layer
"""
# Mask 
x = np.array([[1.0, -0.5], [-2.0, 3.0]])
'''
[[ 1.  -0.5]
 [-2.   3. ]]
'''
mask = (x <= 0)
print(mask)
'''
[[False  True]
 [ True False]]
'''


"""
4.5.2 Sigmoid layer
"""
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out  # Store the output of the forward pass for use in the backward pass
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out  # Derivative of sigmoid: dout * y * (1 - y)
        return dx


"""
4.6.1 Affine layer
"""
# Example
X = np.random.rand(2) # 입력 array([0.71101479, 0.87477799])
W = np.random.rand(2,3) # 가중치 array([[0.13691532, 0.86736043, 0.83357747],
                        #              [0.79764694, 0.95760272, 0.53444365]])
B = np.random.rand(3) # 편향 array([0.61461436, 0.31099111, 0.17659224])

X.shape # (2,)
W.shape # (2, 3)
B.shape # (3,)
