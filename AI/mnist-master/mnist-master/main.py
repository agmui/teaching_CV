import math

import numpy as np
import mnist
from PIL import Image

input_size: int = 28 ** 2
layer_size: int = 16

images = mnist.train_images()
np_image = np.array(images[0])

# expand img to a vector and scale down between 0 and 1
np_vector = np_image.flatten() / 255

a = [np_vector,  # TODO make all zeros
     np.random.uniform(-1, 1, layer_size),
     np.random.uniform(-1, 1, layer_size),
     np.random.uniform(-1, 1, 10)]

W = [np.random.uniform(-1, 1, (layer_size, input_size)),
     np.random.uniform(-1, 1, (layer_size, layer_size)),
     np.random.uniform(-1, 1, (10, layer_size))]

b = [np.random.uniform(-1, 1, layer_size),
     np.random.uniform(-1, 1, layer_size),
     np.random.uniform(-1, 1, 10)]


# show img
# img = Image.fromarray(np_image)
# img.save('my.png')
# img.show()


def ReLU(arr):  # TODO
    arr = np.squeeze(np.asarray(arr))
    new_arr = np.zeros(len(arr))
    for i, n in enumerate(arr):
        # if n > 0:
        #     new_arr[i] = n
        # else:
        #     new_arr[i] = 0
        new_arr[i] = max(0.0, n)
    return new_arr
    # return np.maximum()


def sigmoid(arr):
    a = np.squeeze(np.asarray(arr))
    out = np.zeros(len(a))
    for i, elem in enumerate(a):
        out[i] = 1 / (1 + math.pow(math.e, -elem))
    return out


def feed_forward(nodes, weights, basis):
    return sigmoid(np.dot(weights, nodes) + basis)


def cost(output, ans):
    return np.sum(np.square(output - ans))




a[1] = feed_forward(a[0], W[0], b[0])
a[2] = feed_forward(a[1], W[1], b[1])
a[3] = feed_forward(a[2], W[2], b[2])
print(a[3])
print(cost(a[3], np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])))

def predict(image):
    a[1] = feed_forward(image, W[0], b[0])
    a[2] = feed_forward(a[1], W[1], b[1])
    a[3] = feed_forward(a[2], W[2], b[2])
predict(a[0])
print(cost(a[3], np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])))

# img = Image.fromarray(a[1])
# img.save('my.png')
# img.show()

current_layer = 0  # stores the current layer being backprop'd

# function to start backpropagation on the whole model
# TODO : shuffle input for training
def train(batch):
    error = 0
    for i, image in enumerate(batch):
        output = predict(image)

        =grad_desc(dC_da_first())
        for i in range(3):
            =grad_desc(dC_da_gen())

        # error +=
    error /= len(batch)




# man func for calculating the gradient descent at layer L
def grad_desc(dC_da):
    # dc_db = dC_db()
    # grad = [
    # dc_db
    # some_num*dc_db
    # ]
    num_of_weights = len(W[current_layer])
    num_of_biases = len(b[current_layer])
    out = np.zeros(num_of_weights + num_of_biases)
    # pre_calculated = da/dz * dC/da
    # TODO : Reference stored values of z calculated while running
    _z = z(a[current_layer], W[current_layer], b[current_layer])
    pre_calculated = dsigmoid(z) * dC_da
    for i, weight in enumerate(W[current_layer]):
        out[2 * i] = dC_dw(pre_calculated)
    for i, bias in enumerate(b[current_layer]):
        out[2 * i + 1] = dC_db(pre_calculated)  # might overlap

    return out


# nodes, expected = np.array()
def dC_da_gen(nodes, pre_calculated) -> float:
    s = 0
    for i in range(len(nodes)):
        pass


# nodes, expected = np.array()
def dC_da_first(nodes, expected) -> float:
    return 2 * (nodes - expected)


# calculates dC/dw
def dC_dw(pre_calculated) -> float:
    return dz_dw(a[current_layer]) * pre_calculated


# calculates dz/dw
def dz_dw(nodes) -> float:
    s = 0
    for node in nodes:
        s += node
    return s


def dC_db(pre_calculated: float) -> float:
    return pre_calculated


# iterates through nodes and weights to calc z, which is passed through sigmoid to calculate the next nodes value
# nodes, weights = np.array()
def z(nodes, weights, bias: float) -> float:
    z = 0
    for i in range(len(nodes)):
        z += nodes[i] * weights[i]
    z += bias
    return z


# The derivative of the sigmoid function with respect to z
def dsigmoid(z: float) -> float:
    return sigmoid(z) * sigmoid(1 - z)
