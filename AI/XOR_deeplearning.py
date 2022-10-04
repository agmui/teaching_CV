import math

import numpy as np
from PIL import Image

# expand img to a vector and scale down between 0 and 1

a = [
    np.array([0, 0]),  # inputs
    np.array([0]),  # hiddenlayer
    np.array([0])  # output
]

W = [
    np.random.uniform(-1, 1, (1, 2)),
    np.random.uniform(-1, 1, (1, 1))
]

b = [
    np.random.uniform(-1, 1, 1),  # hidden layer
    np.random.uniform(-1, 1, 1)  # hidden layer
]


# show img
# img = Image.fromarray(np_image)
# img.save('my.png')
# img.show()


def ReLU(arr):  # TODO
    arr = np.squeeze(np.asarray(arr))
    new_arr = np.zeros(len(arr))
    for i, n in enumerate(arr):
        new_arr[i] = max(0.0, n)
    return new_arr
    # return np.maximum()


def sigmoid(arr):
    # a = np.squeeze(np.asarray(arr))
    # out = np.zeros(len(a))
    # for i, elem in enumerate(a):
    #     out[i] = 1 / (1 + math.pow(math.e, -elem))
    # return out
    return 1 / (1 + math.pow(math.e, -arr))


def feed_forward(nodes, weights, basis):
    return sigmoid(np.dot(weights, nodes) + basis)


def cost(output, ans):
    return np.sum(np.square(output - ans))


def predict():
    a[1] = feed_forward(a[0], W[0], b[0])
    a[2] = feed_forward(a[1], W[1], b[1])


a[0] = [0, 0]
predict()
print("output:", a[2])
print("cost:", cost(a[2], np.array([0])))

# img = Image.fromarray(a[1])
# img.save('my.png')
# img.show()

current_layer = 1  # stores the current layer being backprop'd


# function to start backpropagation on the whole model
# TODO : shuffle input for training
def grad_desc(inputs, ans):
    global current_layer
    a[0] = inputs
    predict()

    # gradient decent for first layer
    grad_desc_vector = np.array([])
    dC_da = dC_da_first(ans)
    np.append(grad_desc_vector, grad_desc_one_layer(dC_da))

    # gradient decent for hidden layers
    for i in range(2 - 1):  # number of layers - 1
        current_layer -= 1
        dC_da = dC_da_gen(dC_da)
        np.append(grad_desc_vector, grad_desc_one_layer(dC_da))

    return grad_desc_vector


# man func for calculating the gradient descent at layer L
def grad_desc_one_layer(dC_da):
    num_of_weights = len(W[current_layer])
    num_of_biases = len(b[current_layer])
    out = np.zeros(num_of_weights + num_of_biases)
    # pre_calculated = da/dz * dC/da
    # TODO : Reference stored values of z calculated while running
    _z = z(a[current_layer], W[current_layer], b[current_layer])
    pre_calculated = dsigmoid(_z) * dC_da
    for i, weight in enumerate(W[current_layer]):
        out[2 * i] = dC_dw(pre_calculated)
    for i, bias in enumerate(b[current_layer]):
        out[2 * i + 1] = dC_db(pre_calculated)  # might overlap

    return out


# nodes, expected = np.array()
def dC_da_gen(dC_da) -> float:
    return W[]*dsigmoid()*dC_da


# nodes, expected = np.array()
def dC_da_first(expected) -> float:
    return 2 * (a[2] - expected)


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
