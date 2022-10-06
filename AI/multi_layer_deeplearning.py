import math

import numpy as np
from PIL import Image

# expand img to a vector and scale down between 0 and 1

a = [
    np.array([0, 0]),  # inputs
    np.array([0, 0]),  # hiddenlayer
    np.array([0, 0])  # output
]

W = [
    np.random.uniform(-1, 1, (2, 2)),  # hidden layer
    np.random.uniform(-1, 1, (2, 2))  # output
]

b = [
    np.random.uniform(-1, 1, 2),  # hidden layer
    np.random.uniform(-1, 1, 2)  # output
]

ans = [0, 0]
num_layers = 3


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


def predict():
    a[1] = feed_forward(a[0], W[0], b[0])
    a[2] = feed_forward(a[1], W[1], b[1])


# a[0] = [0, 0]
# predict()
# print("output:", a[2])
# print("cost:", cost(a[2], np.array([0,0])))


# function to start backpropagation on the whole model
# TODO : shuffle input for training
def grad_desc(inputs, ans):
    for i, batch in enumerate(inputs):
        a[0] = batch
        predict()
        dC_da = np.array([0, 0])
        avg_grad = np.array([])
        for L in range(num_layers - 1, 0):
            grad_C, dC_da = grad_desc_one_layer(dC_da, L, ans[i])
            avg_grad.append(grad_C)  # TODO dont append
        np.average(avg_grad)  # TODO find avg
    return avg_grad


# y is a single array with the correct answer
# dC_da is an array because there are as many dC_da as there are activations nodes
def grad_desc_one_layer(dC_da: np.ndarray, layer, y: np.ndarray):
    # step1: find dC/da and z
    z: np.ndarray = np.zeros(len(a[layer]))
    for j in range(len(z)):  # TODO should already have this info
        z[j] = np.sum(W[layer][j]) * np.sum(a[layer - 1]) + b[layer][j]  # can just be a dot product

    if layer == num_layers - 1:  # first case
        for j in range(len(a[2])):
            dC_da[j] = 2 * (a[2][j] - y[j])
    else:  # general case
        for k, n in enumerate(a[layer]):  # TODO may have different dC_da len
            sum_: float = 0
            for j in range(len(n)):
                sum_ += W[layer][k][j] * dsigmoid(z[j]) * dC_da[j]
            dC_da[k] = sum_

    # step2: find dC/dw and dC/db
    dC_db_results: np.ndarray = np.zeros(len(b[0]))
    dC_dw_results: np.ndarray = np.zeros(len(W[0][0]))
    for j in range(len(a[layer])):  # the j and k follow the vid
        dC_db_results[j] = dsigmoid(z[j]) * dC_da[j]
        dC_dw_results[j] = np.sum(a[layer][j]) * dC_db_results[j]

    # Step3: add to grad_C
    grad_C: np.ndarray = np.zeros(len(dC_dw_results) + len(dC_db_results))
    for k, dC_dw in enumerate(dC_dw_results):  # FIXME
        if k % 3 != 0:  # to zipper the w and b properly
            grad_C[k] = dC_dw
    for k, dC_db in enumerate(dC_db_results):  # TODO make geniral case
        if k % 3 == 0:  # to zipper the w and b properly
            grad_C[k] = dC_db
    return grad_C, dC_da


# The derivative of the sigmoid function with respect to z
def dsigmoid(z: float) -> float:
    return sigmoid(z) * sigmoid(1 - z)
