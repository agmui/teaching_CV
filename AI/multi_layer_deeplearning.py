import math

import numpy as np


class NeuralNetwork:
    def __init__(self, W=None, b=None):
        self.a: list[np.ndarray] = [
            np.array([0, 0]),  # inputs
            np.array([0, 0]),  # hiddenlayer
            np.array([0, 0])  # output
        ]

        if W is None:
            self.W: list[np.ndarray] = [
                None,  # don't use this index it is only to make the layer code look like the math
                np.random.uniform(-1, 1, (2, 2)),  # hidden layer
                np.random.uniform(-1, 1, (2, 2))  # output
            ]

        if b is None:
            self.b: list[np.ndarray] = [
                None,  # don't use this index it is only to make the layer code look like the math
                np.random.uniform(-1, 1, 2),  # hidden layer
                np.random.uniform(-1, 1, 2)  # output
            ]
        self.num_layers = 3

    def sigmoid(self, arr: np.ndarray) -> np.ndarray:
        arr_squeeze = np.squeeze(np.asarray(arr))
        out = np.zeros(len(arr_squeeze))
        for i, elem in enumerate(arr_squeeze):
            out[i] = 1 / (1 + math.pow(math.e, -elem))
        return out

    def feed_forward(self, nodes:np.ndarray, weights: np.ndarray, basis: np.ndarray) -> np.ndarray:
        z=(np.dot(weights, nodes))
        x=(np.dot(weights, nodes)+basis)
        y=self.sigmoid(np.dot(weights, nodes)+basis)
        return self.sigmoid(np.dot(weights, nodes) + basis)

    def cost(self, output, ans) -> np.ndarray:
        return np.sum(np.square(output - ans))

    def predict(self):
        self.a[1] = self.feed_forward(self.a[0], self.W[1], self.b[1])
        self.a[2] = self.feed_forward(self.a[1], self.W[2], self.b[2])

    # function to start backpropagation on the whole model
    # TODO : shuffle input for training
    def grad_desc(self, inputs, ans):
        for i, batch in enumerate(inputs):
            self.a[0] = batch
            self.predict()

            dC_da = np.array([0, 0])
            avg_grad = np.array([])
            grad_C, dC_da = self.grad_desc_one_layer(dC_da, 2, ans)  # first case
            for L in range(self.num_layers - 1, 0):
                grad_C, dC_da = self.grad_desc_one_layer(dC_da, L)
                avg_grad.append(grad_C)  # TODO dont append
            np.average(avg_grad)  # TODO find avg
        return avg_grad

    # y is a single array with the correct answer
    # dC_da is an array because there are as many dC_da as there are activations nodes
    def grad_desc_one_layer(self, dC_da: np.ndarray, layer, y: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        # step1: find dC/da and z
        z: np.ndarray = np.zeros(len(self.a[layer]))
        for j in range(len(z)):  # TODO should already have this info
            z[j] = np.sum(self.W[layer][j]) * np.sum(self.a[layer - 1]) + self.b[layer][j]  # can just be a dot product

        if layer == self.num_layers - 1:  # first case
            for j in range(len(self.a[2])):
                dC_da[j] = 2 * (self.a[2][j] - y[j])
        else:  # general case
            z_l1: np.ndarray = np.zeros(len(self.a[layer]))
            for j in range(len(z)):  # TODO should already have this info
                z_l1[j] = np.sum(self.W[layer][j]) * np.sum(self.a[layer - 1]) + self.b[layer][
                    j]  # can just be a dot product

            for k in range(len(self.a[layer])):  # TODO may have different dC_da len
                sum_: float = 0
                sig = self.dsigmoid(z_l1)  # FIXME should be in the for loop below
                for j in range(len(self.a[layer + 1])):
                    sum_ += self.W[layer + 1][j][k] * sig[j] * dC_da[j]  # TODO could cause error cuz flipped j and k
                dC_da[k] = sum_

        # Step2: find dC/dw and dC/db
        # dC_db_results: np.ndarray = np.zeros(len(b[1]))
        dC_db_results = self.dsigmoid(z) * dC_da
        dC_dw_results: np.ndarray = np.zeros(len(self.a[layer]) * len(self.W[layer][0]))  # TODO optimise
        for j in range(len(self.a[layer])):  # the j and k follow the vid
            for k in range(len(self.W[layer][0])):
                dC_dw_results[j + k] = self.a[layer - 1][k] * dC_db_results[j]  # TODO optimise

        # Step3: add to grad_C
        # grad_C: np.ndarray = np.zeros(len(dC_dw_results) + len(dC_db_results))
        # for k, dC_dw in enumerate(dC_dw_results):  # FIXME
        #     if k % 3 != 0:  # to zipper the w and b properly
        #         grad_C[k] = dC_dw
        # for k, dC_db in enumerate(dC_db_results):  # TODO make general case
        #     if k % 3 == 0:  # to zipper the w and b properly
        #         grad_C[k] = dC_db
        return np.concatenate((dC_dw_results, dC_db_results)), dC_da

    # The derivative of the sigmoid function with respect to z
    def dsigmoid(self, z: np.ndarray) -> np.ndarray:
        return self.sigmoid(z) * self.sigmoid(1 - z)


if __name__ == "__main__":
    nn: NeuralNetwork = NeuralNetwork()
    nn.a[0] = [0, 1]
    nn.predict()
    print("output:", nn.a[2])
    print("cost:", nn.cost(nn.a[2], np.array([0, 0])))

    grad_C1, dC_da = nn.grad_desc_one_layer(np.array([0, 0]), 2, np.array([0, 1]))
    grad_C2, dC_da = nn.grad_desc_one_layer(dC_da, 1)
    print(grad_C1, grad_C2)
