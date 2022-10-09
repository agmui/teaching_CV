from unittest import TestCase

import numpy as np

from multi_layer_deeplearning import NeuralNetwork


def set1():
    nn = NeuralNetwork()
    nn.a[0] = [0,0]
    nn.W = [np.zeros((2,2)), np.zeros((2,2))]
    nn.b = [

    ]
    return nn


def set2():
    nn = NeuralNetwork()
    nn.a = [

    ]
    nn.W = [

    ]
    nn.b = [

    ]
    return nn


def set3():
    nn = NeuralNetwork()
    nn.a = [

    ]
    nn.W = [

    ]
    nn.b = [

    ]
    return nn


def set4():
    nn = NeuralNetwork()
    nn.a = [

    ]
    nn.W = [

    ]
    nn.b = [

    ]
    return nn


class TestNeuralNetwork(TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()
        self.nn.W = [

        ]
        self.nn.b = [

        ]


class TestFunc(TestNeuralNetwork):
    def test_sigmoid(self):
        self.assertEqual(self.nn.sigmoid(), 0)
        self.assertEqual(self.nn.sigmoid(), 0)
        self.assertEqual(self.nn.sigmoid(), 0)
        self.assertEqual(self.nn.sigmoid(), 0)

    def test_cost(self):
        self.assertEqual(self.nn.cost(), 0)
        self.assertEqual(self.nn.cost(), 0)
        self.assertEqual(self.nn.cost(), 0)
        self.assertEqual(self.nn.cost(), 0)

    def test_dsigmoid(self):
        self.assertEqual(self.nn.dsigmoid(), 0)
        self.assertEqual(self.nn.dsigmoid(), 0)
        self.assertEqual(self.nn.dsigmoid(), 0)
        self.assertEqual(self.nn.dsigmoid(), 0)


class TestFeedForward(TestNeuralNetwork):
    def test_feed_forward(self):
        nn = set1()


class TestPredict(TestNeuralNetwork):
    def test_predict(self):
        self.fail()


class TestGradDescOneLayer(TestNeuralNetwork):
    def test_grad_desc_one_layer(self):
        self.fail()


class TestGradientDescent(TestNeuralNetwork):
    def test_grad_desc(self):
        self.fail()
