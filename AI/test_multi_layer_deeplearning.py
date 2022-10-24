from unittest import TestCase
import numpy as np
from multi_layer_deeplearning import NeuralNetwork
import math


def set1():
    nn = NeuralNetwork()
    nn.a[0] = np.array([0, 0])
    nn.W = [None, np.zeros((2, 2)), np.zeros((2, 2))]
    nn.b = [None, np.zeros(2), np.zeros(2)]
    return nn


def set2():
    nn = NeuralNetwork()
    nn.a[0] = np.array([1, 1])
    nn.W = [None, np.ones((2, 2)), np.ones((2, 2))]
    nn.b = [None, np.zeros(2), np.zeros(2)]
    return nn


def set3():
    nn = NeuralNetwork()
    nn.a[0] = np.array([0.5, 0.5])
    nn.W = [
        None,
        np.asmatrix([[0.5, 0.5], [0.5, 0.5]]),
        np.asmatrix([[0.5, 0.5], [0.5, 0.5]]),
    ]
    nn.b = [
        None,
        np.asmatrix([0.5, 0.5]),
        np.asmatrix([0.5, 0.5]),
    ]
    return nn


def set4():
    nn = NeuralNetwork()
    nn.a[0] = np.array([0.69, 0.420])
    nn.W = [
        None,
        np.asmatrix([[0.1, 0.3], [0.2, 0.4]]),
        np.asmatrix([[0.5, 0.7], [0.6, 0.8]]),
    ]
    nn.b = [
        None,
        np.asmatrix([0.1, 0.2]),
        np.asmatrix([0.5, 0.6]),
    ]
    return nn


def sig(n):
    return 1 / (1 + math.exp(-n))


class TestNeuralNetwork(TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()
        self.nn.a[0] = np.array([0, 0])
        self.nn.W = [None, np.zeros((2, 2)), np.zeros((2, 2))]
        self.nn.b = [None, np.zeros(2), np.zeros(2)]


class TestFunc(TestNeuralNetwork):
    def test_sigmoid(self):
        assert np.allclose(self.nn.sigmoid(np.array([0, 0])), np.array([0.5, 0.5]))
        # assert self.nn.sigmoid(np.array([1, 1]) == np.array([0.731058578630074, 0.731058578630074]))
        # assert self.nn.sigmoid(0.5) == 0.6224593312018959
        # assert self.nn.sigmoid(0.11) == 0.5274723043446033

    def test_cost(self):
        assert self.nn.cost(np.array([0, 0]), np.array([0, 0])) == 0
        assert self.nn.cost(np.array([1, 1]), np.array([0, 0])) == 2
        assert self.nn.cost(np.array([0.5, 0.5]), np.array([1, 1])) == 0.5
        assert self.nn.cost(np.array([0.69, 0.420]), np.array([0.99, 0.1])) == 0.1924

    def test_dsigmoid(self):
        pass
        # self.assertEqual(self.nn.dsigmoid(), 0)
        # self.assertEqual(self.nn.dsigmoid(), 0)
        # self.assertEqual(self.nn.dsigmoid(), 0)
        # self.assertEqual(self.nn.dsigmoid(), 0)


class TestFeedForward(TestNeuralNetwork):
    def test_feed_forward_one(self):
        nn = set1()
        # np.allclose()


class TestPredict(TestNeuralNetwork):
    def test_predict(self):
        # initializes with all 0s
        self.nn.predict()
        np.testing.assert_array_equal(self.nn.a[2], np.array([0.5, 0.5]))

    def test_predict_zeros(self):
        self.nn = set1()
        self.nn.predict()
        np.testing.assert_array_equal(self.nn.a[2], np.array([0.5, 0.5]))

    def test_predict_ones(self):
        self.nn = set2()
        self.nn.predict()
        expected = sig(sig(2) * 2)
        np.testing.assert_array_equal(self.nn.a[2], np.array([expected, expected]))

    def test_predict_uni(self):
        self.nn = set3()
        self.nn.predict()
        expected = sig(sig(1) + 0.5)
        np.testing.assert_array_equal(self.nn.a[2], np.array([expected, expected]))

    def test_predict_random(self):
        self.nn = set4()
        self.nn.predict()
        expected = [sig((0.5 * sig(0.295)) + (0.7 * sig(0.506)) + 0.5),
                    sig((0.6 * sig(0.295)) + (0.8 * sig(0.506)) + 0.6)]
        np.testing.assert_array_equal(self.nn.a[2], np.array(expected))


class TestGradDescOneLayer(TestNeuralNetwork):
    def test_grad_desc_one_layer(self):
        self.nn = set1()
        expected = []
        output = self.nn.grad_desc_one_layer(,2, )
        np.testing.assert_array_equal()


class TestGradientDescent(TestNeuralNetwork):
    def test_grad_desc(self):
        pass
        # self.fail()
