from unittest import TestCase

from multi_layer_deeplearning import NeuralNetwork


class TestNeuralNetwork(TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()
        self.nn.W = [

        ]
        self.nn.b = [

        ]


class TestNeuralNetwork2(TestCase):
    def setUp(self):
        self.nn = NeuralNetwork()
        self.nn.a = [

        ]
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
        self.fail()


class TestPredict(TestNeuralNetwork):
    def test_predict(self):
        self.fail()


class TestGradDescOneLayer(TestNeuralNetwork2):
    def test_grad_desc_one_layer(self):
        self.fail()


class TestGradientDescent(TestNeuralNetwork2):
    def test_grad_desc(self):
        self.fail()
