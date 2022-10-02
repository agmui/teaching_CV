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

a = [np_vector,  # TODO turn matrix to array
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


# output = feed_forward(
#     np.array([0.1, 0.1]),
#     np.asmatrix(
#         [[1, 2],
#          [1, 2],
#          [1, 2],
#          [1, 2]]
#     ),
#     np.array([0, 0, 0, 0])
# )
# print(output)

a[1] = feed_forward(a[0], W[0], b[0])
a[2] = feed_forward(a[1], W[1], b[1])
a[3] = feed_forward(a[2], W[2], b[2])
print(a[3])
