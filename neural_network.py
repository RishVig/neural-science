import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Create the layers and perform the forward pass
layer_1 = Layer_dense(2, 5)  # n_inputs = 2 (since we have 2 features), n_neurons = 5
activation_1 = Activation_ReLU()

layer_1 = Layer_dense(2,5)

layer_1.forward(X)
activation_1 = Activation_ReLU()
layer_1.forward(X)

activation_1.forward(layer_1.output)
print(np.version)
print(activation_1.output)

# Now, the output of the first layer after applying ReLU activation is in activation_1.output
print("Thank You for viewing our data!!")
information_center = input("Want more information, [y] - yes, [n] - no: ")

if information_center == "y":
    print("Inforamtion at: https://github.com/RishVig/neural-science, graph: https://github.com/RishVig/neural-science/blob/main/graph.py")
else:
    print("Thank you for using your time to listen and watch my presentation")
