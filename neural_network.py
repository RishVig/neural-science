
import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()



class activation_relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)
        
class activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class loss:
    def calculate(self, output,y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class loss_catagoricalcrossentropy(loss):
    def forward(self, y_prediction, y_true):
        samples = len(y_prediction)
        y_prediction_cliped = np.clip(y_prediction, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_prediction_cliped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_prediction_cliped * y_true, axis= 1)

        negative_loglikelehoods = -np.log(correct_confidences)

        return negative_loglikelehoods
X,y = spiral_data(samples = 100, classes = 3)
dense1 = layer_dense(2, 3)

activation1 = activation_relu()

dense2 = layer_dense(3,3)
activation2 = activation_softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = loss_catagoricalcrossentropy()
loss = loss_function.calculate(activation2.output, y)

def accuracy_calculations(inputs):
    
    target_values = [0,1,1]
    maximum = np.argmax(inputs, axis=1)
    answers = np.mean(maximum == target_values)
    print("acc:", answers,"three lines of dataset in use")
    
accuracy_calculations(activation2.output[:3])

print("Loss: ", loss)
