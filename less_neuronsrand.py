import numpy as np


X = np.random.randn(3,3)
class softmax:
    def forward(self, inputs):
        exp_vals = np.exp(inputs - np.max(inputs))
        probabilities = exp_vals / np.sum(exp_vals)
        self.output = probabilities

def accuracy(inputs):
    targetvls = [0,1,1]
    maximum = np.argmax(inputs, axis=1)
    answers = np.mean(maximum == targetvls)
    return answers  # Add this line to return the accuracy value

softmax_activation = softmax()

softmax_activation.forward(X)

print(softmax_activation.output)

print("acc:", accuracy(softmax_activation.output))
