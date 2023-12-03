import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

class NeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.h1 = Neuron(weights[0], bias[0])
        self.h2 = Neuron(weights[1], bias[1])
        self.o1 = Neuron(weights[2], bias[2])
    
    def feedforward(self, inputs):
        output_h1 = self.h1.feedforward(inputs)
        output_h2 = self.h2.feedforward(inputs)
        output_o1 = self.o1.feedforward(np.array([output_h1, output_h2]))

        return output_o1

weights = np.array([[0, 1], [0, 1], [0, 1]])
bias = np.array([0, 0, 0])
network = NeuralNetwork(weights, bias)
x = np.array([2, 3])
print(network.feedforward(x))