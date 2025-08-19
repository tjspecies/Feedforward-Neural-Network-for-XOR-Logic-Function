# FFNNImplementation.py
# Name: Issa Tijani
# Class: DATA 512
# Term: Spring 2025
# Assignment: Homework 3 - Feedforward Neural Network for XOR

import numpy as np
import matplotlib.pyplot as plt


# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Initialize XOR inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
iterations = 10000
use_bias = True

# Initialize weights
np.random.seed(42)
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))

if use_bias:
    bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
    bias_output = np.random.uniform(-1, 1, (1, output_size))
else:
    bias_hidden = np.zeros((1, hidden_size))
    bias_output = np.zeros((1, output_size))

loss_history = []

# Training with batch gradient descent
for i in range(iterations):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Calculate error
    error = y - final_output
    loss = np.mean(np.square(error))
    loss_history.append(loss)

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    if use_bias:
        bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Save model parameters
with open("NNModelParameters.txt", "w") as f:
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Iterations: {iterations}\n")
    f.write(f"Final Error: {loss:.6f}\n")
    f.write(f"Network Structure: Input-{input_size}, Hidden-{hidden_size}, Output-{output_size}\n")
    f.write(f"Use Bias: {use_bias}\n")
    f.write(f"Final Weights (Input-Hidden):\n{weights_input_hidden}\n")
    f.write(f"Final Weights (Hidden-Output):\n{weights_hidden_output}\n")

# Plot cost vs iteration
plt.plot(loss_history)
plt.title('Cost vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.savefig('cost_vs_iteration.png')
plt.close()


# Prediction function
def predict(x):
    hidden = sigmoid(np.dot(x, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
    return np.round(output)


# Test predictions
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {predict(X[i])}, Actual Output: {y[i]}")
