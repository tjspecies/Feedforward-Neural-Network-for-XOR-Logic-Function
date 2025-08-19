# Feedforward Neural Network for XOR Logic Function

Overview: This project implements a feedforward neural network (FFNN) from scratch in Python using NumPy and Matplotlib to solve the XOR logic problem. XOR is not linearly separable, making it an essential case study for understanding neural networks with hidden layers. The network consists of 2 input nodes, 2 hidden nodes, and 1 output node (2-2-1 architecture) with sigmoid activation.
Both Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD) were implemented to compare performance. Bias nodes were also tested to evaluate their impact.

# Dataset
Source: XOR truth table
# Inputs/Outputs:
[0,0] → 0
[0,1] → 1
[1,0] → 1
[1,1] → 0

# Methodology
Architecture: 2-2-1 feedforward neural network
Activation Function: Sigmoid (hidden and output layers)
Loss Function: Mean Absolute Error (MAE)
Training Methods: Batch Gradient Descent (BGD) and Stochastic Gradient Descent (SGD)

Learning Rate: 0.5

Epochs: 10,000

Implementation Steps

Initialize weights and biases randomly with fixed seed.

Implement feedforward propagation using sigmoid.

Apply backpropagation with gradient updates.

Train using both BGD and SGD.

Save final weights, biases, learning parameters, and errors to NNModelParameters.txt.

Plot error trends (error_plot.png) for both training methods.

Implement prediction functions for XOR inputs.

Results
Metric	BGD	SGD
Learning Rate	0.5	0.5
Iterations	10,000	10,000
Final MSE	0.00122	0.00186
Predictions [0,0]	0.019	0.019
Predictions [0,1]	0.983	0.980
Predictions [1,0]	0.983	0.979
Predictions [1,1]	0.017	0.026

Output Stability	More stable	Slightly noisier
BGD: Lower error, smoother convergence, more stable predictions.
SGD: Slightly noisier but faster updates, competitive performance.

# Discussion
Both BGD and SGD successfully learned the XOR function.
BGD proved more stable and precise, while SGD offered efficiency and adaptability.
The results illustrate the trade-offs between batch and stochastic optimization strategies.

# Conclusion

The project demonstrates successful implementation of a neural network for XOR classification. Both BGD and SGD achieved high accuracy, with BGD performing slightly better in stability and precision. This exercise highlights the fundamentals of neural network training and provides a foundation for more complex applications.

# References
Python Documentation: NumPy, Matplotlib
Dataset: XOR Truth Table
Course lecture notes on neural networks and backpropagation
Gradient Descent – Wikipedia
Nielsen, M. (2015). Neural Networks and Deep Learning
