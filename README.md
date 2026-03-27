# Macine-Learning-Tutorial


How Deep Neural Networks Learn — MLP Tutorial

This project demonstrates how Multi-Layer Perceptrons (MLPs) learn hierarchical representations using visualizations and simple simulations.

It is based on a detailed tutorial explaining deep learning concepts step-by-step.

Overview

Deep learning models transform raw input into meaningful features across multiple layers.

This project covers:

Representation learning
Backpropagation
Activation functions (ReLU, Sigmoid, Tanh)
Vanishing gradient problem
Performance comparison
Visualizations Included

The Python script generates the following figures:

1. MLP Architecture
Visual representation of neurons and layers
Shows how data flows through the network

2. Hierarchical Learning
Demonstrates feature extraction
From raw pixels → shapes → concepts

3.  Activation Functions
Comparison of ReLU, Sigmoid, and Tanh
Includes advantages and limitations

4. Training Accuracy
Shows convergence of different activation functions
ReLU performs best

5.  Vanishing Gradient
Demonstrates gradient shrinkage in Sigmoid
ReLU maintains strong gradients

 Experimental Setup
Architecture: 3 hidden layers × 128 neurons
Dataset: MNIST
Optimizer: SGD with momentum
Learning Rate: 0.01
Epochs: 50
Loss Function: Cross-entropy

How to Run
1. Clone the repository
git clone https://github.com/Dankstan517/Macine-Learning-Tutorial.git
cd Macine-Learning-Tutorial

2. Install dependencies
pip install matplotlib numpy

3. Run the script
python machineLearning.py

Author
Dankstan Antony Alfred
University of Hertfordshire
