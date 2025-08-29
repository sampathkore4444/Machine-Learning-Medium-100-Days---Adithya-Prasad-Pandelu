import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Generate input data
x = torch.linspace(-10, 10, 100)  # Generate 100 points between -10 and 10

print(x, x.detach().numpy(), x.numpy())


# Define activation functions using PyTorch
def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return F.relu(x)


def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.01)


def swish(x):
    return x * torch.sigmoid(x)


# Plot Activation Functions
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 3, 1)
plt.plot(x.numpy(), sigmoid(x).detach().numpy(), label="Sigmoid", color="blue")
plt.title("Sigmoid Activation Function")
plt.grid(True)

# Tanh
plt.subplot(2, 3, 2)
plt.plot(x.numpy(), tanh(x).detach().numpy(), label="Tanh", color="green")
plt.title("Tanh Activation Function")
plt.grid(True)

# ReLU
plt.subplot(2, 3, 3)
plt.plot(x.numpy(), relu(x).detach().numpy(), label="ReLU", color="red")
plt.title("ReLU Activation Function")
plt.grid(True)

# Leaky ReLU
plt.subplot(2, 3, 4)
plt.plot(x.numpy(), leaky_relu(x).detach().numpy(), label="Leaky ReLU", color="purple")
plt.title("Leaky ReLU Activation Function")
plt.grid(True)

# Swish
plt.subplot(2, 3, 5)
plt.plot(x.numpy(), swish(x).detach().numpy(), label="Swish", color="orange")
plt.title("Swish Activation Function")
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
