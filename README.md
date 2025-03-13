# TaylorVar: A tiny high-order forward propagation automatic differentiation tool in PyTorch

TaylorVar is a lightweight PyTorch wrapper aimed at providing explicit construction of computational graphs for Taylor-mode automatic differentiation, enabling the computation of higher-order derivatives in a single forward pass without stacked first-order AD as in original PyTorch. It is designed for tasks that require accurate higher-order derivatives (up to 3), such as in Physics-Informed Neural Networks (PINNs), scientific computing, and differential equation solvers.


## Features
- Simple construction of Taylor-mode computational graphs from scalar(or tensor) input.
- Compute up to 3rd-order derivatives in a single forward pass.
- Support lazy calculation of full or partial derivative via indexing.
- Easy integration with PyTorch workflow.
- Support basic operations including:
    - Arithmetic operations (`+`, `-`, `*`) with broadcast mechanism supported
    - Linear layer (`matmul`)
    - Element-wise non-linear functions (use pre-built or custom activation functions)
    - Indexing and slicing
    - Other operations(`reshape`, `view`, `flatten`, `squeeze`, `unsqueeze`, `cat`, `stack`)
- Compatible with PyTorch functional API.


## Installation
```bash
git clone https://github.com/fhl2000/TaylorVar.git
cd TaylorVar
pip install -e .
```

## Quick Start
Below is a simple example demonstrating how to use TaylorVar for computing the value and its derivatives:

```python
import torch
from taylorvar import TaylorVar

# Create input tensor with explicit derivative information
x = torch.tensor([1.0, 2.0])
first_init =torch.eye(2)
# Create TaylorVar, initializing first-order derivative as an identity matrix
x_tv = TaylorVar(x, first=first_init)

# Perform a basic arithmetic operation: f(x₁,x₂) = x₁*x₂ + 3*x₂
y_tv = x_tv[0] * x_tv[1] + 3 * x_tv[1]

# Extract computed results (pytorch tensor returned)
print("Function Value:", y_tv.val)
print("First-order Derivative:", y_tv.first[...])  # used indexing to trigger lazy calculation
print("Second-order Derivative:", y_tv.second[...])
print("Third-order Derivative:", y_tv.third[...])
```

## Examples

### 1. Basic Operations
TaylorVar supports operations with scalars as well as tensors wrapped as `TaylorVar` objects with proper propagation of derivatives.

```python
import torch
from taylorvar import TaylorVar, taylor_activation_wrapper, get_activation_with_derivatives

x = torch.randn(10,2)  # (batch_size, input_dim)
# Create TaylorVar objects
first_init = torch.zeros(x.shape + (x.shape[-1],), dtype=x.dtype, device=x.device)
for i in range(2):
    first_init[...,i,i] = 1.0
x_tv = TaylorVar(x, first_init)

# Linear layer
weight = torch.randn(8,2)    # (hidden_dim, input_dim)
bias = torch.randn(8)        # (hidden_dim)
h_tv = x_tv.linear(weight, bias)   # (batch_size, hidden_dim)

# Activation function (tanh)
tanh_wrapper = taylor_activation_wrapper(*get_activation_with_derivatives('tanh'))
y_tv = tanh_wrapper(h_tv)    # (batch_size, hidden_dim)


# Addition
z_add = h_tv + y_tv
print("Addition Result:", z_add.val)
print("Addition First Derivative:", z_add.first[...])  # (batch_size, hidden_dim, input_dim)  # per-sample Jacobian
print("Addition Second Derivative:", z_add.second[...])  # (batch_size, hidden_dim, input_dim, input_dim)  # per-sample Hessian

# Multiplication
z_mul = h_tv * y_tv
print("Multiplication Result:", z_mul.val)
print("Multiplication First Derivative:", z_mul.first[...])  # (batch_size, hidden_dim, input_dim) 
print("Multiplication Second Derivative:", z_mul.second[...])  # (batch_size, hidden_dim, input_dim, input_dim)

# Forward Laplacian 
laplacian = z_mul.second[0,0] + z_mul.second[1,1]  # (batch_size, hidden_dim), partial calculation of derivatives via indexing

```

### 2. Neural Network Example

You can integrate TaylorVar with standard PyTorch neural networks. Here is a simple feed-forward network demo:

```python
import torch
import torch.nn as nn
from taylorvar import TaylorVar, taylor_activation_wrapper, get_activation_with_derivatives

class SimpleNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = taylor_activation_wrapper(*get_activation_with_derivatives('tanh'))
    
    def forward(self, x, compute_taylor=False):
        if not compute_taylor:
            # Standard forward pass
            x = torch.tanh(self.fc1(x))
            return self.fc2(x)
        else:
            # Taylor-mode forward pass: create a TaylorVar object
            x_tv = TaylorVar(x, first=torch.eye(x.shape[-1]), order=2)
            h = x_tv.linear(self.fc1.weight, self.fc1.bias)
            h = self.activation(h)
            out_tv = h.linear(self.fc2.weight, self.fc2.bias)
            return out_tv

# construct the model and optimizer
model = SimpleNN(in_features=2, hidden_features=64, out_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# construct the input and target
batch_size = 10
input_tensor = torch.randn(batch_size, 2, dtype=torch.float64, requires_grad=True)

target_data = torch.randn(batch_size, 1 dtype=torch.float64)
target_laplacian = torch.zeros(batch_size, 1, dtype=torch.float64)

# compute loss
loss_fn = torch.nn.MSELoss()
output = model(input_tensor, compute_taylor=True)
loss_data = loss_fn(output.val, target_data)
loss_laplacian = loss_fn(output.second[0,0] + output.second[1,1], target_laplacian)
loss = loss_data + loss_laplacian
# backward propagation and update the model parameters
loss.backward()
optimizer.step()
```


## Contribution

This project is still on an early alpha stage, and we welcome contributions! Please submit issues or pull requests if you have suggestions or improvements.

## License

MIT License

## Citation

If you find this project useful and use it in your research, please cite:

```
@software{taylorvar2025,
  author = {Haolong Fan},
  title = {TaylorVar: A Tiny High-Order Forward Propagation Automatic Differentiation Tool in PyTorch},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/fhl2000/TaylorVar}
}
```
