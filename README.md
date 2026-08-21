
# micrograd

![awww](puppy.jpg)

A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

### Installation

```bash
pip install micrograd
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

### Training a neural net

The notebook `demo.ipynb` provides a full demo of training an 2-layer neural network (MLP) binary classifier. This is achieved by initializing a neural net from `micrograd.nn` module, implementing a simple svm "max-margin" binary classification loss and using SGD for optimization. As shown in the notebook, using a 2-layer neural net with two 16-node hidden layers we achieve the following decision boundary on the moon dataset:

![2d neuron](moon_mlp.png)

### Training a GPT

For a more advanced example, see [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95), which trains and samples from a full GPT-2-like transformer in pure, dependency-free Python. It builds on a more efficient and better version of the autograd engine here (storing local gradients at forward time instead of per-op backward closures), and is the complete algorithm in a single file — everything else is just efficiency. See also the accompanying [explainer post](https://karpathy.github.io/2026/02/12/microgpt/) for a detailed walkthrough.

### Tracing / visualization

For added convenience, the notebook `trace_graph.ipynb` produces graphviz visualizations. E.g. this one below is of a simple 2D neuron, arrived at by calling `draw_dot` on the code below, and it shows both the data (left number in each node) and the gradient (right number in each node).

```python
from micrograd import nn
n = nn.Neuron(2)
x = [Value(1.0), Value(-2.0)]
y = n(x)
dot = draw_dot(y)
```

![2d neuron](gout.svg)

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python -m pytest
```

### License

MIT


---

## MyGrad: Personal Reimplementation

This fork retains Andrej Karpathy's original Micrograd implementation and adds my own reimplementation under the `mygrad/` package for learning automatic differentiation and neural network fundamentals.

本仓库保留原始Micrograd代码，并在`mygrad/`目录中增加了我从零实现的自动求导引擎和神经网络模块。

### Implemented Features

- Scalar computation graph
- Reverse-mode automatic differentiation
- Topological graph traversal
- Gradient accumulation
- Addition, multiplication and power operations
- ReLU activation
- Sigmoid activation
- Tanh activation
- Exponential operation
- Logarithm operation
- Neuron, Layer and MLP modules
- Gradient comparison with PyTorch
- Automated tests with pytest
- Moon dataset classification
- Decision-boundary visualization

### Project Structure

```text
micrograd/
├── micrograd/                 # Original Micrograd implementation
│   ├── engine.py
│   └── nn.py
├── mygrad/                    # Personal reimplementation
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
├── student_tests/             # Tests for MyGrad
│   ├── test_mygrad_engine.py
│   ├── test_mygrad_nn.py
│   └── test_mygrad_activations.py
├── examples/
│   ├── train_moons.py
│   └── moons_training_result.png
├── playground.py
├── nn_playground.py
└── README.md