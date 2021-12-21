<p align="center">
    <img height=250 src="./static/images/showml.png" alt="ShowML Logo">
</p>

---

<h2 align="center"><b><i>Show</i></b> the <b><i>M</i></b>achine <b><i>L</i></b>earning Code</h2>

[![Python packaging](https://github.com/hasnainroopawalla/ShowML/actions/workflows/python_packaging.yml/badge.svg?branch=master)](https://github.com/hasnainroopawalla/ShowML/actions/workflows/python_packaging.yml)
[![PyPi version](https://img.shields.io/pypi/v/showml.svg)](https://pypi.python.org/pypi/py_d3/)
[![Python versions](https://img.shields.io/pypi/pyversions/showml.svg?style=plastic)](https://img.shields.io/pypi/pyversions/showml.svg?style=plastic)
![Status](https://img.shields.io/badge/status-stable-green.svg)
---

A Python package of Machine Learning Algorithms implemented from scratch.

The aim of this package is to present the working behind fundamental Machine Learning algorithms in a transparent and modular way.

> **_NOTE:_** The implementations of these algorithms are not thoroughly optimized for high computational efficiency.

## 📝 Table of Contents

- [Getting Started](#getting_started)
- [Package Contents](#package_contents)
- [Contributing](#contributing)

## 🏁 Getting Started <a name = "getting_started"></a>

Install the package:
```
$ pip install showml
```

To clone the repository and view the source files:
```
$ git clone https://github.com/hasnainroopawalla/ShowML.git
$ cd ShowML
$ pip install -r requirements.txt
```
> **_Check out:_** [showml/examples/](https://github.com/hasnainroopawalla/ShowML/tree/master/showml/examples)
>

## 📦 Package Contents <a name = "package_contents"></a>

### Models
- Linear
  - Linear Regression (`showml.linear_model.regression.LinearRegression`)
  - Logistic Regression (`showml.linear_model.regression.LogisticRegression`)

- Non-Linear
  - Sequential (`showml.deep_learning.model.Sequential`)

### Deep Learning
- Layers
  - Dense (`showml.deep_learning.layers.Dense`)

- Activations
  - Sigmoid (`showml.deep_learning.activations.Sigmoid`)
  - ReLu (`showml.deep_learning.activations.Relu`)
  - Softmax (`showml.deep_learning.activations.Softmax`)

### Optimizers
- Stochastic/Batch/Mini-Batch Gradient Descent (`showml.optimizers.SGD`)
- Adaptive Gradient (`showml.optimizers.AdaGrad`)
- Root Mean Squared Propagation (`showml.optimizers.RMSProp`)

### Loss Functions
- Mean Squared Error (`showml.losses.MeanSquaredError`)
- Binary Cross Entropy (`showml.losses.BinaryCrossEntropy`)
- Categorical Cross Entropy (`showml.losses.CrossEntropy`)

### Simulations
- [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) (`showml.simulations.conways_game_of_life.GameOfLife`)


## ✏️ Contributing <a name = "contributing"></a>

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository.
2. Install the necessary dependencies:
```
$ pip install pre-commit mypy pytest
 ```
3. Commit and push your changes to your own branch.
4. Before submitting a Pull Request, run these housekeeping checks locally:-
  - Run [pre-commit](https://pre-commit.com/):
   ```
   $ pre-commit run -a
   ```
  - Run [mypy](https://github.com/python/mypy):
  ```
  $ mypy .
  ```
  - Run [pytest](https://docs.pytest.org):
  ```
  $ pytest
  ```
