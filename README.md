<p align="center">
    <img width=600 src="https://raw.githubusercontent.com/hasnainroopawalla/ShowML/master/static/images/showml.png" alt="ShowML Logo">
</p>

---

[![Develop](https://github.com/hasnainroopawalla/ShowML/actions/workflows/develop.yml/badge.svg)](https://github.com/hasnainroopawalla/ShowML/actions/workflows/develop.yml)
[![Deploy](https://github.com/hasnainroopawalla/ShowML/actions/workflows/deploy.yml/badge.svg)](https://github.com/hasnainroopawalla/ShowML/actions/workflows/deploy.yml)
[![PyPi version](https://img.shields.io/pypi/v/showml.svg)](https://pypi.python.org/pypi/showml/)
[![Python versions](https://img.shields.io/pypi/pyversions/showml.svg?style=plastic)](https://img.shields.io/pypi/pyversions/showml.svg?style=plastic)
![Downloads](https://img.shields.io/pypi/dm/showml.svg)


A Python package of Machine Learning Algorithms implemented from scratch.

The aim of this package is to present the working behind fundamental Machine Learning algorithms in a transparent and modular way.

> **_NOTE:_** The implementations of these algorithms are not thoroughly optimized for high computational efficiency.


## 📝 Table of Contents

- [Getting Started](#getting_started)
- [Contents](#contents)
- [Contributing](#contributing)
- [License](#license)


## 🏁 Getting Started <a name = "getting_started"></a>

### To install the package directly from PyPi:
```
$ pip install showml
```

### To clone the repository and view the source files:
```
$ git clone https://github.com/hasnainroopawalla/ShowML.git
$ cd ShowML
$ pip install -r requirements.txt
```
Remember to add `ShowML/` to the `PYTHONPATH` environment variable before using locally:-

- For Windows:
  ```
  $ set PYTHONPATH=%PYTHONPATH%;<path-to-directory>\ShowML
  ```
- For MacOS:
  ```
  $ export PYTHONPATH=/<path-to-directory>/ShowML:$PYTHONPATH
  ```
- For Linux:
  ```
  $ export PYTHONPATH="${PYTHONPATH}:/<path-to-directory>/ShowML"
  ```
> **_Check out:_** [showml/examples/](https://github.com/hasnainroopawalla/ShowML/tree/master/showml/examples)


## 📦 Contents <a name = "contents"></a>
_ShowML_ currently includes the following content, however, this repository will continue to expand in order to include implementations of many more Machine Learning Algorithms.

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
2. Commit and push your changes to your own branch.
3. Install and run the necessary housekeeping dependencies ([pre-commit](https://pre-commit.com/), [mypy](https://github.com/python/mypy) and [pytest](https://docs.pytest.org)):
    ```
    $ pip install pre-commit mypy pytest
    ```
4. Run these housekeeping checks locally and make sure all of them succeed (required for the CI to pass):-
   ```
   $ pre-commit run -a
   $ mypy .
   $ pytest
   ```
5. Open a Pull Request and I'll review it.


## 📄 License <a name = "license"></a>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/hasnainroopawalla/ShowML/blob/bbaacc81779437ea2ef09d7869b1f8a824f80353/LICENSE) file for details.
