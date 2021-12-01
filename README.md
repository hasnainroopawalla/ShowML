# ShowML

[![Python packaging](https://github.com/hasnainroopawalla/ShowML/actions/workflows/python_packaging.yml/badge.svg?branch=master)](https://github.com/hasnainroopawalla/ShowML/actions/workflows/python_packaging.yml)

**Show** the **ML** Code!

A Python package of Machine Learning Algorithms implemented from scratch.

The aim of this package is to present the working behind fundamental Machine Learning algorithms in a transparent and modular way.

> **_NOTE:_**  The implementations of these algorithms are not thoroughly optimized for high computational efficiency.


## Usage
[showml/examples](https://github.com/hasnainroopawalla/ShowML/tree/master/showml/examples) contains examples of using ShowML to train various Machine Learning models.


## Usage
[showml/examples](https://github.com/hasnainroopawalla/ShowML/tree/master/showml/examples) contains examples of using ShowML to train various Machine Learning models.

## Installation


Install the package
```
$ pip install showml
```

To clone the repository and view the source files
```
$ git clone https://github.com/hasnainroopawalla/ShowML.git
$ cd ShowML
$ pip install -r requirements.txt
```

[How to Contribute](#contributing)


## Contents

### Models
#### Linear
- Linear Regression (`showml.supervised.regression.LinearRegression`)
- Logistic Regression (`showml.supervised.regression.LogisticRegression`)

#### Non-Linear
- Sequential (`showml.deep_learning.network.Sequential`)

### Deep Learning
#### Layers
- Dense (`showml.deep_learning.layers.Dense`)

#### Activations
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


## Contributing
1. Fork the repository
2. Install the necessary dependencies
```
$ pip install pre-commit mypy pytest
 ```
3. Commit and push your changes to your own branch
4. Before submitting a Pull Request, run these housekeeping checks locally
  - Run pre-commit
   ```
   $ pre-commit run -a
   ```
  - Run mypy
  ```
  $ mypy .
  ```
  - Run tests
  ```
  $ pytest
  ```
5. Once everything succeeds, create a Pull Request (CI will be triggered)
