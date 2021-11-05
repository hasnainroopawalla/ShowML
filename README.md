# ShowML

[![Python packaging](https://github.com/hasnainroopawalla/ShowML/actions/workflows/python_packaging.yml/badge.svg?branch=master)](https://github.com/hasnainroopawalla/ShowML/actions/workflows/python_packaging.yml)

**Show** the **ML** Code!

A Python package of Machine Learning Algorithms implemented from scratch.

The aim of this package is to present the working behind fundamental Machine Learning algorithms in a transparent and modular way.

> **_NOTE:_**  The implementations of these algorithms are not thoroughly optimized for high computational efficiency

## Installation

To install the package
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

### Algorithms
- Linear Regression (`from showml.supervised.regression import LinearRegression`)
- Logistic Regression (`from showml.supervised.regression import LogisticRegression`)

### Optimizers
- Batch Gradient Descent (`from showml.optimizers import SGD`)

### Loss Functions
- Mean Squared Error (`from showml.losses import MeanSquareError`)


## Contributing
1. Fork the repository
2. Install the necessary dependencies
```
$ pip install mypy pre-commit pytest
 ```
3. Commit and push your changes
4. Before creating a Pull Request, run these housekeeping checks locally
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
