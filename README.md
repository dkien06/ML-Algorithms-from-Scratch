# Machine Learning Algorithms from Scratch

A collection of fundamental Machine Learning algorithms implemented from scratch using Python and NumPy.

The goal of this repository is to understand the mathematical foundations and implementation details behind common Machine Learning algorithms.

## Algorithms

### 1. Clustering
- **K-Means Clustering** — Unsupervised learning algorithm for grouping data points into clusters.

### 2. Regression
- **Linear Regression** — Predicts continuous values using linear relationships between features and targets.

### 3. Classification
- **Perceptron** — A simple linear classifier based on the perceptron learning algorithm.
- **Logistic Regression** — Binary classification using the sigmoid function and cross-entropy loss.
- **Softmax Regression** — Multiclass classification using the softmax function.
- **Naive Bayes** — Probabilistic classification based on Bayes' theorem and the conditional independence assumption.

### 4. Neural Networks
- **Neural Network** — A feedforward neural network implemented with forward propagation and backpropagation.

## Project Structure

```text
ML-Algorithms-from-Scratch/
├── README.md
├── requirements.txt
│
├── clustering/
│   └── kmeans.py
│
├── regression/
│   └── linear_regression.py
│
├── classification/
│   ├── perceptron.py
│   ├── logistic_regression.py
│   ├── softmax_regression.py
│   └── naive_bayes.py
│
└── neural_network/
    └── neural_network.py
```

## Requirements

- Python 3.9+
- NumPy
- Matplotlib

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Each algorithm is implemented in a separate Python file.

For example, to run K-Means:

```bash
python clustering/kmeans.py
```

Other algorithms can be executed in the same way.

## Learning Objectives

- Understand the mathematical foundations of Machine Learning algorithms.
- Implement algorithms without relying on high-level Machine Learning libraries.
- Practice numerical computation with NumPy.
- Understand optimization methods and model training.

## Author

**Phan Duc Kien**

GitHub: [dkien06](https://github.com/dkien06)
