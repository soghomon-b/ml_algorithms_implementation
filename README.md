# ml-algorithms-implementation

This is my personal space to learn different ML algorithms by specifying, implementing, and testing them.
## Project Structure

```
src/
├── fine_tuning/
│   ├── imag_transformer_finetune.py   # fine-tuning for vision transformers
│   └── llm_finetuning.py              # fine-tuning for large language models
│
├── general_algorithms/
│   ├── linear_regression.py           # Linear regression (closed-form, GD, SGD) (tested)
│   └── logistic_regression.py         # Logistic regression (GD, SGD) (tested)
│
└── neural_networks/
    ├── simple_NNs.py                  # Logistic regression NN, MLPs, regularized NNs
    ├── CNN.py                         # Convolutional Neural Networks
    ├── RNN.py                         # Recurrent Neural Networks
    └── GPT.py                         # Transformer decoder (GPT-like)
```

Tests live under `test/` and validate the correctness of each algorithm. Additional tests are being written accordingly when time permits.

## Features

### General Algorithms
- **Linear Regression** (tested)  
  - Closed-form solution (Normal Equation / `np.linalg.lstsq`)  
  - Gradient Descent (GD)  
  - Stochastic Gradient Descent (SGD)

- **Logistic Regression** (tested)  
  - Binary classification with sigmoid activation  
  - Cross-entropy loss, GD and SGD variants

### Neural Networks
- **Simple NNs**  (tested)
  - Logistic Regression (PyTorch)  
  - Single-layer MLP  
  - Deep Neural Networks (DNNs)  
  - Variants with L1, L2, ElasticNet regularization and Dropout

- **CNNs**: Convolutional layers, pooling, activation pipelines  
- **RNNs**: Basic recurrent architectures for sequence data  
- **GPT**: Decoder-only transformer blocks

### Fine-tuning
- **LLM fine-tuning**: pipelines for adapting pre-trained LLMs  
- **Vision Transformer fine-tuning**: adapting image transformers on downstream tasks  

## Tests

All modules will be covered with pytest test suites, and tests are being added when time permits

## Status

- ✅ Linear & logistic regression (NumPy): implemented and tested
- ✅ Simple neural networks (MLPs, regularization helpers): implemented, tests in progress
- ⚠️ CNN / RNN / GPT modules: experimental / work in progress
- 🚧 Fine-tuning scripts: initial prototypes for future expansion

## Installation

You are welcome to use this repo for ethical and scientific purposes. Clone the repository by running;

```bash
git clone https://github.com/soghomon-n/ml-algorithms-implementation.git
```

Dependencies:
- Python 3.9+  
- numpy  
- torch  
- pytest (for testing)

## Usage

All algorithms expect inputs and targets as `numpy.ndarray` objects. For neural network modules implemented with PyTorch, tensors are internally converted to appropriate floating-point types where necessary.

## Example: Linear Regression (GD)

```python
from src.general_algorithms.linear_regression import LinearRegressionGD
import numpy as np

X = np.random.randn(200, 3)
w_true = np.array([1.5, -2.0, 0.7])
y = X @ w_true + 0.1 * np.random.randn(200)

model = LinearRegressionGD(lr=0.01, n_iters=1000)
model.fit(X, y)

print("Learned weights:", model.w)
print("MSE:", model.mse(X, y))
```
## Goals

- Build core machine learning algorithms from scratch to help me understand them more. 
- Provide PyTorch equivalents for neural network architectures.
- Ensure correctness through unit testing.
- Extend into advanced deep learning architectures, including CNNs, RNNs, and Transformers with fine-tuning.

## License

MIT License © 2025
