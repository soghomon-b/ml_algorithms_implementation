# ml-algorithms-implementation

This repository provides implementations of core machine learning algorithms and neural network architectures. The goal is to construct algorithms from first principles using both NumPy and PyTorch, in order to build an educational resource and a reliable codebase with unit tests.

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

Tests live under `test/` and validate correctness of each algorithm. Additional tests are being written accordingly when time permits.

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
- **Simple NNs**  
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

All modules will be covered with pytest test suites, tests are being added when time permits

Run tests with:
```bash
pytest test
```

Additional tests are being implemented gradually as time allows.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/ml-algorithms-implementation.git
```

Dependencies:
- Python 3.9+  
- numpy  
- torch  
- pytest (for testing)

## Usage

All algorithms expect inputs and targets as `numpy.ndarray` objects. For neural network modules implemented with PyTorch, tensors are internally converted to appropriate floating point types where necessary.

## Goals

- Build core machine learning algorithms from scratch for educational purposes  
- Provide PyTorch equivalents for neural network architectures  
- Ensure correctness through unit testing  
- Extend into advanced deep learning architectures including CNNs, RNNs, and Transformers with fine-tuning  

## License

MIT License © 2025
