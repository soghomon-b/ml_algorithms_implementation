import torch
from torch import nn

# ----- Base -----
class Model(nn.Module):
    """
    Base binary classification model.

    Subclasses must implement `forward(x)` and return logits of shape (N, 1).
    This base class then provides:
      - `predict_proba`: applies sigmoid to logits to get probabilities in [0, 1]
      - `predict`: converts probabilities to {0, 1} using a threshold
    """
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute P(y=1 | x) as sigmoid(logits).

        Args:
            x: Input tensor of shape (N, D) or compatible with the subclass' `forward`.

        Returns:
            Tensor of shape (N, 1) with probabilities in [0, 1].
        """
        logits = self.forward(x)                  # (N, 1)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels {0, 1} from probabilities.

        Args:
            x: Input tensor of shape (N, D).
            threshold: Cutoff for deciding class 1 (default: 0.5).

        Returns:
            Tensor of shape (N, 1) with integer labels in {0, 1}.
        """
        p = self.predict_proba(x)
        return (p >= threshold).to(torch.int64)


# ----- Logistic Regression (single linear layer) -----
class LogisticRegressionNN(Model):
    """
    Logistic regression implemented as a single linear layer.

    This model learns a linear decision boundary and uses the base
    class' `predict_proba` / `predict` to convert logits to probabilities/labels.
    """

    def __init__(self, in_dim: int):
        """
        Args:
            in_dim: Number of input features D.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns logits of shape (N, 1).
        """
        x = x.float()
        logits = self.linear(x)                   # (N, 1)
        return logits


# ----- 1-hidden-layer MLP for binary classification -----
class NeuralNetworkNN(Model):
    """
    A shallow MLP with one hidden layer and ReLU non-linearity.

    Architecture:
        in_dim → Linear → ReLU → Linear → 1 logit
    """

    def __init__(self, in_dim: int, hidden: int):
        """
        Args:
            in_dim: Number of input features D.
            hidden: Size of the hidden layer.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns logits of shape (N, 1).
        """
        x = x.float()
        h = self.relu(self.linear(x))
        logits = self.output_layer(h)             # (N, 1)
        return logits


# ----- Deeper MLP -----
class DeepNeuralNetworkNN(Model):
    """
    A deeper MLP with two hidden layers and ReLU activations.

    Architecture:
        in_dim → Linear → ReLU → Linear → ReLU → Linear → 1 logit
    """

    def __init__(self, in_dim: int, hidden: int):
        """
        Args:
            in_dim: Number of input features D.
            hidden: Size of each hidden layer.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns logits of shape (N, 1).
        """
        return self.model(x.float())


# ----- Dropout variant -----
class NeuralNetworkDropout(Model):
    """
    1-hidden-layer MLP with dropout regularization.

    Architecture:
        in_dim → Linear → ReLU → Dropout(p) → Linear → 1 logit

    Dropout is active only in training mode (`model.train()`).
    """

    def __init__(self, in_dim: int, hidden: int, p: float = 0.5):
        """
        Args:
            in_dim: Number of input features D.
            hidden: Size of the hidden layer.
            p: Dropout probability (fraction of units to drop).
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns logits of shape (N, 1).
        """
        x = x.float()
        h = self.dropout(self.relu(self.linear(x)))
        logits = self.output_layer(h)
        return logits


# ----- Regularization helpers -----
def l1_penalty_on_weights(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 penalty over all weight parameters of a model.

    This is useful for L1 regularization (encouraging sparsity in weights).

    Args:
        model: Any nn.Module whose weights we want to regularize.

    Returns:
        A scalar tensor equal to the sum of |w| over all weight tensors.
    """
    l1 = torch.tensor(0.0, device=next(model.parameters(), torch.tensor(0.)).device)
    for name, p in model.named_parameters():
        if p.requires_grad and "weight" in name:
            l1 = l1 + p.abs().sum()
    return l1


def decide_hs(d: int) -> int:
    """
    Heuristic to choose a hidden size based on the input dimension.

    Args:
        d: Input dimension (number of features).

    Returns:
        A reasonable hidden size, capped at 256, that grows with d but
        does not explode for large d.
    """
    if d < 4:
        return 4
    if d < 16:
        return d
    if d < 64:
        return int(1.5 * d)
    return min(int(1.5 * d), 256)