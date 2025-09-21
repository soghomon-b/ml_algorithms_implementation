import torch
from torch import nn

# ----- Base -----
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.forward(x)                  # (N,1)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        p = self.predict_proba(x)
        return (p >= threshold).to(torch.int64)


# ----- Logistic Regression (single linear layer) -----
class LogisticRegressionNN(Model):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = x.float()
        logits = self.linear(x)                   # (N,1)
        return logits                             # return logits, not sigmoid


# ----- 1-hidden-layer MLP for binary classification -----
class NeuralNetworkNN(Model):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.float()
        h = self.relu(self.linear(x))
        logits = self.output_layer(h)             # (N,1)
        return logits


# ----- Deeper MLP -----
class DeepNeuralNetworkNN(Model):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.model(x.float())              # logits


# ----- Dropout variant -----
class NeuralNetworkNNDropout(Model):
    def __init__(self, in_dim: int, hidden: int, p: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.output_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.float()
        h = self.dropout(self.relu(self.linear(x)))
        logits = self.output_layer(h)
        return logits


# ----- Regularization helpers -----
def l1_penalty_on_weights(model: nn.Module):
    l1 = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and "weight" in name:
            l1 = l1 + p.abs().sum()
    return l1

def decide_hs(d: int) -> int:
    if d < 4:
        return 4
    if d < 16:
        return d
    if d < 64:
        return int(1.5 * d)
    return min(int(1.5 * d), 256)
