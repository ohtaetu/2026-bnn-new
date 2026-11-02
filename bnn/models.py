"""784-512-10 networks from the paper: FP32, BinaryConnect and XNOR-popcount BNN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """sign(x) with +1 for x >= 0 (Eq. 2). Backward: straight-through, cancelled for |x| > 1."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1).to(grad.dtype)


sign_ste = SignSTE.apply


class BinaryLinear(nn.Linear):
    """Linear layer whose forward pass uses sign(W). The real-valued W is kept for the update
    and clipped to [-1, 1] after every step (see clip_latent_weights)."""

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        nn.init.uniform_(self.weight, -1.0, 1.0)
        self.frozen = None

    def forward(self, x):
        if self.frozen is not None:
            return F.linear(x, self.frozen)
        return F.linear(x, sign_ste(self.weight))

    def binary_weight(self):
        return torch.where(self.weight >= 0, 1.0, -1.0)

    def freeze(self):
        """Binarize once for inference timing instead of on every forward pass."""
        self.frozen = self.binary_weight().detach()


class FP32MLP(nn.Module):
    """Fig. 2(a): 32-bit weights and activations, sigmoid hidden layer."""

    def __init__(self, n_in=784, n_hidden=512, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        return self.fc2(torch.sigmoid(self.fc1(x)))


class BinaryConnectMLP(nn.Module):
    """Fig. 2(b): 1-bit weights, 32-bit activations."""

    def __init__(self, n_in=784, n_hidden=512, n_out=10):
        super().__init__()
        self.fc1 = BinaryLinear(n_in, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = BinaryLinear(n_hidden, n_out)
        self.bn2 = nn.BatchNorm1d(n_out)

    def forward(self, x):
        return self.bn2(self.fc2(torch.sigmoid(self.bn1(self.fc1(x)))))


class XnorMLP(nn.Module):
    """Fig. 2(c): 1-bit weights and 1-bit activations. Input pixels are binarized with a fixed
    threshold (127.5 or 90 in the paper), so every layer can run as XNOR-popcount."""

    def __init__(self, n_in=784, n_hidden=512, n_out=10):
        super().__init__()
        self.fc1 = BinaryLinear(n_in, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.fc2 = BinaryLinear(n_hidden, n_out)
        self.bn2 = nn.BatchNorm1d(n_out)

    def forward(self, xb):
        h = sign_ste(self.bn1(self.fc1(xb)))
        return self.bn2(self.fc2(h))


def build_model(name):
    return {"fp32": FP32MLP, "bc": BinaryConnectMLP, "xnor": XnorMLP}[name]()


@torch.no_grad()
def clip_latent_weights(model):
    for m in model.modules():
        if isinstance(m, BinaryLinear):
            m.weight.clamp_(-1.0, 1.0)
