"""MNIST / Fashion-MNIST as uint8 tensors, kept whole on the GPU (both sets are small)."""
import torch
from torchvision import datasets

MEAN_STD = {"mnist": (0.1307, 0.3081), "fashion": (0.2860, 0.3530)}


def load(name, root="data"):
    cls = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}[name]
    train = cls(root, train=True, download=True)
    test = cls(root, train=False, download=True)
    return (train.data.reshape(-1, 784), train.targets), (test.data.reshape(-1, 784), test.targets)


def binarize(pixels, threshold):
    """Map uint8 pixels to +1 (pixel >= threshold) or -1."""
    return torch.where(pixels.float() >= threshold, 1.0, -1.0)


def to_input(pixels, model_name, dataset, threshold):
    if model_name == "xnor":
        return binarize(pixels, threshold)
    mean, std = MEAN_STD[dataset]
    return (pixels.float() / 255.0 - mean) / std
