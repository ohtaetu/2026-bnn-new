"""Build testset.bin for a packed model, so the CUDA program can run without training.

  python make_testset.py pretrained/mnist_xnor_t127.5/model.bin --dataset mnist
  cuda\\bnn_xnor.exe pretrained/mnist_xnor_t127.5/model.bin pretrained/mnist_xnor_t127.5/testset.bin
"""
import argparse
from pathlib import Path

from bnn import data, pack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="model.bin written by train.py")
    p.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    p.add_argument("--out", help="default: testset.bin next to the model")
    a = p.parse_args()

    m = pack.load_model(a.model)
    _, (pixels, labels) = data.load(a.dataset)
    pixels, labels = pixels.numpy(), labels.numpy()
    pred = pack.predict(m, pixels)
    out = Path(a.out) if a.out else Path(a.model).with_name("testset.bin")
    pack.save_testset(pixels, labels, pred, out)
    print(f"integer-path accuracy {100.0 * (pred == labels).mean():.2f}%  -> {out}")


if __name__ == "__main__":
    main()
