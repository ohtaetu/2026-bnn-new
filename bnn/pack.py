"""Turn a trained XnorMLP into the bit-packed, integer-only form used by Algorithm 2.

Bit convention (Eq. 3): +1 -> 1, -1 -> 0. Element i of a vector goes to bit (i % 32) of
uint32 word (i // 32); padding bits are 0 in both operands, so they never count as a mismatch.

For n real bits, the +-1 dot product is  n - 2 * popcount(x XOR w), which is the same as
2 * popcount(XNOR(x, w)) - n of Algorithm 2 restricted to the real bits.

Hidden layer:  h = sign(BN(z)).  BN is monotonic in z, so it folds into one integer threshold
per neuron: h = +1  <=>  z >= T.  Neurons with a negative BN scale get their weight row
negated (bits inverted) so that every neuron uses the same ">= T" test.
Output layer:  logits = a * z + c  (BN folded into one scale and offset per class).
"""
import math
import struct

import numpy as np
import torch

MAGIC_MODEL = b"BNN1"
MAGIC_TEST = b"BNT1"
NEVER = 1 << 20  # threshold no dot product can reach


def pack_bits(bits):
    """bits: bool array [rows, n] -> uint32 array [rows, ceil(n/32)]."""
    rows, n = bits.shape
    words = (n + 31) // 32
    padded = np.zeros((rows, words * 32), dtype=bool)
    padded[:, :n] = bits
    return np.packbits(padded, axis=1, bitorder="little").view("<u4")


def _bn_affine(bn):
    s = torch.sqrt(bn.running_var + bn.eps)
    a = (bn.weight / s).double().cpu().numpy()
    c = (bn.bias - bn.weight * bn.running_mean / s).double().cpu().numpy()
    return a, c  # bn(z) = a * z + c


@torch.no_grad()
def fold(model, pixel_threshold):
    w1 = model.fc1.binary_weight().cpu().numpy() > 0  # [512, 784] bool, True = +1
    w2 = model.fc2.binary_weight().cpu().numpy() > 0  # [10, 512]
    a1, c1 = _bn_affine(model.bn1)
    a2, c2 = _bn_affine(model.bn2)

    t1 = np.empty(len(a1), dtype=np.int32)
    for j, (a, c) in enumerate(zip(a1, c1)):
        if a > 0:      # a*z + c >= 0  <=>  z >= -c/a
            t1[j] = math.ceil(-c / a)
        elif a < 0:    # a*z + c >= 0  <=>  z <= -c/a  <=>  (-z) >= c/a
            w1[j] = ~w1[j]
            t1[j] = math.ceil(c / a)
        else:          # constant output sign(c)
            t1[j] = -NEVER if c >= 0 else NEVER

    return {
        "n_in": w1.shape[1], "n_hidden": w1.shape[0], "n_out": w2.shape[0],
        "pixel_threshold": math.ceil(pixel_threshold),  # bit = pixel >= this (pixels are integers)
        "w1": pack_bits(w1), "t1": t1, "w2": pack_bits(w2),
        "a2": a2.astype(np.float32), "c2": c2.astype(np.float32),
    }


def pack_images(pixels, pixel_threshold):
    return pack_bits(np.asarray(pixels) >= pixel_threshold)


def popcount(x):
    return np.bitwise_count(x).astype(np.int32)


def predict(m, pixels, chunk=1000):
    """Reference integer-only inference, the exact computation the CUDA kernel performs."""
    out = []
    for i in range(0, len(pixels), chunk):
        x = pack_images(pixels[i:i + chunk], m["pixel_threshold"])            # [B, 25]
        mism = popcount(x[:, None, :] ^ m["w1"][None]).sum(-1)               # [B, 512]
        z1 = m["n_in"] - 2 * mism
        h = pack_bits(z1 >= m["t1"][None])                                     # [B, 16]
        mism2 = popcount(h[:, None, :] ^ m["w2"][None]).sum(-1)              # [B, 10]
        z2 = m["n_hidden"] - 2 * mism2
        logits = m["a2"][None] * z2.astype(np.float32) + m["c2"][None]
        out.append(logits.argmax(1))
    return np.concatenate(out).astype(np.uint8)


def footprint_bytes(m):
    return {
        "weight_bits": (m["n_in"] * m["n_hidden"] + m["n_hidden"] * m["n_out"]) // 8,
        "packed_total": m["w1"].nbytes + m["w2"].nbytes + m["t1"].nbytes + m["a2"].nbytes + m["c2"].nbytes,
    }


def save_model(m, path):
    with open(path, "wb") as f:
        f.write(MAGIC_MODEL)
        f.write(struct.pack("<6I", m["n_in"], m["n_hidden"], m["n_out"],
                            m["w1"].shape[1], m["w2"].shape[1], m["pixel_threshold"]))
        for key in ("t1", "w1", "w2", "a2", "c2"):
            f.write(np.ascontiguousarray(m[key]).tobytes())


def load_model(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC_MODEL, f"{path}: not a BNN1 model file"
        n_in, n_hidden, n_out, w_in, w_h, pixel_threshold = struct.unpack("<6I", f.read(24))

        def take(dtype, count):
            return np.frombuffer(f.read(np.dtype(dtype).itemsize * count), dtype=dtype)

        t1 = take("<i4", n_hidden)
        w1 = take("<u4", n_hidden * w_in).reshape(n_hidden, w_in)
        w2 = take("<u4", n_out * w_h).reshape(n_out, w_h)
        a2, c2 = take("<f4", n_out), take("<f4", n_out)
    return {"n_in": n_in, "n_hidden": n_hidden, "n_out": n_out, "pixel_threshold": pixel_threshold,
            "w1": w1, "t1": t1, "w2": w2, "a2": a2, "c2": c2}


def save_testset(pixels, labels, expected, path):
    pixels = np.asarray(pixels, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(MAGIC_TEST)
        f.write(struct.pack("<2I", pixels.shape[0], pixels.shape[1]))
        f.write(pixels.tobytes())
        f.write(np.asarray(labels, dtype=np.uint8).tobytes())
        f.write(np.asarray(expected, dtype=np.uint8).tobytes())
