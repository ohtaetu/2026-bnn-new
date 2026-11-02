# Binary Neural Network with XNOR-Popcount Inference on CUDA

[![DOI](https://img.shields.io/badge/DOI-10.54644%2Fjte.2025.1756-blue)](https://doi.org/10.54644/jte.2025.1756)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-sm__86%20(RTX%203060)-76B900)

A from-scratch reimplementation of the binary neural network (BNN) from our paper
**"Optimizing Binary Neural Network for Resource-Constrained Edge Devices in IoT Applications"**
(*Journal of Technical Education Science*, vol. 21, no. 1, 2026). The original code is no longer
available, so this repository (`-new`) rebuilds it from the paper and improves it: training with 1-bit
weights and activations using batch normalization, Adam and a cross-entropy loss, then inference with the
bit-packing + XNOR-popcount scheme of the paper's Algorithm 2, running as a fused CUDA kernel on an
NVIDIA GeForce RTX 3060. The fully binary network now reaches **96.43%** on MNIST (80.82% in the paper).

> [!IMPORTANT]
> **If you use this code, please cite the paper:**
>
> V. M. Nguyen, T. T. Ngo, T. D. Tran, M. T. Nguyen, and M. H. Vo, "Optimizing binary neural network for
> resource-constrained edge devices in IoT applications," *Journal of Technical Education Science*,
> vol. 21, no. 1, pp. 12–22, Feb. 2026, doi: [10.54644/jte.2025.1756](https://doi.org/10.54644/jte.2025.1756).
>
> BibTeX is in [How to cite](#how-to-cite). GitHub's **"Cite this repository"** button exports it too.

## Results

Test accuracy on the 10,000 test images after the last of 100 epochs (no early stopping, single seed),
and inference time on an NVIDIA GeForce RTX 3060. The "Original paper" column is Tables 2 and 3 of the
paper. This version is retrained with an improved recipe (see [What is new](#what-is-new-compared-with-the-original-paper)), which is why every number is higher.

| Dataset | Model | W/A bits | Input threshold | **This version (%)** | Original paper (%) | Memory (KB) | GPU, 10k images (µs/image) | GPU, 1 image (ms) |
|---|---|---|---|---|---|---|---|---|
| MNIST | BNN, XNOR-popcount | 1/1 | 127.5 | **96.43** | 80.82 | 52.7 | **0.0305** | **0.0118** |
| MNIST | BNN, BinaryConnect | 1/32 | – | **97.62** | 88.13 | 49.6 | 0.1538 | 0.1560 |
| MNIST | Full 32-bit | 32/32 | – | **98.32** | 96.55 | 1590.0 | 0.1417 | 0.0666 |
| Fashion-MNIST | BNN, XNOR-popcount | 1/1 | 127.5 | **82.11** | 67.16 | 52.7 | **0.0306** | **0.0100** |
| Fashion-MNIST | BNN, XNOR-popcount | 1/1 | 90 | **84.27** | 70.99 | 52.7 | **0.0311** | **0.0111** |
| Fashion-MNIST | BNN, BinaryConnect | 1/32 | – | **88.42** | 82.74 | 49.6 | 0.1566 | 0.1740 |
| Fashion-MNIST | Full 32-bit | 32/32 | – | **89.78** | 86.77 | 1590.0 | 0.1419 | 0.0547 |

- **Accuracy:** the fully binary XNOR-popcount network reaches **96.43%** on MNIST (80.82% in the paper)
  and **84.27%** on Fashion-MNIST with threshold 90 (70.99% in the paper). As in the paper, threshold 90
  works better than 127.5 on Fashion-MNIST.
- **Speed:** the XNOR models run on the fused CUDA kernel of this repository: about **0.03 µs per image**
  (≈ 32 million images/s) on the whole test set, about **4.6× faster** than the 32-bit model in PyTorch
  (cuBLAS) on the same GPU, and 0.01 ms for a single image versus 0.05–0.07 ms.
- **Memory:** 52.7 KB for the packed XNOR model, including padding, thresholds and output scales,
  versus 1590 KB for the 32-bit model (about 30× smaller), in line with the paper's 51 KB vs 1587 KB.
- **Bit-exact:** on every XNOR model, the GPU kernel, the C++ CPU reference, the NumPy integer path and
  the PyTorch model give the same prediction on all 10,000 test images.

Timing notes: 32-bit and BinaryConnect numbers are PyTorch eager-mode forward passes; XNOR numbers are
one launch of the fused CUDA kernel, input bit-packing included, host-to-device copies excluded.
Single-image times are dominated by kernel-launch overhead. BinaryConnect accuracy fluctuates by a few
points between epochs (see the curves), so its last-epoch value is less stable than the others.

![Test accuracy per epoch](results/training_curves.png)

Raw numbers: [results/summary.json](results/summary.json).

## What the code does

### Network and training

The three networks of the paper (Fig. 2), all **784–512–10** fully connected:

| Model | Weights | Activations | Input |
|---|---|---|---|
| Full 32-bit | 32-bit | 32-bit (sigmoid) | normalized pixels |
| BinaryConnect | 1-bit (±1) | 32-bit (sigmoid) | normalized pixels |
| XNOR-popcount | 1-bit (±1) | 1-bit (±1) | pixels binarized at a threshold (127.5 or 90) |

Binary layers keep a real-valued *latent* weight for the optimizer and use `sign(W)` in the forward
pass. The gradient of `sign` is the straight-through estimator (STE): it passes through unchanged for
|x| ≤ 1 and is zero outside (Section 2.3 of the paper). Latent weights are clipped to [−1, 1] after every
step. Batch normalization after each binary layer (not used in the original paper) stabilizes training
and costs nothing at inference, because it folds into the integer thresholds described below.

### From a trained model to integer-only inference

`bnn/pack.py` turns the trained XNOR model into the form Algorithm 2 needs:

- **Bit packing (Eq. 3):** +1 → bit 1, −1 → bit 0. Element *i* goes to bit *i* mod 32 of `uint32` word ⌊*i*/32⌋.
  A 784-pixel image becomes 25 words; a 512-neuron hidden layer becomes 16 words.
- **XNOR-popcount:** for *n* real bits, the ±1 dot product is `n − 2·popcount(x XOR w)`, which equals
  `2·popcount(XNOR(x, w)) − n` from Algorithm 2. Padding bits are 0 in both operands, so they never count.
- **BatchNorm folding:** `sign(BN(z))` is monotonic in the integer `z`, so it becomes a single integer
  comparison `z ≥ T` per neuron. Neurons with a negative BN scale get their weight row inverted so that
  every neuron uses the same `≥` test. The output layer keeps one float scale and offset per class.

The hidden layer therefore needs no floating point at all, and the whole model is about 53 KB.
`pack.predict` is a NumPy reference of this integer-only path; it agrees with the PyTorch model on
every test image (see the bit-exactness list above).

### CUDA kernel (`cuda/bnn_xnor.cu`)

One fused kernel goes from raw `uint8` pixels to class labels. A block has 512 threads, one per hidden neuron:

1. **Bit packing on the GPU:** each of the 32 lanes in a warp tests one pixel against the threshold, and
   `__ballot_sync` packs the 32 results into one `uint32` word in a single instruction.
2. **Hidden layer:** each thread holds its neuron's 25 packed weight words in registers and computes
   `z = 784 − 2·Σ __popc(x[k] ^ w[k])`. Each XOR + `__popc` covers 32 synapses at once (SIMD within a register).
3. **Sign and re-packing:** `z ≥ T` is packed again with `__ballot_sync`, so warp *k* writes hidden word *k* directly.
4. **Output layer:** lanes 0–9 of warp 0 do the 512-bit XNOR-popcount for the 10 classes, apply the folded
   BatchNorm, and a warp-shuffle reduction picks the arg-max.

Weights are loaded once per block and reused for every image the block processes (4 images per step).
The program checks its predictions against a CPU reference and against the Python integer path, and
reports throughput for the whole test set and latency for a single image.

## What is new compared with the original paper

This repository is a **new, improved implementation**, not a copy of the original code. The network
(784–512–10), the binarization (sign + STE), the input thresholds and the XNOR-popcount inference of
Algorithm 2 are the same as in the paper. The **training recipe is new**:

| | Original paper | This version |
|---|---|---|
| Loss | mean squared error (Algorithm 1) | **cross-entropy** |
| Optimizer | plain gradient descent, learning rate 0.1, no further tuning | **Adam**, learning rate 1e-3, **cosine learning-rate decay** |
| Batch normalization | **not used** | **after every binary layer**, folded into integer thresholds for inference |
| Weight update | weights re-binarized right after every update (Algorithm 1) | real-valued latent weights kept for the optimizer, clipped to [−1, 1], binarized in the forward pass |
| Epochs / batch size | 100 / 200 | 100 / 200 (unchanged) |
| Inference | Python on an Intel Core i5-1035G1 CPU | fused CUDA kernel on an NVIDIA GeForce RTX 3060 GPU |

**Why the paper's numbers are lower.** Without batch normalization, with an MSE loss and with weights
re-binarized after every step, the original BNNs trained poorly: 80.82% on MNIST and 67.16–70.99% on
Fashion-MNIST for the fully binary network. Re-binarizing after every step also explains the jagged
training curves in the paper's Figures 3 and 4. With the recipe above, the same 784–512–10 binary
network reaches 96.43% and 84.27%. The paper's results show what the original training achieved; they
are not a limit of the BNN method itself.

Timings are not directly comparable with the paper's either: the paper measured on a laptop CPU, this
version on a GPU.

## Quick start

Requirements: an NVIDIA GPU with CUDA, the CUDA Toolkit (`nvcc`), Python 3.10+ with PyTorch.
Tested on Windows 11, RTX 3060 (sm_86), CUDA 12.6, PyTorch 2.8, Visual Studio 2022.

```bash
pip install -r requirements.txt

# build the CUDA program
cuda\build.bat                                   # Windows (finds MSVC automatically)
nvcc -O3 -std=c++17 -arch=sm_86 -o cuda/bnn_xnor cuda/bnn_xnor.cu   # Linux

# reproduce everything: 7 trainings x 100 epochs + CUDA inference, writes results/
python run_all.py

# re-measure all timings without retraining (keep the GPU otherwise idle)
python run_all.py --skip-train
```

For another GPU, change `-arch` (for example `sm_75`, `sm_89`), or on Windows run
`set CUDA_ARCH=sm_89` before `cuda\build.bat`.

### Run the CUDA kernel with the pretrained models

`pretrained/` contains the three packed XNOR models from the table above (about 53 KB each):

```bash
python make_testset.py pretrained/mnist_xnor_t127.5/model.bin --dataset mnist
cuda\bnn_xnor.exe pretrained/mnist_xnor_t127.5/model.bin pretrained/mnist_xnor_t127.5/testset.bin
```

### Train a single configuration

```bash
python train.py --dataset mnist   --model fp32
python train.py --dataset mnist   --model bc
python train.py --dataset fashion --model xnor --threshold 90
```

Each run writes `runs/<name>/metrics.json` (accuracy per epoch, GPU timings) and `model.pt`; XNOR runs
also write `model.bin` and `testset.bin` for the CUDA program.

## Repository layout

```
bnn/models.py        FP32, BinaryConnect and XNOR networks; sign with STE; binary linear layer
bnn/data.py          MNIST / Fashion-MNIST loading and input binarization
bnn/pack.py          bit packing, BatchNorm folding, NumPy integer-only reference, file formats
train.py             train one configuration, export packed model
run_all.py           reproduce Tables 2 and 3, write results/
make_testset.py      build testset.bin for a packed model
cuda/bnn_xnor.cu     fused XNOR-popcount kernel, CPU reference, benchmark
cuda/build.bat       Windows build script
pretrained/          packed XNOR models (model.bin)
results/             results table, summary.json, training curves
```

## How to cite

If this code helps your work, please cite the paper:

```bibtex
@article{Nguyen2026OptimizingBNN,
  title     = {Optimizing Binary Neural Network for Resource-Constrained Edge Devices in IoT Applications},
  author    = {Nguyen, Van Minh and Ngo, Tien Tu and Tran, Tien Dung and Nguyen, Minh Tam and Vo, Minh Huan},
  journal   = {Journal of Technical Education Science},
  volume    = {21},
  number    = {1},
  pages     = {12--22},
  month     = feb,
  year      = {2026},
  issn      = {2615-9740},
  doi       = {10.54644/jte.2025.1756},
  url       = {https://doi.org/10.54644/jte.2025.1756},
  publisher = {Ho Chi Minh City University of Technology and Engineering}
}
```

**IEEE:** V. M. Nguyen, T. T. Ngo, T. D. Tran, M. T. Nguyen, and M. H. Vo, "Optimizing binary neural network
for resource-constrained edge devices in IoT applications," *Journal of Technical Education Science*,
vol. 21, no. 1, pp. 12–22, Feb. 2026, doi: 10.54644/jte.2025.1756.

**APA:** Nguyen, V. M., Ngo, T. T., Tran, T. D., Nguyen, M. T., & Vo, M. H. (2026). Optimizing binary neural
network for resource-constrained edge devices in IoT applications. *Journal of Technical Education Science*,
*21*(1), 12–22. https://doi.org/10.54644/jte.2025.1756

The paper is open access under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/):
[full text (PDF)](https://ohtaetu.github.io/papers/nguyen2026_bnn_jte.pdf).

## Author

Tu Tien Ngo, School of Electrical Engineering, Kookmin University · [ohtaetu.github.io](https://ohtaetu.github.io/)

## License

Code: [MIT](LICENSE). The paper itself is published by Ho Chi Minh City University of Technology and
Engineering under CC BY-NC 4.0.
