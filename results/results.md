Test accuracy on the 10,000 test images after the last epoch (no early stopping), and inference time on NVIDIA GeForce RTX 3060.

| Dataset | Model | W/A bits | Input threshold | Accuracy (%) | Paper (%) | Memory (KB) | GPU time, 10k batch (µs/image) | GPU time, batch 1 (ms) | Engine |
|---|---|---|---|---|---|---|---|---|---|
| MNIST | BNN, XNOR-popcount | 1/1 | 127.5 | **96.43** | 80.82 | 52.7 | 0.0305 | 0.0118 | CUDA XNOR-popcount kernel |
| MNIST | BNN, BinaryConnect | 1/32 | – | **97.62** | 88.13 | 49.6 | 0.1538 | 0.1560 | PyTorch FP32 (cuBLAS) |
| MNIST | Full 32-bit | 32/32 | – | **98.32** | 96.55 | 1590.0 | 0.1417 | 0.0666 | PyTorch FP32 (cuBLAS) |
| Fashion-MNIST | BNN, XNOR-popcount | 1/1 | 127.5 | **82.11** | 67.16 | 52.7 | 0.0306 | 0.0100 | CUDA XNOR-popcount kernel |
| Fashion-MNIST | BNN, XNOR-popcount | 1/1 | 90 | **84.27** | 70.99 | 52.7 | 0.0311 | 0.0111 | CUDA XNOR-popcount kernel |
| Fashion-MNIST | BNN, BinaryConnect | 1/32 | – | **88.42** | 82.74 | 49.6 | 0.1566 | 0.1740 | PyTorch FP32 (cuBLAS) |
| Fashion-MNIST | Full 32-bit | 32/32 | – | **89.78** | 86.77 | 1590.0 | 0.1419 | 0.0547 | PyTorch FP32 (cuBLAS) |

Bit-exactness of the CUDA kernel (predictions that differ, out of 10,000):

- `mnist_xnor_t127.5`: GPU vs CPU reference 0, GPU vs Python integer path 0, integer path vs PyTorch float model 0
- `fashion_xnor_t127.5`: GPU vs CPU reference 0, GPU vs Python integer path 0, integer path vs PyTorch float model 0
- `fashion_xnor_t90`: GPU vs CPU reference 0, GPU vs Python integer path 0, integer path vs PyTorch float model 0
