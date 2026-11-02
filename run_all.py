"""Reproduce Tables 2 and 3: train every configuration, run the CUDA XNOR kernel on the
XNOR models, and write results/results.md, results/summary.json and the training curves.

  python run_all.py              # full run, 100 epochs as in the paper
  python run_all.py --epochs 5   # quick check
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "cuda" / ("bnn_xnor.exe" if sys.platform == "win32" else "bnn_xnor")

CONFIGS = [  # (dataset, model, threshold, paper accuracy %)
    ("mnist", "xnor", 127.5, 80.82),
    ("mnist", "bc", None, 88.13),
    ("mnist", "fp32", None, 96.55),
    ("fashion", "xnor", 127.5, 67.16),
    ("fashion", "xnor", 90.0, 70.99),
    ("fashion", "bc", None, 82.74),
    ("fashion", "fp32", None, 86.77),
]
LABEL = {"xnor": "BNN, XNOR-popcount", "bc": "BNN, BinaryConnect", "fp32": "Full 32-bit"}
BITS = {"xnor": (1, 1), "bc": (1, 32), "fp32": (32, 32)}
COLOR = {("xnor", 127.5): "#1f77b4", ("xnor", 90.0): "#17becf", ("bc", None): "#ff7f0e", ("fp32", None): "#2ca02c"}


def run_name(ds, model, thr):
    return f"{ds}_{model}" + (f"_t{thr:g}" if model == "xnor" else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--runs", default="runs")
    p.add_argument("--skip-train", action="store_true",
                   help="reuse existing runs/ and only re-measure timings (run on an otherwise idle GPU)")
    a = p.parse_args()
    runs = ROOT / a.runs
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    if not EXE.exists():
        sys.exit(f"{EXE} not found: build it first (cuda/build.bat on Windows, see README)")

    summary = []
    for ds, model, thr, paper in CONFIGS:
        name = run_name(ds, model, thr)
        cmd = [sys.executable, "train.py", "--dataset", ds, "--model", model,
               "--epochs", str(a.epochs), "--out", str(runs)]
        if thr is not None:
            cmd += ["--threshold", str(thr)]
        if a.skip_train:
            cmd.append("--time-only")
        subprocess.run(cmd, cwd=ROOT, check=True)
        metrics = json.loads((runs / name / "metrics.json").read_text())
        row = {"run": name, "dataset": ds, "model": model, "threshold": thr, "paper_acc": paper,
               "test_acc": metrics["test_acc"], "torch_gpu_time": metrics["torch_gpu_time"],
               "fp32_param_bytes": metrics["fp32_param_bytes"], "history": metrics["history"]}
        if model == "xnor":
            cuda_json = runs / name / "cuda.json"
            subprocess.run([str(EXE), str(runs / name / "model.bin"), str(runs / name / "testset.bin"),
                            str(cuda_json)], cwd=ROOT, check=True)
            row["cuda"] = json.loads(cuda_json.read_text())
            row["packed"] = metrics["packed"]
        summary.append(row)

    (results_dir / "summary.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "history"} for r in summary], indent=2))
    write_table(summary, results_dir / "results.md")
    plot_curves(summary, results_dir / "training_curves.png")
    print((results_dir / "results.md").read_text())


def memory_kb(r):
    if r["model"] == "xnor":
        return r["packed"]["footprint_bytes"]["packed_total"] / 1024
    if r["model"] == "bc":  # 1-bit weights; BN/bias would fold into per-neuron constants
        return (784 * 512 + 512 * 10) / 8 / 1024
    return r["fp32_param_bytes"] / 1024


def write_table(summary, path):
    gpu = next((r["cuda"]["gpu"] for r in summary if "cuda" in r), "GPU")
    lines = [
        f"Test accuracy on the 10,000 test images after the last epoch (no early stopping), "
        f"and inference time on {gpu}.",
        "",
        "| Dataset | Model | W/A bits | Input threshold | Accuracy (%) | Paper (%) | Memory (KB) "
        "| GPU time, 10k batch (µs/image) | GPU time, batch 1 (ms) | Engine |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        w, act = BITS[r["model"]]
        if "cuda" in r:
            batch, single, engine = (r["cuda"]["batch_all"]["us_per_image"], r["cuda"]["batch_1"]["ms_per_call"],
                                     "CUDA XNOR-popcount kernel")
        else:
            t = r["torch_gpu_time"]
            batch, single, engine = t["batch_all"]["us_per_image"], t["batch_1"]["ms_per_call"], "PyTorch FP32 (cuBLAS)"
        thr = f"{r['threshold']:g}" if r["threshold"] is not None else "–"
        lines.append(f"| {'MNIST' if r['dataset'] == 'mnist' else 'Fashion-MNIST'} | {LABEL[r['model']]} | {w}/{act} "
                     f"| {thr} | **{r['test_acc']:.2f}** | {r['paper_acc']:.2f} | {memory_kb(r):.1f} "
                     f"| {batch:.4f} | {single:.4f} | {engine} |")
    checks = [r for r in summary if "cuda" in r]
    lines += ["", "Bit-exactness of the CUDA kernel (predictions that differ, out of 10,000):", ""]
    for r in checks:
        c = r["cuda"]
        lines.append(f"- `{r['run']}`: GPU vs CPU reference {c['mismatch_gpu_vs_cpu']}, "
                     f"GPU vs Python integer path {c['mismatch_gpu_vs_python']}, "
                     f"integer path vs PyTorch float model {r['packed']['disagreements_with_torch']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, ds, title in ((axes[0], "mnist", "MNIST"), (axes[1], "fashion", "Fashion-MNIST")):
        for r in (r for r in summary if r["dataset"] == ds):
            label = LABEL[r["model"]] + (f" (t={r['threshold']:g})" if r["threshold"] is not None else "")
            ax.plot([h["epoch"] for h in r["history"]], [h["test_acc"] for h in r["history"]], label=label,
                    color=COLOR[(r["model"], r["threshold"])])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("test accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
