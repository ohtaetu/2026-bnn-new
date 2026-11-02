"""Repeat the runtime measurements N times on trained runs and report mean ± std.
Run it on an otherwise idle GPU after run_all.py:

  python bench_runtime.py --repeats 5
"""
import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "cuda" / ("bnn_xnor.exe" if sys.platform == "win32" else "bnn_xnor")
XNOR_RUNS = [("mnist", "mnist_xnor_t127.5"), ("fashion", "fashion_xnor_t127.5"), ("fashion", "fashion_xnor_t90")]
TORCH_RUNS = [("mnist", "fp32"), ("mnist", "bc"), ("fashion", "fp32"), ("fashion", "bc")]


def mean_std(values):
    return {"mean": st.mean(values), "std": st.stdev(values) if len(values) > 1 else 0.0, "values": values}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--runs", default="runs")
    a = p.parse_args()
    runs = ROOT / a.runs
    out = {}

    for _, name in XNOR_RUNS:
        batch, single, cpu = [], [], []
        tmp = runs / name / "cuda_repeat.json"
        for _ in range(a.repeats):
            subprocess.run([str(EXE), str(runs / name / "model.bin"), str(runs / name / "testset.bin"), str(tmp)],
                           check=True, capture_output=True)
            j = json.loads(tmp.read_text())
            batch.append(j["batch_all"]["us_per_image"])
            single.append(j["batch_1"]["ms_per_call"])
            cpu.append(j["cpu_reference_ms_per_image"])
        tmp.unlink()
        out[name] = {"engine": "CUDA XNOR-popcount kernel", "batch_us_per_image": mean_std(batch),
                     "single_image_ms": mean_std(single), "cpu_reference_ms_per_image": mean_std(cpu)}

    for ds, model in TORCH_RUNS:
        name = f"{ds}_{model}"
        batch, single = [], []
        for _ in range(a.repeats):
            subprocess.run([sys.executable, "train.py", "--dataset", ds, "--model", model, "--out", str(runs),
                            "--time-only"], cwd=ROOT, check=True, capture_output=True)
            t = json.loads((runs / name / "metrics.json").read_text())["torch_gpu_time"]
            batch.append(t["batch_all"]["us_per_image"])
            single.append(t["batch_1"]["ms_per_call"])
        out[name] = {"engine": "PyTorch (cuBLAS)", "batch_us_per_image": mean_std(batch),
                     "single_image_ms": mean_std(single)}

    (ROOT / "results" / "runtime.json").write_text(json.dumps(out, indent=2))
    for name, r in out.items():
        b, s = r["batch_us_per_image"], r["single_image_ms"]
        line = (f"{name:22s} {r['engine']:26s} batch {b['mean']:.4f} ± {b['std']:.4f} us/image   "
                f"1 image {s['mean']:.4f} ± {s['std']:.4f} ms")
        if "cpu_reference_ms_per_image" in r:
            line += f"   CPU ref {r['cpu_reference_ms_per_image']['mean']:.4f} ms/image"
        print(line)


if __name__ == "__main__":
    main()
