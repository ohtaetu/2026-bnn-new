"""Train one configuration from the paper and save metrics (and, for XNOR, the packed model).

  python train.py --dataset mnist   --model fp32
  python train.py --dataset mnist   --model bc
  python train.py --dataset mnist   --model xnor --threshold 127.5
  python train.py --dataset fashion --model xnor --threshold 90
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bnn import data, pack
from bnn.models import BinaryLinear, build_model, clip_latent_weights


def run_name(a):
    return f"{a.dataset}_{a.model}" + (f"_t{a.threshold:g}" if a.model == "xnor" else "")


@torch.no_grad()
def evaluate(model, x, y, batch=10000):
    model.eval()
    correct = sum((model(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum().item()
                  for i in range(0, len(x), batch))
    return 100.0 * correct / len(x)


@torch.no_grad()
def time_torch(model, x, reps=50):
    """GPU forward time: whole test set as one batch, and batch size 1."""
    model.eval()
    for m in model.modules():
        if isinstance(m, BinaryLinear):
            m.freeze()
    out = {}
    for name, inp, n in (("batch_all", x, len(x)), ("batch_1", x[:1], 1)):
        for _ in range(10):
            model(inp)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            model(inp)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / reps
        out[name] = {"ms_per_call": ms, "us_per_image": 1000.0 * ms / n}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    p.add_argument("--model", choices=["fp32", "bc", "xnor"], required=True)
    p.add_argument("--threshold", type=float, default=127.5, help="input binarization threshold (xnor)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--time-only", action="store_true",
                   help="load the saved model.pt and only re-measure torch_gpu_time")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda")
    out = Path(a.out) / run_name(a)
    out.mkdir(parents=True, exist_ok=True)

    (xtr_px, ytr), (xte_px, yte) = data.load(a.dataset)
    xtr = data.to_input(xtr_px, a.model, a.dataset, a.threshold).to(dev)
    xte = data.to_input(xte_px, a.model, a.dataset, a.threshold).to(dev)
    ytr, yte = ytr.to(dev), yte.to(dev)

    model = build_model(a.model).to(dev)
    if a.time_only:
        model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
        result = json.loads((out / "metrics.json").read_text())
        result["torch_gpu_time"] = time_torch(model, xte)
        (out / "metrics.json").write_text(json.dumps(result, indent=2))
        print(f"[{run_name(a)}] re-timed: {result['torch_gpu_time']}")
        return

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps_per_epoch = len(xtr) // a.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps_per_epoch)

    history = []
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        total = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * a.batch:(s + 1) * a.batch]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if a.model != "fp32":
                clip_latent_weights(model)
            total += loss.item()
        tr_acc, te_acc = evaluate(model, xtr, ytr), evaluate(model, xte, yte)
        history.append({"epoch": epoch, "loss": total / steps_per_epoch, "train_acc": tr_acc, "test_acc": te_acc})
        if epoch == 1 or epoch % 10 == 0:
            print(f"[{run_name(a)}] epoch {epoch:3d}  loss {total / steps_per_epoch:.4f}  "
                  f"train {tr_acc:.2f}%  test {te_acc:.2f}%")

    result = {
        "run": run_name(a), "args": vars(a), "train_seconds": time.time() - t0,
        "test_acc": history[-1]["test_acc"], "history": history,
        "torch_gpu_time": time_torch(model, xte),
        "fp32_param_bytes": 4 * sum(t.numel() for t in model.parameters()),
    }

    if a.model == "xnor":
        m = pack.fold(model, a.threshold)
        pred = pack.predict(m, xte_px.numpy())
        int_acc = 100.0 * float((pred == yte.cpu().numpy()).mean())
        with torch.no_grad():
            torch_pred = model(xte).argmax(1).cpu().numpy()
        result["packed"] = {
            "integer_path_test_acc": int_acc,
            "disagreements_with_torch": int((pred != torch_pred).sum()),
            "footprint_bytes": pack.footprint_bytes(m),
            "pixel_threshold_int": m["pixel_threshold"],
        }
        pack.save_model(m, out / "model.bin")
        pack.save_testset(xte_px.numpy(), yte.cpu().numpy(), pred, out / "testset.bin")
        print(f"[{run_name(a)}] integer-only path: {int_acc:.2f}%  "
              f"(differs from PyTorch on {result['packed']['disagreements_with_torch']} images)")

    torch.save(model.state_dict(), out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[{run_name(a)}] final test accuracy {result['test_acc']:.2f}%  -> {out}")


if __name__ == "__main__":
    main()
