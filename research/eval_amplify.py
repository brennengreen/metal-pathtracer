"""Rigorous, assumption-free measurement of the hypothesis:

    "Cast one ray + inference  ==  how many physically path-traced samples?"

Instead of assuming Monte-Carlo error scales as 1/sqrt(N), we *measure* the real
convergence curve. For a held-out camera view the Metal tracer emits M independent
single-sample (1 spp) images plus an independent high-spp reference. Averaging the
first K of those images is exactly a K-spp path-traced estimate, so relMSE(K) vs
the reference is the true convergence curve. We then locate the K at which plain
path tracing reaches the neural amplifier's error: that K is the directly measured
effective sample count of "one ray + inference".

Run (after train.py):
  python3 research/eval_amplify.py --stack research/stack --model research/results
"""
import argparse, json, os
import numpy as np
import torch

from model import Amplifier, GEOM_COLS, RAD_COLS


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def tonemap_u8(img):
    return (srgb(aces(img)) * 255 + 0.5).astype(np.uint8)


def rel_mse(pred, ref, mask, eps=1e-2):
    p, r = pred[mask], ref[mask]
    return float(np.mean((p - r) ** 2 / (r ** 2 + eps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="research/stack")
    ap.add_argument("--model", default="research/results")
    ap.add_argument("--out", default="research/results")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.stack, "stack_meta.json")))
    W, H, M = meta["width"], meta["height"], meta["stackM"]
    hw = W * H
    ref = np.fromfile(os.path.join(args.stack, "ref.bin"), dtype=np.float32).reshape(hw, 3)
    stack = np.fromfile(os.path.join(args.stack, "stack.bin"), dtype=np.float32).reshape(M, hw, 3)
    feats = np.fromfile(os.path.join(args.stack, "feats.bin"), dtype=np.float32).reshape(hw, 21)
    hit = feats[:, 0] > 0.5
    print(f"view={meta['view']} scene={meta['scene']} {W}x{H}  M={M}  refSpp={meta['refSpp']}  "
          f"shaded px={int(hit.sum())}")

    # --- neural amplifier prediction on this exact held-out view ------------
    ckpt = torch.load(os.path.join(args.model, "model.pt"), map_location="cpu")
    z = np.load(os.path.join(args.model, "standardize.npz"))
    mean, std = z["mean"], z["std"]
    Xp = feats.copy()
    Xp[:, RAD_COLS] = np.log1p(np.maximum(Xp[:, RAD_COLS], 0.0))
    Xp[:, GEOM_COLS] = (Xp[:, GEOM_COLS] - mean) / std
    model = Amplifier(num_bands=ckpt["bands"], hidden=ckpt["hidden"], depth=ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    with torch.no_grad():
        pred = torch.expm1(model(torch.tensor(Xp))).clamp(min=0).numpy()
    neural_rel = rel_mse(pred, ref, hit)

    # --- true Monte-Carlo convergence curve --------------------------------
    Ks, rels = [], []
    cum = np.zeros((hw, 3), dtype=np.float64)
    for k in range(1, M + 1):
        cum += stack[k - 1]
        if k & (k - 1) == 0 or k == M:           # powers of two (+ last)
            est = (cum / k).astype(np.float32)
            Ks.append(k); rels.append(rel_mse(est, ref, hit))
    Ks, rels = np.array(Ks), np.array(rels)

    # Effective spp = where the MC curve crosses the neural relMSE (log-log interp).
    eff = float("inf")
    for i in range(1, len(Ks)):
        if rels[i] <= neural_rel <= rels[i - 1]:
            lx0, lx1 = np.log(Ks[i - 1]), np.log(Ks[i])
            ly0, ly1 = np.log(rels[i - 1]), np.log(rels[i])
            eff = float(np.exp(lx0 + (np.log(neural_rel) - ly0) * (lx1 - lx0) / (ly1 - ly0)))
            break
    if rels[-1] > neural_rel:
        # Neural beats even the largest measured K — extrapolate via 1/K slope.
        eff = float(Ks[-1] * rels[-1] / neural_rel)
        eff_note = f">= {Ks[-1]} (extrapolated)"
    else:
        eff_note = f"{eff:.1f}"

    out = {
        "view": meta["view"], "scene": meta["scene"],
        "neural_relMSE": neural_rel,
        "mc_curve": {"K": Ks.tolist(), "relMSE": rels.tolist()},
        "one_spp_relMSE": float(rels[0]),
        "effective_spp": eff, "effective_spp_note": eff_note,
    }
    json.dump(out, open(os.path.join(args.out, "amplify.json"), "w"), indent=2)

    # --- plot the curve ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.loglog(Ks, rels, "o-", label="path tracing (measured)", color="#1f77b4")
        plt.axhline(neural_rel, color="#d62728", ls="--",
                    label=f"neural amplifier (1 cast ray)\nrelMSE={neural_rel:.4f}")
        if np.isfinite(eff):
            plt.axvline(eff, color="#2ca02c", ls=":", label=f"effective spp ≈ {eff:.0f}")
        plt.xlabel("samples per pixel (path tracing)")
        plt.ylabel("relative MSE vs reference (shaded px)")
        plt.title(f"Effective sample amplification — held-out view {meta['view']} ({meta['scene']})")
        plt.legend(); plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "convergence.png"), dpi=120)
        print("wrote convergence.png")
    except Exception as e:
        print("matplotlib unavailable, skipping plot:", e)

    print("\n============ DIRECTLY MEASURED EFFECTIVE SAMPLES ============")
    print(f"  1 spp path tracing      relMSE = {rels[0]:.4f}")
    print(f"  {Ks[-1]:>3d} spp path tracing    relMSE = {rels[-1]:.4f}")
    print(f"  neural (1 cast ray)     relMSE = {neural_rel:.4f}")
    print(f"  >> one ray + inference  ≈  {eff_note} path-traced samples")
    print("============================================================")


if __name__ == "__main__":
    main()
