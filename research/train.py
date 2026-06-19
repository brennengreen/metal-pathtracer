"""Train + evaluate the neural ray amplifier, and quantify the hypothesis:

    "Cast one ray, get many samples."

We compare, on *held-out camera views*, the error of:
  (a) the single physically-cast ray  (1 spp)      — the baseline
  (b) the neural amplifier's prediction from that single ray
against the high-spp path-traced reference.

Because Monte-Carlo error falls as 1/sqrt(N), the ratio of baseline error to
neural error squared estimates the "effective sample amplification": how many
real samples the one ray + inference is worth.

Run:  python3 research/train.py --data research/data --out research/results
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from model import Amplifier, RadianceCache, build_cache_features, GEOM_COLS, RAD_COLS


def load_dataset(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    n, d = meta["n"], meta["inDim"]
    X = np.fromfile(os.path.join(path, "X.bin"), dtype=np.float32).reshape(n, d)
    Y = np.fromfile(os.path.join(path, "Y.bin"), dtype=np.float32).reshape(n, 3)
    return meta, X, Y


def view_mask(meta, views):
    per = meta["perView"]
    idx = np.zeros(meta["n"], dtype=bool)
    for v in views:
        idx[v * per:(v + 1) * per] = True
    return idx


# --- tonemapping (matches the Metal resolve kernel) -------------------------
def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def tonemap_u8(img):
    return (srgb(aces(img)) * 255 + 0.5).astype(np.uint8)


def preprocess(X, mean, std):
    """log-transform radiance columns; standardize geometric columns."""
    Xp = X.copy()
    Xp[:, RAD_COLS] = np.log1p(np.maximum(Xp[:, RAD_COLS], 0.0))
    Xp[:, GEOM_COLS] = (Xp[:, GEOM_COLS] - mean) / std
    return Xp


def metrics(pred, ref, eps=1e-2):
    """Relative MSE (HDR-aware) and tonemapped PSNR."""
    rmse = float(np.mean((pred - ref) ** 2) / (np.mean(ref ** 2) + eps))
    rel = float(np.mean((pred - ref) ** 2 / (ref ** 2 + eps)))
    mse_tm = np.mean((tonemap_u8(pred) / 255.0 - tonemap_u8(ref) / 255.0) ** 2)
    psnr = float(10 * np.log10(1.0 / max(mse_tm, 1e-12)))
    return {"relMSE": rel, "normMSE": rmse, "psnr_tonemapped": psnr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data")
    ap.add_argument("--out", default="research/results")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--model", default="amplifier", choices=["amplifier", "cache"],
                    help="amplifier = denoise the 1-spp sample (residual); "
                         "cache = predict converged radiance from ray+geometry+material "
                         "alone (no traced sample, no recursion).")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)

    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    meta, X, Y = load_dataset(args.data)
    W, H = meta["width"], meta["height"]
    tr = view_mask(meta, meta["trainViews"])
    va = view_mask(meta, meta["valViews"])
    print(f"device={dev}  model={args.model}  samples={meta['n']}  "
          f"train={tr.sum()}  val={va.sum()}  "
          f"trainViews={meta['trainViews']}  valViews={meta['valViews']}")

    if args.model == "cache":
        # Neural radiance cache: inputs are ONLY the ray + geometry + material.
        Xtr_raw = build_cache_features(X[tr])
        Xva_raw = build_cache_features(X[va])
        mean = Xtr_raw.mean(0); std = Xtr_raw.std(0) + 1e-6
        Xtr = ((Xtr_raw - mean) / std).astype(np.float32)
        Xva = ((Xva_raw - mean) / std).astype(np.float32)
        model = RadianceCache(num_bands=args.bands, hidden=args.hidden,
                              depth=args.depth, in_raw=Xtr.shape[1]).to(dev)
    else:
        # Standardization stats from training geometric columns only.
        mean = X[tr][:, GEOM_COLS].mean(0)
        std = X[tr][:, GEOM_COLS].std(0) + 1e-6
        Xtr = preprocess(X[tr], mean, std)
        Xva = preprocess(X[va], mean, std)
        model = Amplifier(num_bands=args.bands, hidden=args.hidden, depth=args.depth).to(dev)
    Ytr_log = np.log1p(np.maximum(Y[tr], 0.0))

    Xtr_t = torch.tensor(Xtr, device=dev)
    Ytr_t = torch.tensor(Ytr_log, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params={nparams}")

    n = Xtr_t.shape[0]
    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            xb, yb = Xtr_t[b], Ytr_t[b]
            pred_log = model(xb)
            # log-space MSE + relative linear term (HDR-aware, like NRC).
            loss = nn.functional.mse_loss(pred_log, yb)
            pred_lin = torch.expm1(pred_log).clamp(min=0)
            ref_lin = torch.expm1(yb)
            loss = loss + 0.1 * (((pred_lin - ref_lin) ** 2) / (ref_lin ** 2 + 1e-2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.shape[0]
        sched.step()
        history.append(tot / n)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {history[-1]:.5f}  lr {sched.get_last_lr()[0]:.2e}")
    train_secs = time.time() - t0

    # ---- Evaluation on held-out views -------------------------------------
    model.eval()
    with torch.no_grad():
        pred_log = model(Xva_t)
        pred = torch.expm1(pred_log).clamp(min=0).cpu().numpy()
    one = X[va][:, 18:21]                      # the single cast ray (1 spp), raw
    ref = Y[va]                                # high-spp reference
    hit = X[va][:, 0] > 0.5

    def report(mask, label):
        m_one = metrics(one[mask], ref[mask])
        m_net = metrics(pred[mask], ref[mask])
        amp = (m_one["normMSE"] / max(m_net["normMSE"], 1e-12))
        return {"label": label, "pixels": int(mask.sum()),
                "one_spp": m_one, "neural": m_net,
                "effective_spp_amplification": amp}

    results = {
        "device": str(dev), "model": args.model,
        "train_seconds": train_secs, "params": nparams,
        "epochs": args.epochs, "scene": meta.get("scene", "?"),
        "targetSpp": meta["targetSpp"], "valViews": meta["valViews"],
        "all_pixels": report(np.ones_like(hit), "all pixels"),
        "hit_pixels": report(hit, "shaded (hit) pixels"),
        "final_train_loss": history[-1],
    }
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    # Persist model + preprocessing so eval_amplify.py can score a held-out view.
    torch.save({"state_dict": model.state_dict(), "model": args.model,
                "bands": args.bands, "hidden": args.hidden, "depth": args.depth,
                "in_raw": Xtr.shape[1]},
               os.path.join(args.out, "model.pt"))
    np.savez(os.path.join(args.out, "standardize.npz"), mean=mean, std=std)

    # ---- Visual comparison for the first held-out view --------------------
    per = meta["perView"]; v0 = meta["valViews"][0]
    sl = slice(0, per)  # val arrays are concatenation of val views in order
    one_img = one[sl].reshape(H, W, 3)
    pred_img = pred[sl].reshape(H, W, 3)
    ref_img = ref[sl].reshape(H, W, 3)
    err_one = np.abs(tonemap_u8(one_img).astype(int) - tonemap_u8(ref_img).astype(int)).astype(np.uint8)
    err_net = np.abs(tonemap_u8(pred_img).astype(int) - tonemap_u8(ref_img).astype(int)).astype(np.uint8)
    try:
        from PIL import Image
        top = np.hstack([tonemap_u8(one_img), tonemap_u8(pred_img), tonemap_u8(ref_img)])
        bot = np.hstack([err_one, err_net, np.zeros_like(err_net)])
        grid = np.vstack([top, bot])
        Image.fromarray(grid).save(os.path.join(args.out, "comparison.png"))
        print("wrote comparison.png  [top: 1spp | neural | reference] [bottom: |err| maps]")
    except Exception as e:
        print("PIL unavailable, skipping image:", e)

    # ---- Console summary ---------------------------------------------------
    h = results["hit_pixels"]
    net_label = "cache " if args.model == "cache" else "neural"
    src = ("ray+geometry+material, NO traced sample" if args.model == "cache"
           else "one cast ray + inference")
    print("\n================ HYPOTHESIS RESULT (held-out views) ================")
    print(f"  model: {args.model}  ({src})")
    print(f"  shaded pixels: {h['pixels']}")
    print(f"  1-spp     relMSE={h['one_spp']['relMSE']:.4f}  PSNR={h['one_spp']['psnr_tonemapped']:.2f} dB")
    print(f"  {net_label}    relMSE={h['neural']['relMSE']:.4f}  PSNR={h['neural']['psnr_tonemapped']:.2f} dB")
    print(f"  >> effective sample amplification (normMSE ratio): "
          f"{h['effective_spp_amplification']:.1f}x  (one cast ray ~ "
          f"{h['effective_spp_amplification']:.0f} path-traced samples)")
    print("====================================================================")


if __name__ == "__main__":
    main()
