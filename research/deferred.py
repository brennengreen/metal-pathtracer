"""Neural deferred shading — Stage 1: minimal G-buffer image -> beauty image.

The hypothesis line: cast only primary rays, keep a *minimal* deferred G-buffer
(hit position, shading normal, albedo) — the cheap data a rasteriser already has
— and let a small screen-space U-Net synthesise the fully shaded, globally-
illuminated image. This is the convolutional, spatial-context counterpart to the
per-pixel radiance cache (research/model.py `RadianceCache`): the U-Net can use
neighbours, which is also the prerequisite for the temporal (motion-vector) loop
in Stage 2.

The dataset is the SAME one the exporter already writes (DataExport.swift): each
camera view's per-pixel feature rows are reshaped back into an [H, W, C] image,
and we slice out the minimal G-buffer channels as input and the converged
radiance as the target. Held-out *views* (never seen during training) measure
generalisation, exactly like the cache experiment.

Run:
    python3 research/deferred.py --data research/data_showcase --out research/results_deferred
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

# Channel layout of the 21-dim exported feature vector (see research/model.py).
HIT, NORMAL, ALBEDO, POS = [0], list(range(1, 4)), list(range(7, 10)), list(range(12, 15))
GBUF_COLS = HIT + NORMAL + ALBEDO + POS          # minimal deferred input (10 ch)


# --- tonemapping + metrics (match the Metal resolve kernel / train.py) ------
def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def tonemap_u8(img):
    return (srgb(aces(img)) * 255 + 0.5).astype(np.uint8)

def metrics(pred, ref, eps=1e-2):
    rel = float(np.mean((pred - ref) ** 2 / (ref ** 2 + eps)))
    mse_tm = np.mean((tonemap_u8(pred) / 255.0 - tonemap_u8(ref) / 255.0) ** 2)
    psnr = float(10 * np.log10(1.0 / max(mse_tm, 1e-12)))
    return {"relMSE": rel, "psnr_tonemapped": psnr}


def load_images(path):
    """Load a per-view export into images + the minimal G-buffer channels, handling
    both formats: the jittered-views dataset (X.bin, 21-dim → slice GBUF_COLS) and
    the orbit-sequence dataset (Xseq.bin, already the 10-ch textured G-buffer)."""
    meta = json.load(open(os.path.join(path, "meta.json")))
    W, H = meta["width"], meta["height"]
    if os.path.exists(os.path.join(path, "Xseq.bin")):          # instanced/textured sequence
        F = meta["frames"]; GB = meta["gbuf"]
        Xg = np.fromfile(os.path.join(path, "Xseq.bin"), np.float32).reshape(F, H, W, GB)
        Y = np.fromfile(os.path.join(path, "Yseq.bin"), np.float32).reshape(F, H, W, 3)
        nval = max(4, F // 5)
        return meta, Xg, Y, list(range(F - nval)), list(range(F - nval, F))
    V, d = meta["views"], meta["inDim"]                          # jittered-views, 21-dim
    X = np.fromfile(os.path.join(path, "X.bin"), np.float32).reshape(V, H, W, d)
    Y = np.fromfile(os.path.join(path, "Y.bin"), np.float32).reshape(V, H, W, 3)
    return meta, X[..., GBUF_COLS], Y, meta["trainViews"], meta["valViews"]


# --- compact U-Net ----------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Screen-space G-buffer -> log-radiance, 3 down/up levels with skips."""

    def __init__(self, in_ch, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.bott = ConvBlock(base * 4, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = ConvBlock(base * 4 + base * 4, base * 2)
        self.d2 = ConvBlock(base * 2 + base * 2, base)
        self.d1 = ConvBlock(base + base, base)
        self.head = nn.Conv2d(base, 3, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.bott(self.pool(s3))
        d3 = self.d3(torch.cat([self.up(b), s3], 1))
        d2 = self.d2(torch.cat([self.up(d3), s2], 1))
        d1 = self.d1(torch.cat([self.up(d2), s1], 1))
        return self.head(d1)                         # log-radiance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data")
    ap.add_argument("--out", default="research/results_deferred")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

    meta, Xg, Y, train_v, val_v = load_images(args.data)
    H, W = meta["height"], meta["width"]
    GB = Xg.shape[-1]
    print(f"device={dev}  scene={meta.get('scene','?')}  frames/views={Xg.shape[0]}  "
          f"{W}x{H}  train={len(train_v)}  val={len(val_v)}  in_ch={GB}")

    Ylog = np.log1p(np.maximum(Y, 0.0))              # HDR-aware target

    # Standardise input channels using TRAIN views only.
    flat = Xg[train_v].reshape(-1, GB)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    alb_fig = Xg[val_v[0]][..., 4:7].copy()          # raw albedo of 1st held-out frame (for figure)
    Xg = (Xg - mean) / std

    def to_chw(a):                                    # [V,H,W,C] -> tensor [V,C,H,W]
        return torch.tensor(a.transpose(0, 3, 1, 2), device=dev)
    Xtr, Ytr = to_chw(Xg[train_v]), to_chw(Ylog[train_v])
    Xva, Yva = to_chw(Xg[val_v]), to_chw(Ylog[val_v])

    model = UNet(GB, base=args.base).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"U-Net params={nparams}")

    nt = Xtr.shape[0]; cz = min(args.crop, H, W)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        # Random crops + flips from the train views = data augmentation.
        idx = torch.randint(0, nt, (args.batch,))
        ys = torch.randint(0, H - cz + 1, (1,)).item()
        xs = torch.randint(0, W - cz + 1, (1,)).item()
        xb = Xtr[idx][:, :, ys:ys + cz, xs:xs + cz]
        yb = Ytr[idx][:, :, ys:ys + cz, xs:xs + cz]
        if torch.rand(1).item() < 0.5:
            xb, yb = torch.flip(xb, [3]), torch.flip(yb, [3])
        pred = model(xb)
        loss = nn.functional.mse_loss(pred, yb)
        pl, rl = torch.expm1(pred).clamp(min=0), torch.expm1(yb)
        loss = loss + 0.1 * (((pl - rl) ** 2) / (rl ** 2 + 1e-2)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}")
    train_secs = time.time() - t0

    # --- evaluate on held-out views (full frames) --------------------------
    model.eval()
    with torch.no_grad():
        pred = torch.expm1(model(Xva)).clamp(min=0).cpu().numpy().transpose(0, 2, 3, 1)
    ref = Y[val_v]
    m = metrics(pred, ref)
    results = {"model": "deferred_unet", "device": str(dev), "params": nparams,
               "train_seconds": train_secs, "epochs": args.epochs,
               "scene": meta.get("scene", "?"), "valViews": val_v,
               "in_channels": GB, "held_out": m}
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    torch.save({"state_dict": model.state_dict(), "base": args.base,
                "in_ch": GB}, os.path.join(args.out, "model.pt"))
    np.savez(os.path.join(args.out, "standardize.npz"), mean=mean, std=std)

    # --- comparison figure for the first held-out view ---------------------
    try:
        from PIL import Image
        top = np.hstack([tonemap_u8(np.clip(alb_fig, 0, 1)),  # albedo (a G-buffer input)
                         tonemap_u8(pred[0]),               # U-Net prediction
                         tonemap_u8(ref[0])])               # path-traced reference
        err = np.abs(tonemap_u8(pred[0]).astype(int) - tonemap_u8(ref[0]).astype(int)).astype(np.uint8)
        bot = np.hstack([np.zeros_like(err), err, np.zeros_like(err)])
        Image.fromarray(np.vstack([top, bot])).save(os.path.join(args.out, "comparison.png"))
        print("wrote comparison.png  [albedo-in | U-Net | reference] / [_ | |err| | _]")
    except Exception as e:
        print("PIL unavailable, skipping image:", e)

    print("\n============== DEFERRED SHADING (held-out views) ==============")
    print(f"  scene={meta.get('scene','?')}  in-channels={GB} (pos+normal+albedo+hit)")
    print(f"  U-Net  relMSE={m['relMSE']:.4f}  PSNR={m['psnr_tonemapped']:.2f} dB")
    print("===============================================================")


if __name__ == "__main__":
    main()
