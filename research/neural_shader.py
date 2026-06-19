"""Train a compact PER-PIXEL neural shader for realtime in-Metal inference.

This is the realtime path: a small MLP that maps a single hit's G-buffer
(shading normal, textured albedo, scene-normalized position) to the converged
sun+sky radiance — no convolution, no recursion, no history. Because it is
per-pixel it ports directly to a Metal compute shader (one thread per pixel),
so the trained weights run inside the render loop at frame rate.

The architecture here is FIXED to match the Metal kernel `neuralShadeInst`:
    raw 9  = normal.xyz, albedo.xyz, posScaled.xyz   (standardized)
    input  = raw9 ++ Fourier(posScaled, BANDS)        -> IN dims
    MLP    = IN -> H -> H -> H -> 3   (GELU/erf), output = log radiance

Exports research/results_neural/shader.bin (+ meta.json) that the Swift loader
reads into MTLBuffers.

Run:  python3 research/neural_shader.py --data research/data_sponza --out research/results_neural
"""
import argparse, json, os, time, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deferred import load_images, aces, srgb, tonemap_u8, metrics

# Architecture constants — MUST match the Metal kernel.
BANDS = 4
H = 64
RAW = 9                       # normal(3) + albedo(3) + position(3)
IN = RAW + 3 * 2 * BANDS      # + Fourier of the 3 position components


class PerPixelShader(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(IN, H)
        self.l2 = nn.Linear(H, H)
        self.l3 = nn.Linear(H, H)
        self.l4 = nn.Linear(H, 3)
        # frequencies for the positional Fourier features (match the kernel)
        self.register_buffer("freqs", 2.0 ** torch.arange(BANDS) * torch.pi)

    def fourier(self, pos):                       # pos: [...,3] standardized
        proj = pos[..., None] * self.freqs        # [...,3,BANDS]
        enc = torch.cat([proj.sin(), proj.cos()], -1)  # [...,3,2B]
        return enc.flatten(-2)                    # [...,3*2B]

    def forward(self, raw9):                      # raw9 already standardized
        pos = raw9[..., 6:9]
        x = torch.cat([raw9, self.fourier(pos)], -1)
        g = nn.functional.gelu
        x = g(self.l1(x)); x = g(self.l2(x)); x = g(self.l3(x))
        return self.l4(x)                         # log radiance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data_sponza")
    ap.add_argument("--out", default="research/results_neural")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

    meta, Xg, Y, train_v, val_v = load_images(args.data)   # Xg [F,H,W,10]
    Hh, Ww = meta["height"], meta["width"]
    print(f"device={dev}  scene={meta.get('scene','?')}  frames={Xg.shape[0]}  "
          f"{Ww}x{Hh}  train={len(train_v)}  val={len(val_v)}")

    # raw9 = normal(1:4) + albedo(4:7) + position(7:10) of the 10-ch G-buffer.
    def raw9_of(a):
        return np.concatenate([a[..., 1:4], a[..., 4:7], a[..., 7:10]], -1).astype(np.float32)
    hit = Xg[..., 0] > 0.5

    Xtr_img, Ytr_img = raw9_of(Xg[train_v]), Y[train_v]
    htr = hit[train_v]
    Xtr = Xtr_img[htr]                                   # [N,9] hit pixels only
    Ytr = np.log1p(np.maximum(Ytr_img[htr], 0.0))

    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xt = torch.tensor(Xtr, device=dev); Yt = torch.tensor(Ytr, device=dev)

    model = PerPixelShader().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"per-pixel MLP  IN={IN}  H={H}  params={sum(p.numel() for p in model.parameters())}")

    n = Xt.shape[0]; t0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            pred = model(Xt[b])
            loss = nn.functional.mse_loss(pred, Yt[b])
            pl, rl = torch.expm1(pred).clamp(min=0), torch.expm1(Yt[b])
            loss = loss + 0.1 * (((pl - rl) ** 2) / (rl ** 2 + 1e-2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * b.shape[0]
        sched.step()
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot / n:.5f}  lr {sched.get_last_lr()[0]:.2e}")
    train_secs = time.time() - t0

    # --- held-out evaluation (full frames; misses left black) --------------
    model.eval()
    def shade_frame(g):
        r9 = (raw9_of(g) - mean) / std
        with torch.no_grad():
            out = torch.expm1(model(torch.tensor(r9, device=dev))).clamp(min=0).cpu().numpy()
        out[g[..., 0] <= 0.5] = 0.0
        return out
    preds = np.stack([shade_frame(Xg[v]) for v in val_v])
    ref = Y[val_v].copy(); ref[Xg[val_v][..., 0] <= 0.5] = 0.0
    m = metrics(preds, ref)

    # --- export weights for the Metal kernel -------------------------------
    def wb(lin):   # PyTorch Linear weight [out,in] row-major + bias
        return lin.weight.detach().cpu().numpy().astype(np.float32).ravel(), \
               lin.bias.detach().cpu().numpy().astype(np.float32).ravel()
    blobs = [mean.astype(np.float32), std.astype(np.float32)]
    layers = []
    for lin in (model.l1, model.l2, model.l3, model.l4):
        w, b = wb(lin); blobs += [w, b]
        layers.append([lin.out_features, lin.in_features])
    np.concatenate(blobs).tofile(os.path.join(args.out, "shader.bin"))
    json.dump({"in_dim": IN, "hidden": H, "bands": BANDS, "raw_dim": RAW,
               "layers": layers, "scene": meta.get("scene", "?"),
               "held_out_psnr": m["psnr_tonemapped"]},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)

    try:
        from PIL import Image
        Image.fromarray(np.hstack([tonemap_u8(preds[0]), tonemap_u8(ref[0])])).save(
            os.path.join(args.out, "comparison.png"))
        print("wrote comparison.png  [neural shade | reference]")
    except Exception as e:
        print("PIL unavailable:", e)

    print("\n========= PER-PIXEL NEURAL SHADER (held-out frames) =========")
    print(f"  scene={meta.get('scene','?')}  params={sum(p.numel() for p in model.parameters())}")
    print(f"  PSNR={m['psnr_tonemapped']:.2f} dB   relMSE={m['relMSE']:.4f}   train {train_secs:.0f}s")
    print(f"  exported {args.out}/shader.bin  (realtime Metal weights)")
    print("=============================================================")


if __name__ == "__main__":
    main()
