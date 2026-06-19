"""Neural deferred shading — Stage 2: temporal stability via motion vectors.

Stage 1 shades each frame independently, so it shimmers under camera motion. Here
the network additionally receives the *previous output reprojected by the motion
vectors* (backward warp) and learns to blend it with the current G-buffer — the
same reproject-and-accumulate trick temporal denoisers / TAA use. We measure both
quality (PSNR vs reference) and **temporal stability** (motion-compensated flicker)
on a held-out arc of the camera orbit, against a no-history ablation.

Dataset: research/data_seq (exported with `pathtracer --exportseq`):
    Xseq.bin [F,H,W,10]  G-buffer    MVseq.bin [F,H,W,2]  motion (px, current-prev)
    Yseq.bin [F,H,W,3]   converged target

Run:  python3 research/temporal.py --data research/data_seq --out research/results_temporal
"""
import argparse, json, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deferred import UNet, aces, srgb, tonemap_u8, metrics


def load_seq(path):
    meta = json.load(open(os.path.join(path, "meta.json")))
    F_, H, W, GB = meta["frames"], meta["height"], meta["width"], meta["gbuf"]
    X = np.fromfile(os.path.join(path, "Xseq.bin"), np.float32).reshape(F_, H, W, GB)
    MV = np.fromfile(os.path.join(path, "MVseq.bin"), np.float32).reshape(F_, H, W, 2)
    Y = np.fromfile(os.path.join(path, "Yseq.bin"), np.float32).reshape(F_, H, W, 3)
    return meta, X, MV, Y


def warp(img, mv):
    """Backward-warp [N,C,H,W] by motion [N,2,H,W] (px, current-prev): the surface
    at current pixel p was at p-mv last frame, so sample the previous frame there."""
    N, C, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device),
                            torch.arange(W, device=img.device), indexing="ij")
    base = torch.stack([xs, ys], 0).float()[None]              # [1,2,H,W] = (x,y)
    src = base - mv
    gx = 2 * src[:, 0] / max(W - 1, 1) - 1
    gy = 2 * src[:, 1] / max(H - 1, 1) - 1
    grid = torch.stack([gx, gy], -1)                           # [N,H,W,2]
    return F.grid_sample(img, grid, align_corners=True, padding_mode="zeros")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data_seq")
    ap.add_argument("--out", default="research/results_temporal")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")

    meta, X, MV, Y = load_seq(args.data)
    Fn, H, W, GB = X.shape
    nval = max(8, Fn // 4)
    train_f, val_f = list(range(Fn - nval)), list(range(Fn - nval, Fn))
    print(f"device={dev}  scene={meta.get('scene','?')}  frames={Fn}  {W}x{H}  "
          f"train={len(train_f)}  heldout-arc={len(val_f)}")

    # Standardise G-buffer channels with train frames; target -> log space.
    flat = X[train_f].reshape(-1, GB)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    Xs = ((X - mean) / std).astype(np.float32)
    Ylog = np.log1p(np.maximum(Y, 0.0)).astype(np.float32)

    def chw(a, i):                                             # frame i -> [1,C,H,W]
        return torch.tensor(a[i].transpose(2, 0, 1)[None], device=dev)
    Xt = [chw(Xs, i) for i in range(Fn)]
    Mt = [chw(MV, i) for i in range(Fn)]
    Yt = [chw(Ylog, i) for i in range(Fn)]

    model = UNet(GB + 3, base=args.base).to(dev)               # G-buffer + warped history
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    print(f"temporal U-Net params={sum(p.numel() for p in model.parameters())}")

    def loss_fn(pred, tgt):
        l = F.mse_loss(pred, tgt)
        pl, rl = torch.expm1(pred).clamp(min=0), torch.expm1(tgt)
        return l + 0.1 * (((pl - rl) ** 2) / (rl ** 2 + 1e-2)).mean()

    zeros = torch.zeros(1, 3, H, W, device=dev)
    t0 = time.time()
    for step in range(args.steps):
        model.train()
        L = args.window
        start = np.random.randint(0, len(train_f) - L + 1)
        prev = zeros
        loss = 0.0
        for k in range(L):
            t = train_f[start + k]
            warped = warp(prev.detach(), Mt[t]) if k > 0 else zeros   # history is a fixed input
            pred = model(torch.cat([Xt[t], warped], 1))        # log radiance
            loss = loss + loss_fn(pred, Yt[t])
            prev = pred
        loss = loss / L
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.2e}")
    train_secs = time.time() - t0

    # --- evaluate on the held-out arc -------------------------------------
    def rollout(use_history):
        model.eval(); preds = []; prev = zeros
        with torch.no_grad():
            for j, t in enumerate(val_f):
                warped = warp(prev, Mt[t]) if (use_history and j > 0) else zeros
                pred = model(torch.cat([Xt[t], warped], 1))
                preds.append(pred); prev = pred
        return [torch.expm1(p).clamp(min=0)[0].cpu().numpy().transpose(1, 2, 0) for p in preds]

    def flicker(frames):
        """Motion-compensated temporal gradient over co-visible, hit pixels (tonemapped)."""
        tot, n = 0.0, 0
        for j in range(1, len(val_f)):
            t = val_f[j]
            cur = torch.tensor(tonemap_u8(frames[j]) / 255.0, device=dev).permute(2, 0, 1)[None].float()
            prv = torch.tensor(tonemap_u8(frames[j - 1]) / 255.0, device=dev).permute(2, 0, 1)[None].float()
            wprev = warp(prv, Mt[t])[0].permute(1, 2, 0).cpu().numpy()
            mask = (X[t][..., 0] > 0.5) & (np.abs(MV[t]).sum(-1) > 0)   # hit + has motion
            d = (tonemap_u8(frames[j]) / 255.0 - wprev)[mask]
            tot += float(np.mean(d ** 2)) if mask.sum() else 0.0; n += 1
        return tot / max(n, 1)

    ref = [Y[t] for t in val_f]
    temporal = rollout(use_history=True)
    ablation = rollout(use_history=False)                      # same net, history disabled
    q_t = metrics(np.stack(temporal), np.stack(ref))
    q_a = metrics(np.stack(ablation), np.stack(ref))
    fl_t, fl_a, fl_ref = flicker(temporal), flicker(ablation), flicker(ref)

    results = {"model": "temporal_unet", "device": str(dev),
               "params": sum(p.numel() for p in model.parameters()),
               "train_seconds": train_secs, "steps": args.steps, "window": args.window,
               "scene": meta.get("scene", "?"), "heldout_frames": val_f,
               "temporal": {"quality": q_t, "flicker": fl_t},
               "no_history_ablation": {"quality": q_a, "flicker": fl_a},
               "reference_flicker": fl_ref}
    json.dump(results, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    torch.save({"state_dict": model.state_dict(), "base": args.base, "in_ch": GB + 3},
               os.path.join(args.out, "model.pt"))

    # --- figure: 4 consecutive held-out frames, temporal vs reference ------
    try:
        from PIL import Image
        idx = np.linspace(1, len(val_f) - 1, 4).astype(int)
        top = np.hstack([tonemap_u8(temporal[i]) for i in idx])
        bot = np.hstack([tonemap_u8(ref[i]) for i in idx])
        Image.fromarray(np.vstack([top, bot])).save(os.path.join(args.out, "comparison.png"))
        print("wrote comparison.png  [top: temporal model] [bottom: path-traced reference]")
    except Exception as e:
        print("PIL unavailable, skipping image:", e)

    print("\n============ TEMPORAL DEFERRED SHADING (held-out arc) ============")
    print(f"  temporal     PSNR={q_t['psnr_tonemapped']:.2f} dB   flicker={fl_t:.5f}")
    print(f"  no-history   PSNR={q_a['psnr_tonemapped']:.2f} dB   flicker={fl_a:.5f}")
    print(f"  reference                          flicker={fl_ref:.5f}  (MC noise)")
    gain = (fl_a / fl_t) if fl_t > 0 else float('inf')
    print(f"  >> motion-vector history cuts flicker {gain:.2f}x vs no-history")
    print("=================================================================")


if __name__ == "__main__":
    main()
