"""Interactive neural deferred-shading viewer.

Orbit the camera and watch the trained U-Net synthesise the fully shaded image,
live, from only a minimal deferred G-buffer (position, normal, albedo) — no path
tracing, no recursion. The Metal renderer runs as a resident G-buffer *server*
(`pathtracer --gbufserver`): each camera produces a single primary-ray G-buffer
in ~milliseconds, which this script feeds to the network (PyTorch / MPS).

    python3 research/play.py                      # interactive window
    python3 research/play.py --selftest out.png   # one frame -> PNG (no GUI)

Controls (interactive):  arrow keys orbit · w/s zoom · a/d/q/e pan · r reset · esc quit
"""
import argparse, os, subprocess, sys, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deferred import UNet, GBUF_COLS


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)

def tonemap_u8(img):
    return (srgb(aces(img)) * 255 + 0.5).astype(np.uint8)


class GBufferServer:
    """Drive the resident Metal `--gbufserver` process: send a camera, get a G-buffer."""

    def __init__(self, binary, scene, width, height, bounces):
        self.W, self.H = width, height
        self.path = f"/tmp/pt_gbuf_{os.getpid()}.bin"
        self.proc = subprocess.Popen(
            [binary, "--gbufserver", "--scene", scene, "--width", str(width),
             "--height", str(height), "--bounces", str(bounces), "--out", self.path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.proc.stdout.readline().split()
        assert ready and ready[0] == "READY", f"server handshake failed: {ready}"
        _, w, h, cx, cy, cz, ex, ey, ez, fov = ready[:10]
        self.center = np.array([float(cx), float(cy), float(cz)])
        self.eye0 = np.array([float(ex), float(ey), float(ez)])
        self.fov = float(fov)

    def gbuffer(self, eye, target, fov):
        msg = ",".join(f"{v:.5f}" for v in [*eye, *target, fov]) + "\n"
        self.proc.stdin.write(msg); self.proc.stdin.flush()
        ack = self.proc.stdout.readline()                 # wait for "OK"
        assert ack.strip() == "OK", f"server error: {ack!r}"
        g = np.fromfile(self.path, dtype=np.float32).reshape(self.H, self.W, len(GBUF_COLS))
        return g

    def close(self):
        try:
            self.proc.stdin.write("quit\n"); self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.terminate()


def load_model(results_dir, dev):
    ck = torch.load(os.path.join(results_dir, "model.pt"), map_location=dev)
    model = UNet(ck["in_ch"], base=ck["base"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    st = np.load(os.path.join(results_dir, "standardize.npz"))
    return model, st["mean"].astype(np.float32), st["std"].astype(np.float32)


def shade(model, mean, std, gbuf, dev):
    x = (gbuf - mean) / std
    t = torch.tensor(x.transpose(2, 0, 1)[None], device=dev)
    with torch.no_grad():
        pred = torch.expm1(model(t)).clamp(min=0)[0].cpu().numpy().transpose(1, 2, 0)
    return tonemap_u8(pred)


def orbit_eye(center, radius, yaw, pitch):
    return center + radius * np.array([math.cos(pitch) * math.sin(yaw),
                                       math.sin(pitch),
                                       math.cos(pitch) * math.cos(yaw)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="showcase")
    ap.add_argument("--results", default="research/results_deferred")
    ap.add_argument("--binary", default=".build/release/pathtracer")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bounces", type=int, default=6)
    ap.add_argument("--selftest", default=None, help="render one frame to this PNG and exit")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model(args.results, dev)
    srv = GBufferServer(args.binary, args.scene, args.size, args.size, args.bounces)

    center = srv.center
    off = srv.eye0 - center
    radius = float(np.linalg.norm(off)) or 1.0
    state = {"yaw": math.atan2(off[0], off[2]),
             "pitch": math.asin(np.clip(off[1] / radius, -0.999, 0.999)),
             "radius": radius, "target": center.copy(), "fov": srv.fov}

    def render():
        eye = orbit_eye(state["target"], state["radius"], state["yaw"], state["pitch"])
        g = srv.gbuffer(eye, state["target"], state["fov"])
        return shade(model, mean, std, g, dev)

    if args.selftest:
        from PIL import Image
        Image.fromarray(render()).save(args.selftest)
        print(f"wrote {args.selftest}  (neural deferred shading, {args.size}x{args.size})")
        srv.close(); return

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Neural deferred shading — minimal G-buffer → photoreal")
    im = ax.imshow(render()); ax.axis("off")
    ax.set_title("orbit: arrows · zoom: w/s · pan: a/d/q/e · reset: r · quit: esc", fontsize=9)

    def on_key(ev):
        s, rot, pan = state, 0.08, state["radius"] * 0.06
        right = np.array([math.cos(s["yaw"]), 0, -math.sin(s["yaw"])])
        if ev.key in ("left",): s["yaw"] -= rot
        elif ev.key == "right": s["yaw"] += rot
        elif ev.key == "up": s["pitch"] = min(1.5, s["pitch"] + rot)
        elif ev.key == "down": s["pitch"] = max(-1.5, s["pitch"] - rot)
        elif ev.key == "w": s["radius"] *= 0.92
        elif ev.key == "s": s["radius"] *= 1.08
        elif ev.key == "a": s["target"] = s["target"] - right * pan
        elif ev.key == "d": s["target"] = s["target"] + right * pan
        elif ev.key == "q": s["target"][1] += pan
        elif ev.key == "e": s["target"][1] -= pan
        elif ev.key == "r":
            o = srv.eye0 - center
            s.update(yaw=math.atan2(o[0], o[2]),
                     pitch=math.asin(np.clip(o[1] / radius, -0.999, 0.999)),
                     radius=radius, target=center.copy())
        elif ev.key == "escape":
            plt.close(fig); return
        else:
            return
        im.set_data(render()); fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Interactive viewer ready — focus the window and use the keys.")
    plt.show()
    srv.close()


if __name__ == "__main__":
    main()
