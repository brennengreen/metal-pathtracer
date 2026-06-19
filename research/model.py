"""Neural ray amplifier — model definition.

Hypothesis under test: cast ONE physical ray per pixel, then use a learned model
to predict what many samples would have produced (the converged radiance). The
network sees the single cast ray's 1-spp radiance plus first-hit G-buffer features
and regresses the multi-sample mean. This is a per-scene neural radiance cache /
amplifier in the spirit of Mueller et al., "Real-time Neural Radiance Caching for
Path Tracing" (2021), and radiance-regression caches.

Design choices:
  * Radiance is high-dynamic-range, so the single-sample input and the target are
    transformed with log1p and the network learns in log space (stable, HDR-aware).
  * Residual prediction: output is a correction added to the single sample's
    log-radiance, so the cheap-but-noisy signal is the starting point (denoising).
  * A small frequency (Fourier) encoding is applied to the geometric inputs
    (position, normal, view dir, bounce dir) to capture higher-frequency lighting.
"""
import torch
import torch.nn as nn


# Feature layout of the 21-dim input vector exported by the Metal tracer.
#  0      hit flag
#  1: 4   shading normal (xyz)
#  4: 7   view direction wo (xyz)
#  7:10   baseColor (xyz)
# 10      metallic
# 11      roughness
# 12:15   first-hit position, scene-normalized (xyz)
# 15:18   first sampled bounce direction (the one cast ray) (xyz)
# 18:21   single-sample (1 spp) radiance estimate (xyz)
GEOM_COLS = list(range(1, 18))      # standardized + partially Fourier-encoded
RAD_COLS = [18, 19, 20]             # single-sample radiance -> log space
FOURIER_COLS = list(range(12, 18))  # position + bounce direction


class FourierFeatures(nn.Module):
    """Concatenate sin/cos of geometrically-meaningful inputs at several scales."""

    def __init__(self, num_bands: int = 6):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(num_bands) * torch.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x[..., None] * self.freqs                      # (..., D, B)
        enc = torch.cat([proj.sin(), proj.cos()], dim=-1)     # (..., D, 2B)
        return enc.flatten(-2)                                # (..., D*2B)


class Amplifier(nn.Module):
    def __init__(self, num_bands: int = 6, hidden: int = 256, depth: int = 5):
        super().__init__()
        self.fourier = FourierFeatures(num_bands)
        # 21 raw dims + Fourier expansion of the 6 FOURIER_COLS.
        in_dim = 21 + len(FOURIER_COLS) * 2 * num_bands
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: standardized features with RAD_COLS already in log space.
        Returns predicted converged radiance in *log space* (residual over input)."""
        enc = self.fourier(x[..., FOURIER_COLS])
        feat = torch.cat([x, enc], dim=-1)
        delta = self.net(feat)
        one_sample_log = x[..., RAD_COLS]
        return one_sample_log + delta            # residual in log-radiance space


# ---------------------------------------------------------------------------
# Neural radiance cache (the "one inference, no recursion" variant).
#
# Hypothesis under test (a sharper form of the amplifier): regress the converged
# outgoing radiance L_o(x, wo) from ONLY the cast ray + hit geometry + material —
# no traced 1-spp sample, no sampled bounce direction, no recursion. A single MLP
# evaluation stands in for the whole rendering-equation integral over the BSDF
# lobe. Because L_o depends on the scene's incident radiance, position is part of
# the input (this is a per-scene cache, in the spirit of Mueller et al. 2021).
# ---------------------------------------------------------------------------

# Layout of the raw cache feature matrix built by `build_cache_features`:
#  0: 3  shading normal
#  3: 6  view direction wo (the cast ray, toward camera)
#  6: 9  ideal reflection R = reflect(-wo, n)  (centre of the specular lobe)
#  9:12  baseColor
# 12     metallic
# 13     roughness
# 14:17  first-hit position, scene-normalized
CACHE_IN_DIM = 17
CACHE_FOURIER_COLS = list(range(6, 9)) + list(range(14, 17))   # reflection dir + position


def build_cache_features(X):
    """Assemble the cache input from the exported 21-dim vectors, using ONLY the
    ray, geometry and material (cols 1:12 + 12:15) — never the sampled bounce
    (15:18) or the traced 1-spp radiance (18:21). Adds the deterministic mirror
    reflection of the view ray, a cheap closed-form cue for the specular lobe
    that needs no bounce sampling."""
    import numpy as np
    n = X[:, 1:4]
    wo = X[:, 4:7]
    base = X[:, 7:10]
    metal = X[:, 10:11]
    rough = X[:, 11:12]
    pos = X[:, 12:15]
    ndotwo = np.sum(n * wo, axis=1, keepdims=True)
    R = 2.0 * ndotwo * n - wo                       # reflect(-wo, n)
    return np.concatenate([n, wo, R, base, metal, rough, pos], axis=1).astype(np.float32)


class RadianceCache(nn.Module):
    """Predicts converged outgoing radiance directly (absolute log-radiance) from
    the ray + geometry + material — no traced sample, so no residual anchor."""

    def __init__(self, num_bands: int = 6, hidden: int = 256, depth: int = 5,
                 in_raw: int = CACHE_IN_DIM):
        super().__init__()
        self.fourier = FourierFeatures(num_bands)
        in_dim = in_raw + len(CACHE_FOURIER_COLS) * 2 * num_bands
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: standardized cache features. Returns converged radiance in log space
        (absolute, not a residual — there is no single-sample input to correct)."""
        enc = self.fourier(x[..., CACHE_FOURIER_COLS])
        return self.net(torch.cat([x, enc], dim=-1))
