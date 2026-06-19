#!/usr/bin/env bash
#
# download_sponza.sh — fetch the Crytek *Sponza* atrium (the canonical global-
# illumination test scene) and lay it out so the Metal path tracer's
# `--scene sponza` loader can read it.
#
# Source mirror: github.com/jimmiebergmann/Sponza (the widely-used OBJ/MTL
# conversion of Frank Meinl's original Crytek model). It ships:
#
#     sponza.obj    ~18 MB   145k verts / 262k tris / 393 usemtl groups
#     sponza.mtl    ~6 KB    25 materials (all share a gray Kd — see note)
#     textures/     ~160 MB  43 .tga maps (diffuse + normal/displacement)
#
# NOTE on materials: every Sponza material has the *same* gray diffuse
# (Kd ≈ 0.47). ALL of the colour/detail lives in the `map_Kd` .tga textures,
# so texture mapping is mandatory to see "materials working" — the renderer
# does exactly this. A few alpha-mask maps (`*_mask.tga`, used by leaves/
# chains/thorns) are absent from this mirror and 404; that is expected and
# harmless (those parts render opaque).
#
# Usage:
#     research/download_sponza.sh [DEST_DIR]
#
# Default DEST_DIR is "assets/sponza" (where `--scene sponza` looks by default).
# Downloads are resumable (curl -C -); re-run to continue an interrupted fetch.
#
set -euo pipefail

DEST="${1:-assets/sponza}"
BASE="https://raw.githubusercontent.com/jimmiebergmann/Sponza/master"

echo "Crytek Sponza downloader"
echo "  source : $BASE"
echo "  dest   : $DEST"
echo "  size   : ~180 MB unpacked (obj + mtl + 43 textures)"
echo

mkdir -p "$DEST/textures"

fetch() { # url  outfile
  echo ">> $(basename "$2")"
  curl -fL --retry 5 --retry-delay 3 -C - -o "$2" "$1"
}

# --- geometry + material library ------------------------------------------
fetch "$BASE/sponza.obj" "$DEST/sponza.obj"
fetch "$BASE/sponza.mtl" "$DEST/sponza.mtl"

# --- textures: parse every `textures/*.tga` referenced by the .mtl ---------
# (covers map_Kd / map_Disp / map_d). Missing maps 404 — tolerate and continue.
echo
echo ">> textures referenced by sponza.mtl …"
missing=0
grep -oE 'textures/[^[:space:]]+\.tga' "$DEST/sponza.mtl" | sort -u | while read -r rel; do
  if curl -fL --retry 3 --retry-delay 2 -C - -o "$DEST/$rel" "$BASE/$rel" 2>/dev/null; then
    echo "   ok   $rel"
  else
    echo "   miss $rel (absent from mirror — ok, renders opaque)"
  fi
done

echo
echo "Done. Render the iconic colonnade (auto-framed, materials + textures + GI):"
echo
echo "    swift build -c release"
echo "    .build/release/pathtracer --scene sponza \\"
echo "        --width 1600 --height 900 --spp 384 --bounces 6 --clamp 8 \\"
echo "        --out docs/sponza.png"
echo
echo "  Optional env toggles (all read by the sponza loader):"
echo "    SPONZA_CAM=\"ex,ey,ez,tx,ty,tz\"  override camera (native coords)"
echo "    SPONZA_NOTEX=1                    force flat gray Kd (no textures)"
echo "    SPONZA_KEEPROOF=1                 keep the roof cap (darkens interior)"
echo "    SPONZA_DBG=1                      print measured atrium-core bounds"
