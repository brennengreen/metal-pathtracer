#!/usr/bin/env bash
#
# download_moana.sh — fetch the real Disney *Moana Island Scene* data set and
# unpack it into a directory the Metal path tracer's `--scene moana` loader can read.
#
# The renderer's MoanaLoader consumes the **Base** package (json/ + obj/). The
# packages are large; sizes (from disneyanimation.com/resources/moana-island-scene):
#
#     base      45 GB download / 93 GB unpacked   <- needed by this renderer
#     animation 24 GB download / 131 GB unpacked
#     usd       15 GB download / 17 GB unpacked
#     pbrt       6 GB download / 41 GB unpacked
#     pbrtV4   5.6 GB download / 30 GB unpacked
#
# Usage:
#     research/download_moana.sh [base|animation|usd|pbrt|pbrtv4] [DEST_DIR]
#
# Default package is "base", default DEST_DIR is "./moana". The download is
# resumable (curl -C -); re-run the script to continue an interrupted transfer.
#
set -euo pipefail

PKG="${1:-base}"
DEST="${2:-./moana}"

case "$PKG" in
  base)      url="https://wdas-datasets-disneyanimation-com.s3-us-west-2.amazonaws.com/moanaislandscene/island-basepackage-v1.1.tgz"; need_gb=93 ;;
  animation) url="https://wdas-datasets-disneyanimation-com.s3-us-west-2.amazonaws.com/moanaislandscene/island-animation-v1.1.tgz"; need_gb=131 ;;
  usd)       url="https://datasets.disneyanimation.com/moanaislandscene/island-usd-v2.1.tgz"; need_gb=17 ;;
  pbrt)      url="https://wdas-datasets-disneyanimation-com.s3-us-west-2.amazonaws.com/moanaislandscene/island-pbrt-v1.1.tgz"; need_gb=41 ;;
  pbrtv4)    url="https://datasets.disneyanimation.com/moanaislandscene/island-pbrtV4-v2.0.tgz"; need_gb=30 ;;
  *) echo "unknown package '$PKG'. choose one of: base animation usd pbrt pbrtv4" >&2; exit 2 ;;
esac
file="$(basename "$url")"

echo "Moana Island Scene downloader"
echo "  package : $PKG"
echo "  url     : $url"
echo "  dest    : $DEST"
echo "  license : research/educational use; see"
echo "            https://media.disneyanimation.com/uploads/production/data_set_asset/4/asset/License.txt"
echo

# --- disk-space guard ------------------------------------------------------
mkdir -p "$DEST"
avail_kb="$(df -Pk "$DEST" | awk 'NR==2{print $4}')"
avail_gb=$(( avail_kb / 1024 / 1024 ))
echo "  unpacked needs ~${need_gb} GB; ${avail_gb} GB free at $DEST"
if (( avail_gb < need_gb + 10 )); then
  echo
  echo "!! Not enough free space: need ~$((need_gb + 10)) GB (download + unpacked), have ${avail_gb} GB." >&2
  echo "   The Moana data set will not fit here. Run this on a machine/volume with room," >&2
  echo "   then copy the unpacked 'island/' directory over, or point --moana at it via a" >&2
  echo "   network/external mount." >&2
  exit 1
fi

# --- download (resumable) --------------------------------------------------
echo
echo ">> downloading $file (resumable; re-run to continue)…"
curl -fL --retry 5 --retry-delay 5 -C - -o "$DEST/$file" "$url"

# --- extract ---------------------------------------------------------------
echo
echo ">> extracting $file …"
tar -xzf "$DEST/$file" -C "$DEST"

# The base package unpacks to $DEST/island/{json,obj,...}
island="$DEST/island"
echo
if [[ -d "$island/json" && -d "$island/obj" ]]; then
  echo "Done. Render the real scene with:"
  echo
  echo "    .build/release/pathtracer --scene moana --moana \"$island\" \\"
  echo "        --width 1280 --height 720 --spp 64 --bounces 6 --clamp 8 \\"
  echo "        --maxInst 4000000 --out moana.png"
  echo
  echo "  (--maxInst caps instances to fit RAM; raise it as memory allows."
  echo "   add --moanaTranspose only if geometry looks sheared.)"
else
  echo "Extracted to $DEST. Point --moana at the directory containing json/ and obj/."
fi
