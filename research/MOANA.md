# Rendering the Disney *Moana Island Scene* with this path tracer

The [Moana Island Scene](https://www.disneyanimation.com/resources/moana-island-scene/)
is Walt Disney Animation Studios' production data set for the island of Motunui.
It is the canonical stress test for **massive instancing**: a handful of unique
base meshes (~28 million unique quads/triangles) are replicated through nested
instancing into roughly **15 billion** renderable primitives — palms, ferns,
debris, coral, rock — far more than fits in memory as unique geometry.

This renderer supports the real data through the same two-level acceleration
structure (BLAS-per-mesh + a top-level instance acceleration structure) that the
scene was designed to exercise.

---

## TL;DR

```bash
# 1. get the data (needs a machine with ~100 GB free — see below)
research/download_moana.sh base ./moana          # -> ./moana/island/{json,obj,...}

# 2. render it
.build/release/pathtracer --scene moana --moana ./moana/island \
    --width 1280 --height 720 --spp 64 --bounces 6 --clamp 8 \
    --maxInst 4000000 --out moana.png
```

If you only want to *see* the technique without the 45 GB download, render the
bundled procedural Moana-scale island instead:

```bash
.build/release/pathtracer --scene island --width 1280 --height 720 --spp 192 \
    --bounces 5 --clamp 6 --out island.png         # ~18k instances, presentable
.build/release/pathtracer --scene island --density 700 --kernel normals \
    --width 640 --height 360 --out scale.png        # ~13M instances, 1.0B+ tris
```

---

## The data packages

| Package    | Download | Unpacked | Contains                              | Used here |
|------------|---------:|---------:|---------------------------------------|:---------:|
| **base**   |   45 GB  |   93 GB  | `json/` element + archive descriptions, `obj/` geometry, materials | **yes** |
| animation  |   24 GB  |  131 GB  | per-frame animated transforms          | no |
| usd        |   15 GB  |   17 GB  | USD + RenderMan bindings               | no |
| pbrt       |    6 GB  |   41 GB  | pbrt-v3 translation                    | no |
| pbrtV4     |  5.6 GB  |   30 GB  | pbrt-v4 translation (Matt Pharr)       | no |

The renderer reads the **base** package: it parses Disney's element JSON directly,
so no PBRT/USD conversion is required.

> **License.** The data is provided by Disney for research/educational use under
> their own license — read it before downloading:
> <https://media.disneyanimation.com/uploads/production/data_set_asset/4/asset/License.txt>

### Why the data isn't bundled

The base package is **45 GB compressed / 93 GB unpacked**. It cannot live in this
repository and will not fit on a typical laptop volume that is already near full.
`download_moana.sh` checks free space and refuses to start if there isn't room,
telling you to fetch it on a roomier volume and point `--moana` at the unpacked
`island/` directory (a local copy, external SSD, or network mount all work).

---

## How the loader maps Moana's format onto the engine

`Sources/PathTracer/MoanaLoader.swift` walks the data exactly as Disney lays it
out (see `island-README-v1.1.pdf`):

```
island/
  json/<element>/<element>.json          element: transformMatrix, geomObjFile,
                                          instancedCopies, instancedPrimitiveJsonFiles
  json/<element>/<element>_<archive>.json archive: objPath -> { instanceName: [16 floats] }
  obj/<element>/<element>.obj             base geometry
  obj/<element>/archives/<...>.obj        archived primitive geometry
```

* **Element** — the base `geomObjFile` is instanced once at `transformMatrix`,
  plus any whole-element `instancedCopies`.
* **Archives** — each `instancedPrimitiveJsonFiles` entry points at an archive
  JSON that maps an OBJ path to a large map of named per-instance transforms (the
  millions of scattered palms/rocks/debris). Each archive OBJ becomes one BLAS;
  every transform becomes one TLAS instance.
* **Matrices** — 16 floats, **column-major**, translation in slots `[12..14]`.
  This is applied directly; pass `--moanaTranspose` if a particular element looks
  sheared.
* **Geometry** is OBJ; the loader caches each OBJ → mesh so repeated archive
  references share a single BLAS.
* **Materials** are assigned by an element-name heuristic (sand / foliage / wood /
  rock / water). The real per-face Ptex materials are **not** parsed — this is a
  geometry/instancing renderer, not a full shading-network pipeline.

### Flags

| Flag | Meaning |
|------|---------|
| `--moana <dir>` | path to the unpacked `island/` root (or a single element `.json`) |
| `--maxInst <N>` | cap total instances (RAM guard); `0` = unlimited |
| `--moanaTranspose` | transpose every matrix (convention fallback) |
| `--width/--height/--spp/--bounces/--clamp/--exposure` | usual render controls |

Memory rule of thumb: each instance costs ~128 B (instance descriptor + shading
record) on top of the BLAS/TLAS. ~4 M instances ≈ a few GB; start with
`--maxInst 4000000` on a 16 GB laptop and raise it as headroom allows.

---

## Validation without the full data set

Because the 93 GB data won't fit everywhere, the loader is validated against a
**faithful miniature** in the exact same directory/JSON layout, generated by
`research/make_moana_mini.py`:

```bash
python3 research/make_moana_mini.py /tmp/moana_mini
.build/release/pathtracer --scene moana --moana /tmp/moana_mini \
    --width 800 --height 450 --spp 64 --out moana_mini.png
```

It writes three elements (`isBeach` ground + `instancedCopies`, `isPandanus`
with a 520-instance archive, `isMountainRock` with a 140-instance archive) using
column-major matrices, then loads them through the real code path. The result
(`docs/moana_loader_validation.png`) shows every instance placed upright with the
correct per-instance scale/rotation — confirming the element-JSON parsing, the
archive parsing, the matrix convention, the OBJ cache, the `instancedCopies` path
and the BLAS/TLAS build are all correct. The same code reads the real `island/`
directory unchanged; only the data size differs.
