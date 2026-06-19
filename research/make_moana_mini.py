#!/usr/bin/env python3
"""Generate a *faithful* miniature of the Disney Moana Island Scene directory
layout so the Swift `MoanaLoader` can be validated end-to-end without the real
~140 GB data set.

It reproduces the exact nesting the real data uses:

    <root>/
      json/<element>/<element>.json            # element: transformMatrix, geomObjFile,
                                                #   instancedCopies, instancedPrimitiveJsonFiles
      json/<element>/<element>_<archive>.json   # archive: objPath -> { instanceName: [16 floats] }
      obj/<element>/<element>.obj               # base geometry
      obj/<element>/archives/<archive>_geo.obj  # archived primitive geometry

Matrices are written as 16 floats in **column-major** order with the translation
in slots [12..14] and 1.0 in [15] — the Moana convention the loader assumes.

Usage:  python3 research/make_moana_mini.py [out_dir]
"""
import json, math, os, random, sys

random.seed(7)
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moana_mini"


def col_major(scale, yaw, tx, ty, tz):
    """4x4 TRS (rotate about Y, uniform scale) as 16 column-major floats."""
    c, s = math.cos(yaw), math.sin(yaw)
    # columns: X, Y, Z basis (scaled+rotated), then translation
    return [
        scale * c, 0.0, scale * -s, 0.0,   # column 0 (X)
        0.0,       scale, 0.0,      0.0,   # column 1 (Y)
        scale * s, 0.0, scale * c,  0.0,   # column 2 (Z)
        tx,        ty,  tz,         1.0,   # column 3 (translation)
    ]


def write_obj(path, verts, faces):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def ground_quad(half=70.0, y=0.0):
    v = [(-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half)]
    return v, [(0, 1, 2), (0, 2, 3)]


def cone(radius=0.9, height=2.6, sides=8, ybase=0.0):
    v = [(0.0, ybase + height, 0.0)]  # apex
    for i in range(sides):
        a = 2 * math.pi * i / sides
        v.append((radius * math.cos(a), ybase, radius * math.sin(a)))
    v.append((0.0, ybase, 0.0))  # base center
    faces = []
    for i in range(sides):
        nxt = 1 + (i + 1) % sides
        faces.append((0, 1 + i, nxt))            # side
        faces.append((sides + 1, nxt, 1 + i))    # bottom
    return v, faces


def blob(radius=0.9, squash=0.55):
    v = [(radius, 0, 0), (-radius, 0, 0), (0, radius * squash, 0),
         (0, -radius * squash, 0), (0, 0, radius), (0, 0, -radius)]
    f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return v, f


def make_element_with_archive(name, geo_verts, geo_faces, n_inst, spread,
                              scale_range, base_geom=None):
    """Write one element directory: optional base geometry + one instanced archive."""
    eldir = os.path.join(ROOT, "json", name)
    os.makedirs(eldir, exist_ok=True)

    arch_obj_rel = f"obj/{name}/archives/{name}_geo.obj"
    write_obj(os.path.join(ROOT, arch_obj_rel), geo_verts, geo_faces)

    # archive JSON: objPath -> { instanceName: [16 floats] }
    insts = {}
    for k in range(n_inst):
        x = random.uniform(-spread, spread)
        z = random.uniform(-spread, spread)
        sc = random.uniform(*scale_range)
        yaw = random.uniform(0, 2 * math.pi)
        insts[f"{name}:inst_{k:04d}"] = col_major(sc, yaw, x, 0.0, z)
    archive_rel = f"json/{name}/{name}_xg.json"
    with open(os.path.join(ROOT, archive_rel), "w") as f:
        json.dump({arch_obj_rel: insts}, f)

    element = {
        "name": name,
        "transformMatrix": col_major(1.0, 0.0, 0.0, 0.0, 0.0),
        "instancedPrimitiveJsonFiles": {
            "archive_xg": {"type": "archive", "jsonFile": archive_rel}
        },
    }
    if base_geom is not None:
        base_rel = f"obj/{name}/{name}.obj"
        write_obj(os.path.join(ROOT, base_rel), base_geom[0], base_geom[1])
        element["geomObjFile"] = base_rel
        # exercise the instancedCopies path with one shifted whole-element copy
        element["instancedCopies"] = {
            f"{name}_copy1": {"transformMatrix": col_major(1.0, 0.0, 0.0, 0.0, 0.0)}
        }
    with open(os.path.join(eldir, f"{name}.json"), "w") as f:
        json.dump(element, f, indent=1)


def main():
    if os.path.exists(ROOT):
        import shutil
        shutil.rmtree(ROOT)
    # isBeach: ground geometry (sand) — base geom + instancedCopies, no archive trees
    beach_dir = os.path.join(ROOT, "json", "isBeach")
    os.makedirs(beach_dir, exist_ok=True)
    gv, gf = ground_quad()
    write_obj(os.path.join(ROOT, "obj/isBeach/isBeach.obj"), gv, gf)
    beach = {
        "name": "isBeach",
        "transformMatrix": col_major(1.0, 0.0, 0.0, 0.0, 0.0),
        "geomObjFile": "obj/isBeach/isBeach.obj",
        "instancedCopies": {
            "isBeach_copy1": {"transformMatrix": col_major(1.0, 0.0, 0.0, -0.05, 0.0)}
        },
    }
    with open(os.path.join(beach_dir, "isBeach.json"), "w") as f:
        json.dump(beach, f, indent=1)

    # isPandanus: ~500 instanced green plants (cones) scattered on the beach
    cv, cf = cone()
    make_element_with_archive("isPandanus", cv, cf, n_inst=520, spread=60.0,
                              scale_range=(0.7, 1.5))

    # isMountainRock: ~140 instanced rocks (grey) clustered toward the centre
    bv, bf = blob()
    make_element_with_archive("isMountainRock", bv, bf, n_inst=140, spread=34.0,
                              scale_range=(0.8, 2.4))

    # report
    n_files = sum(len(fs) for _, _, fs in os.walk(ROOT))
    print(f"wrote synthetic Moana-format scene to {ROOT}")
    print(f"  elements: isBeach (ground+copy), isPandanus (520 plants), isMountainRock (140 rocks)")
    print(f"  files: {n_files}")


if __name__ == "__main__":
    main()
