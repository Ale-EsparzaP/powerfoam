"""GT panel for the LERF-OVS segmentation comparison figure: the 2D polygon
ground truth, in image space -- the third arm alongside PowerFoam and the
SFS/3DGS baseline.

Two renders per labeled frame: the plain reference photo (the one
splat-distiller's own eval scores against), and a class-coloured version using
the SAME palette this project already uses for the 3D panels
(seg_palette._HUES), so a colour means the same class in every column of the
figure.

GT format (LangSplat's LERF-OVS release, `<lerf_ovs>/label/<scene>/<frame>.json`
-- installed locally at data/lerf_ovs_gt_label/<scene>/, see the download note
in run_lerf_seg_figure.py): one polygon per labeled INSTANCE, not one mask per
class -- several instances can share a category name (waldo_kitchen's
frame_00089 has five separate "knife" polygons). Painted painter's-algorithm
ordered by ascending `layer` (the GT's own z-order), matching
unproject_lerf_gt.py::rasterize_frame_labels's convention.

Query selection: a single frame can carry up to 18 distinct categories, far
more than the project's 8-hue budget (seg_palette._HUES). Rather than picking
a subset by hand, this ranks categories by total labelled pixel AREA (summed
over their instances, using the json's own `area` field) and keeps the
largest 8 -- the same "highlight some classes, desaturate the rest" rule
query_colours already applies in the 3D panels, just applied to as many
classes as the palette can carry instead of to one or two. Whatever is left
goes to the same grey bucket as real background.

The returned `queries` list (rank order) is the contract with the other two
panels: passing it verbatim as --queries to render_seg_powerfoam.py and
render_seg_gsplat_ply.py is what makes a colour mean the same category in all
three columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seg_palette import GREY, _HUES, _hex_to_rgb  # noqa: E402


def rank_categories(objects, max_classes=8):
    """Distinct category names, ranked by total labelled area, largest first."""
    area = OrderedDict()
    for o in objects:
        c = o["category"]
        area[c] = area.get(c, 0.0) + float(o.get("area", 0.0))
    ranked = sorted(area, key=lambda c: -area[c])
    return ranked[:max_classes], ranked[max_classes:]


def render_gt_panel(scene, frame, label_root, outdir, queries=None, max_classes=8):
    info = json.load(open(os.path.join(label_root, scene, f"{frame}.json")))
    objects = info["objects"]
    dropped = []
    if queries is None:
        queries, dropped = rank_categories(objects, max_classes)

    plain = Image.open(os.path.join(label_root, scene, f"{frame}.jpg")).convert("RGB")
    W, H = plain.size
    plain_arr = np.asarray(plain, dtype=np.float32) / 255.0

    qidx = {q: i for i, q in enumerate(queries)}
    canvas = Image.new("I", (W, H), -1)
    draw = ImageDraw.Draw(canvas)
    for o in sorted(objects, key=lambda o: o.get("layer", 0)):
        idx = qidx.get(o["category"], -1)
        if idx < 0:
            continue
        poly = [(float(x), float(y)) for x, y in o["segmentation"]]
        if len(poly) < 3:
            continue
        draw.polygon(poly, fill=idx)
    label_canvas = np.array(canvas, dtype=np.int64)

    palette = np.array([_hex_to_rgb(_HUES[i % len(_HUES)]) for i in range(len(queries))],
                       dtype=np.float32)
    luma = (plain_arr * np.array([0.2126, 0.7152, 0.0722])).sum(-1, keepdims=True)
    coloured = np.repeat(luma, 3, axis=2).copy()
    hit = label_canvas >= 0
    if queries:
        coloured[hit] = palette[label_canvas[hit]]

    os.makedirs(outdir, exist_ok=True)
    import imageio.v2 as imageio
    plain_path = os.path.join(outdir, f"{scene}_{frame}_gt_plain.png")
    coloured_path = os.path.join(outdir, f"{scene}_{frame}_gt_coloured.png")
    imageio.imwrite(plain_path, (plain_arr * 255).astype(np.uint8))
    imageio.imwrite(coloured_path, (np.clip(coloured, 0, 1) * 255).astype(np.uint8))

    per_query = {q: int((label_canvas == i).sum()) for i, q in enumerate(queries)}
    legend = [(q, tuple(float(c) for c in palette[i])) for i, q in enumerate(queries)]
    meta = {
        "arm": "gt", "scene": scene, "frame": frame, "width": W, "height": H,
        "queries": queries, "dropped_categories": dropped,
        "num_instances": len(objects), "legend": legend,
        "per_query_px": per_query,
        "coloured_px_frac": float(hit.mean()),
        "plain_path": plain_path, "coloured_path": coloured_path,
    }
    with open(os.path.join(outdir, f"{scene}_{frame}_gt_labels.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frame", required=True, help="e.g. frame_00002 (no extension)")
    ap.add_argument("--label-root", default="data/lerf_ovs_gt_label")
    ap.add_argument("--max-classes", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    meta = render_gt_panel(a.scene, a.frame, a.label_root, a.outdir,
                           max_classes=a.max_classes)
    print(f"[gt] {a.scene}/{a.frame}: {meta['num_instances']} instances, "
          f"{len(meta['queries'])} coloured classes "
          f"({100*meta['coloured_px_frac']:.1f}% of pixels), "
          f"{len(meta['dropped_categories'])} dropped to grey")
    for q, n in sorted(meta["per_query_px"].items(), key=lambda kv: -kv[1]):
        print(f"       {q:<20} {n:>9,} px")


if __name__ == "__main__":
    main()
