"""Plain RGB render of a gsplat LERF checkpoint -- the 3DGS-arm counterpart of
../render_rgb_powerfoam.py, and the reference panel beside render_seg_gsplat_ply.py's
class-coloured one.

Full spherical-harmonic evaluation (sh_degree=3, colors = cat([sh0, shN])),
unlike render_seg_gsplat_ply.py's background, which only needs the DC term
because it is desaturated to luma anyway. This panel IS the reference render,
so it should look like the real reconstruction.

LERF checkpoints only (gsplat simple_trainer's ckpt_*.pt + the scene's COLMAP
dir for cameras) -- see colmap_cameras.py for why the ScanNet++ .ply path
render_seg_gsplat_ply.py also supports is not mirrored here.
"""
import argparse
import os
import sys

import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_cameras import load_cameras_colmap  # noqa: E402


def load_gsplat_ckpt(path, device):
    """Same activation convention as render_seg_gsplat_ply.py::load_gsplat_ckpt
    (scales log-space, opacities logits, quats unnormalised) -- verified there,
    reused here rather than re-verified."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sp = ck["splats"]
    quats = sp["quats"].to(device)
    colors = torch.cat([sp["sh0"], sp["shN"]], dim=1).to(device)  # (N, 16, 3), degree 3
    return (sp["means"].to(device),
            quats / quats.norm(dim=-1, keepdim=True),
            torch.exp(sp["scales"].to(device)),
            torch.sigmoid(sp["opacities"].to(device)),
            colors)


def load_occam_ckpt(path, device):
    """Occam's LGS checkpoint -- see render_seg_gsplat_ply.py::load_occam_ckpt
    for the full tuple-layout derivation. This variant returns FULL SH
    (features_dc + features_rest, 16 coeffs) rather than the DC-only term,
    since this script IS the reference render."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    params, _first_iter = ck
    xyz, f_dc, f_rest, scaling, rotation, opacity = params[1:7]
    quats = rotation.detach().to(device)
    colors = torch.cat([f_dc, f_rest], dim=1).detach().to(device)
    return (xyz.detach().to(device),
            quats / quats.norm(dim=-1, keepdim=True),
            torch.exp(scaling.detach().to(device)),
            torch.sigmoid(opacity.detach().to(device))[:, 0],
            colors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", help="gsplat simple_trainer checkpoint")
    ap.add_argument("--occam", help="Occam's LGS checkpoint (chkpnt*_langfeat_*.pth)")
    ap.add_argument("--colmap", required=True, help="scene dir holding sparse/0")
    ap.add_argument("--views", default="0")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    dev = "cuda"

    if bool(args.ckpt) == bool(args.occam):
        raise SystemExit("pass exactly one of --ckpt or --occam")
    means, quats, scales, opac, colors = (load_gsplat_ckpt(args.ckpt, dev) if args.ckpt
                                          else load_occam_ckpt(args.occam, dev))
    cams = load_cameras_colmap(args.colmap)
    idx = ([int(v) for v in args.views.split(",")] if args.views.strip().lower() != "all"
           else list(range(len(cams))))
    os.makedirs(args.outdir, exist_ok=True)
    import imageio.v2 as imageio
    for i in idx:
        if i >= len(cams):
            print(f"  [skip view {i}] only {len(cams)} cameras"); continue
        vm, K, W, H, name = cams[i]
        with torch.no_grad():
            img, _alpha, _ = rasterization(
                means, quats, scales, opac, colors,
                torch.tensor(vm, dtype=torch.float32, device=dev)[None],
                torch.tensor(K, dtype=torch.float32, device=dev)[None],
                W, H, sh_degree=3,
            )
        out = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        p = os.path.join(args.outdir, f"{args.scene}_rgb_view{i:04d}.png")
        imageio.imwrite(p, out)
        print(f"  wrote {p} ({name})")


if __name__ == "__main__":
    main()
