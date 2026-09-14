"""Export a SparseFeatureOperator-format COO operator from a frozen gsplat
3D-Gaussian-Splatting ScanNet baseline checkpoint (gaussian_baseline_scannet),
using the SAME camera views (same colmap train split, same intrinsics, same
resolution) that powerfoam's own export_feature_operator.py uses for the
matching PowerFoam truefrozen checkpoint on the same scene -- so the two
operators are comparable at matched view count and matched camera rays, not
just matched view count.

Camera info is loaded directly from the powerfoam colmap dataset via
pycolmap, replicating data_loader/colmap.py's exact train-split rule
(`sorted(names)[idx % 8 != 0]`), because the GS checkpoint itself
(ckpt_29999_rank0.pt) stores only Gaussian parameters, no camera poses.

`colors` are never used to compute the compositing weights in
export_view_operator (only row/col/value depend on means/quats/scales/
opacities) -- see export_gsplat_operator.py -- so a 1-channel dummy is used
to avoid materializing SH-evaluated RGB for a checkpoint we don't need to
render pixels from.

Usage:
    python export_gsplat_scannet_operator.py --scene scene0062_00 \
        --checkpoint /home/rajehyl/gaussian_baseline_scannet/scene0062_00/ckpts/ckpt_29999_rank0.pt \
        --colmap-dir /home/rajehyl/powerfoam/data/scannet/scene0062_00_colmap \
        --views all \
        --output /home/x_pelcasae/powerfoam/artifacts/scannet_scene0062_00_gsfroz/train_operator_full.pt
"""
import argparse
import os
import time
from pathlib import Path

import gsplat_env_gsview  # noqa: F401  must precede `import gsplat`

import numpy as np
import pycolmap
import torch

from export_gsplat_operator import export_view_operator


def load_train_views(colmap_dir):
    """Replicates data_loader/colmap.py's COLMAPDataset train split exactly:
    sorted image names, keep indices where idx % 8 != 0."""
    recon = pycolmap.Reconstruction()
    recon.read(os.path.join(colmap_dir, "sparse/0/"))
    if len(recon.cameras) > 1:
        raise ValueError("Multiple cameras are not supported")
    names = sorted(im.name for im in recon.images.values())
    idx = np.arange(len(names))
    train_names = list(np.array(names)[idx % 8 != 0])

    cam = list(recon.cameras.values())[0]
    width, height = cam.width, cam.height
    K = torch.tensor(cam.calibration_matrix(), dtype=torch.float32)

    images_by_name = {im.name: im for im in recon.images.values()}
    viewmats = []
    for name in train_names:
        im = images_by_name[name]
        w2c = im.cam_from_world().matrix()  # (3,4)
        w2c4 = np.eye(4, dtype=np.float32)
        w2c4[:3, :4] = w2c
        viewmats.append(torch.tensor(w2c4, dtype=torch.float32))
    viewmats = torch.stack(viewmats)  # (N, 4, 4)
    return train_names, viewmats, K, width, height


def parse_views(views_arg, num_available):
    if views_arg is None or views_arg.strip().lower() == "all":
        return list(range(num_available))
    indices = [int(v) for v in views_arg.split(",") if v.strip() != ""]
    for idx in indices:
        if idx < 0 or idx >= num_available:
            raise ValueError(f"view index {idx} out of range [0, {num_available})")
    return indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--colmap-dir", required=True)
    ap.add_argument("--views", default="all", help="comma list of TRAIN-SPLIT-LOCAL indices, or 'all'")
    ap.add_argument("--max_hits_per_pixel", type=int, default=64)
    ap.add_argument("--transmittance_floor", type=float, default=1e-3,
                     help="matches powerfoam export_feature_operator.py's --transmittance_threshold default (1e-3)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = args.device
    train_names, viewmats, K, width, height = load_train_views(args.colmap_dir)
    num_available = len(train_names)
    indices = parse_views(args.views, num_available)
    print(f"[export_gsplat_scannet_operator] scene={args.scene} train_split_size={num_available} "
          f"views_used={len(indices)} resolution={width}x{height}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    splats = ckpt["splats"]
    means = splats["means"].to(device)
    quats = splats["quats"].to(device)
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    num_primitives = means.shape[0]
    colors = torch.zeros(num_primitives, 1, device=device)  # dummy: not used by weight computation
    K = K.to(device)

    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij",
    )
    pixel_grid = torch.stack([pixel_y.reshape(-1), pixel_x.reshape(-1)], dim=-1)  # (H*W, 2)

    all_rows, all_cols, all_vals, all_view_ids, all_pixels = [], [], [], [], []
    t0 = time.time()
    for local_id, view_idx in enumerate(indices):
        viewmat = viewmats[view_idx].to(device)
        row_indices, col_indices, values, _, _ = export_view_operator(
            means, quats, scales, opacities, colors, viewmat, K, width, height,
            max_hits_per_pixel=args.max_hits_per_pixel,
            transmittance_floor=args.transmittance_floor,
        )
        all_rows.append(row_indices + local_id * height * width)
        all_cols.append(col_indices)
        all_vals.append(values)
        all_view_ids.append(torch.full((height * width,), local_id, dtype=torch.long))
        all_pixels.append(pixel_grid.cpu())
        print(f"[export_gsplat_scannet_operator] view {local_id + 1}/{len(indices)} "
              f"(train-local {view_idx}, {train_names[view_idx]}) nnz={row_indices.numel()}")

    elapsed = time.time() - t0
    state = {
        "row_indices": torch.cat(all_rows).cpu(),
        "col_indices": torch.cat(all_cols).cpu(),
        "values": torch.cat(all_vals).cpu(),
        "num_rows": len(indices) * height * width,
        "num_primitives": num_primitives,
        "row_view_ids": torch.cat(all_view_ids).cpu(),
        "row_pixels": torch.cat(all_pixels).cpu(),
    }
    print(f"[export_gsplat_scannet_operator] TIMING elapsed_sec={elapsed:.3f} total_nnz={state['row_indices'].numel()} "
          f"expected_rows={state['num_rows']}")
    assert state["row_indices"].numel() == state["values"].numel() == state["col_indices"].numel()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[export_gsplat_scannet_operator] wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
