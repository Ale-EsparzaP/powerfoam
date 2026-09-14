# Local copy, patched for this checkout only -- ORIGINAL at
# /home/rajehyl/spp_eval/eval_surface_chamfer.py (read-only, not ours to edit; see
# CLAUDE.md's convention: copy + fix locally rather than modify a coauthor's file).
#
# WHY A COPY WAS NEEDED (not just a PYTHONPATH override): the original hardcodes
# `sys.path.insert(0, "/home/rajehyl/powerfoam")`, which makes it ALWAYS import the
# coauthor's `configs`/`powerfoam` package regardless of PYTHONPATH -- fine for scoring an
# existing baseline checkpoint (old Params schema), but it silently uses the WRONG Params
# dataclass (missing density_activation/facet_normal_weight) for our own new arms' checkpoints.
# Patched here to prefer OUR checkout. The `SCENES` scene->split dict (originally imported from
# the coauthor's run_cluster_classify_eval.py) is inlined below instead of imported, so this
# file has no runtime dependency on any file under /home/rajehyl.
#
# NOTE (2026-09-15): rewritten after a checkout-wipe incident (see
# MyResearchVault/Stage0-Surface-Prior-Art.md); content restored from this session's own
# context, unchanged from the version validated end to end that night.
"""Path A: protocol-matched surface extraction + Chamfer Distance for PowerFoam.

Follows the de-facto standard established by 2DGS and used by GOF / TrimGS, so the number
is directly comparable to published Gaussian-splatting geometry results:

  1. render a MEDIAN-DEPTH map per training view (depth where accumulated transmittance
     crosses 0.5 -- not expected depth, which smears across semi-transparent primitives),
  2. TSDF-fuse the depth maps volumetrically (Open3D ScalableTSDFVolume),
  3. marching cubes -> mesh,
  4. sample points on the mesh and compute Chamfer Distance against the ScanNet GT points.

PowerFoam already computes exactly the quantity 2DGS fuses: rasterize.py's ray kernel
solves `next_trans < options.depth_quantile` analytically as
`t_near + log(trans / q) / sigma`, so depth_quantile=0.5 IS the median depth. NOTE the
renderer's VisOptions is a warp struct that zero-initializes, and the default built inside
Rasterizer.visualize() never sets depth_quantile -- leaving it 0.0 makes the crossing test
never fire and depth silently render all-zero. This bug already bit unproject_lerf_gt.py,
so the options struct is always constructed explicitly here.

Reported metrics follow the standard accuracy/completeness decomposition:
  accuracy      = mean over reconstructed points of distance to nearest GT point
  completeness  = mean over GT points of distance to nearest reconstructed point
  chamfer-L1    = (accuracy + completeness) / 2
plus precision/recall/F-score at a distance threshold (default 5 cm, the ScanNet
convention). GT points outside the reconstructed region are optionally cropped by
distance-to-nearest-camera so unobserved geometry is not charged against completeness.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/x_pelcasae/feature-foam-lifting/src")
sys.path.insert(0, "/home/x_pelcasae/powerfoam")

import numpy as np
import torch
import open3d as o3d
import configargparse
import warp as wp

from configs import Params, add_group
from data_loader import DataHandler
from powerfoam.scene import PowerfoamScene
from powerfoam.rasterize import VisOptions

# Inlined verbatim from /home/rajehyl/powerfoam/run_cluster_classify_eval.py (SCENES dict),
# to remove the runtime import dependency on that file.
SCENES = {
    "scene0000_00": "train", "scene0062_00": "train", "scene0070_00": "train",
    "scene0097_00": "train", "scene0140_00": "train", "scene0200_00": "train",
    "scene0347_00": "train", "scene0400_00": "train", "scene0590_00": "train",
    "scene0645_00": "val",
}


def cos_map(camera):
    """Per-pixel cosine between the ray direction and the camera's optical axis, needed
    because the renderer marches along NORMALIZED ray directions (so its depth is ray
    distance from the eye), while Open3D's TSDF integrator expects planar z-depth."""
    device = camera.eye.device if hasattr(camera.eye, "device") else "cpu"
    forward = torch.cross(camera.up, camera.right, dim=-1)
    forward = forward / torch.norm(forward)
    h, w = camera.height, camera.width
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    x = 2.0 * j / (float(w) - 1.0) - 1.0
    y = 1.0 - 2.0 * i / (float(h) - 1.0)
    ray_dir = (x[:, :, None] * camera.right[None, None, :]
               + y[:, :, None] * camera.up[None, None, :]
               + forward[None, None, :])
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
    cos = (ray_dir * forward[None, None, :]).sum(dim=-1)
    return cos.cpu().numpy()


def camera_to_o3d(camera):
    h, w = camera.height, camera.width
    forward = torch.cross(camera.up, camera.right, dim=-1)
    forward = forward / torch.norm(forward)
    fx = w / (2.0 * torch.norm(camera.right))
    fy = h / (2.0 * torch.norm(camera.up))
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, float(fx), float(fy), w / 2.0, h / 2.0)
    right = camera.right / torch.norm(camera.right)
    up = camera.up / torch.norm(camera.up)
    R = torch.stack([right, -up, forward], dim=0).cpu().numpy().astype(np.float64)
    t = -R @ camera.eye.cpu().numpy().astype(np.float64)
    extr = np.eye(4)
    extr[:3, :3] = R
    extr[:3, 3] = t
    return intr, extr


def render_median_depths(model, cameras, max_views=None, min_alpha=0.5, quantile=0.5):
    """Yield (index, camera, HxW median-depth) per view, with depth zeroed wherever the
    ray did not actually terminate on solid geometry.

    The alpha mask is essential, not cosmetic: without it every pixel gets a depth,
    including rays through empty space that merely accumulate enough density to cross the
    0.5 transmittance quantile."""
    n = len(cameras) if max_views is None else min(max_views, len(cameras))
    for i in range(n):
        cam = cameras[i]
        dq = quantile * torch.ones(cam.height, cam.width, 1, device=model.device)
        with torch.no_grad():
            result = model.forward(cam, depth_quantiles=dq)
        alpha = result[1]
        depth = result[4][:, :, 0]
        valid = (depth > 0) & (alpha > min_alpha)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        yield i, cam, depth.cpu().numpy()


def chamfer(rec_pts, gt_pts, thresh=0.05):
    rec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rec_pts))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    d_rec2gt = np.asarray(rec.compute_point_cloud_distance(gt))     # accuracy
    d_gt2rec = np.asarray(gt.compute_point_cloud_distance(rec))     # completeness
    acc, comp = float(d_rec2gt.mean()), float(d_gt2rec.mean())
    prec = float((d_rec2gt < thresh).mean())
    rec_r = float((d_gt2rec < thresh).mean())
    f1 = 2 * prec * rec_r / max(prec + rec_r, 1e-9)
    return {"accuracy": acc, "completeness": comp, "chamfer_l1": (acc + comp) / 2,
            "precision": prec, "recall": rec_r, "fscore": f1,
            "acc_median": float(np.median(d_rec2gt)), "comp_median": float(np.median(d_gt2rec))}


def main():
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="scene0000_00")
    p.add_argument("--variant", default="nonfrozen")
    p.add_argument("--ckpt-dir", default=None,
                   help="explicit checkpoint dir, bypassing the variant convention")
    p.add_argument("--gt-root", default="/home/rajehyl/scannet_gt")
    p.add_argument("--voxel", type=float, default=0.02, help="TSDF voxel size (m)")
    p.add_argument("--sdf-trunc", type=float, default=0.08, help="TSDF truncation (m)")
    p.add_argument("--depth-trunc", type=float, default=6.0, help="max usable depth (m)")
    p.add_argument("--max-views", type=int, default=None)
    p.add_argument("--n-sample", type=int, default=1_000_000, help="points sampled on the mesh")
    p.add_argument("--crop-dist", type=float, default=0.15,
                   help="drop GT points farther than this from any reconstructed point when "
                        "computing completeness (0 disables); guards against charging "
                        "never-observed geometry against the reconstruction")
    p.add_argument("--thresh", type=float, default=0.05, help="F-score distance threshold (m)")
    p.add_argument("--depth-quantile", type=float, default=0.5,
                   help="transmittance crossing used as the surface. 0.5 = median (the 2DGS "
                        "convention). Lower values take the FRONT of a soft density slab, which "
                        "matters for a volumetric representation whose cells are several cm thick.")
    p.add_argument("--min-alpha", type=float, default=0.5,
                   help="minimum accumulated opacity for a pixel's depth to be fused")
    p.add_argument("--out-mesh", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda"
    scene = args.scene
    ckpt_dir = args.ckpt_dir or f"output/scannet_{scene}_{args.variant}"

    wp.init()
    parser = configargparse.ArgParser()
    add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True)
    cargs = parser.parse_args(["-c", f"{ckpt_dir}/config.yaml"])
    dh = DataHandler(cargs)
    dh.reload("all", downsample=cargs.downsample[-1])
    model = PowerfoamScene(cargs)
    model.initialize_from_dataset(dh, device=device)
    model.load_pt(f"{ckpt_dir}/model.pt")
    model.update_vis_cache()
    cameras = dh.cameras

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

    n_used = 0
    for i, cam, depth in render_median_depths(model, cameras, args.max_views, args.min_alpha, args.depth_quantile):
        valid = np.isfinite(depth) & (depth > 0)
        if valid.sum() < 100:
            continue
        d = np.where(valid, depth, 0.0).astype(np.float32) * cos_map(cam)
        intr, extr = camera_to_o3d(cam)
        color = o3d.geometry.Image(np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(d), depth_scale=1.0,
            depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, extr)
        n_used += 1
        if n_used % 25 == 0:
            print(f"  fused {n_used} views", flush=True)
    print(f"[{scene}] fused {n_used} median-depth views (voxel={args.voxel}, trunc={args.sdf_trunc})")

    mesh = volume.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    print(f"  mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    if args.out_mesh:
        o3d.io.write_triangle_mesh(args.out_mesh, mesh)
        print(f"  wrote {args.out_mesh}")
    if len(mesh.triangles) == 0:
        print("  EMPTY MESH -- check depth rendering / camera convention")
        return

    rec = np.asarray(mesh.sample_points_uniformly(number_of_points=args.n_sample).points)
    split = SCENES[scene]
    gt = np.load(Path(args.gt_root) / split / scene / "coord.npy").astype(np.float64)

    res_full = chamfer(rec, gt, args.thresh)
    out = {"scene": scene, "variant": args.variant, "views_fused": n_used,
           "voxel": args.voxel, "sdf_trunc": args.sdf_trunc,
           "n_vertices": len(mesh.vertices), "n_triangles": len(mesh.triangles),
           "full": res_full}
    print(f"\n[{scene}] FULL GT: acc={res_full['accuracy']*100:.2f}cm "
          f"comp={res_full['completeness']*100:.2f}cm CD-L1={res_full['chamfer_l1']*100:.2f}cm "
          f"F@{args.thresh*100:.0f}cm={res_full['fscore']:.3f}")

    if args.crop_dist > 0:
        recpc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rec))
        gtpc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))
        dists = np.asarray(gtpc.compute_point_cloud_distance(recpc))
        keep = dists < args.crop_dist
        res_crop = chamfer(rec, gt[keep], args.thresh)
        out["cropped"] = {**res_crop, "gt_kept_fraction": float(keep.mean()),
                          "crop_dist": args.crop_dist}
        print(f"[{scene}] CROPPED to observed ({keep.mean()*100:.1f}% of GT kept): "
              f"acc={res_crop['accuracy']*100:.2f}cm comp={res_crop['completeness']*100:.2f}cm "
              f"CD-L1={res_crop['chamfer_l1']*100:.2f}cm F={res_crop['fscore']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
