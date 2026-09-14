"""Isolated test: does facet_normal_loss alone recover accurate normals from a REAL trained
checkpoint's geometry (points/radii/density), with NO rendering and NO other loss term?

This is deliberately different from the synthetic plane/sphere/corner test (which validates the
math on toy geometry) -- it asks whether the SIGNAL is strong enough on REAL ScanNet occupancy
to beat the ~50 deg baseline the full photometric pipeline currently achieves for the dipole.
Freshly re-randomizes the quaternions (simulating "no orientation learned yet") and optimizes
ONLY them against facet_normal_loss, using the checkpoint's own trained points/radii/density
as a FIXED, real occupancy field.

RESULT (2026-09-14): the loss always converges to exactly 0 (the construction is internally
self-consistent) but GT normal error does not improve on the unfrozen checkpoint (60.07->60.08
deg) and gets WORSE than random on the frozen checkpoint (60.01->71.62 deg, still 65.07 deg even
restricting to only the highest-confidence occupied/empty transitions via a stricter
min_contrast). Ruled out as a units bug (alpha is a real, non-degenerate bimodal distribution).
The occupied/empty pattern in a trained density field does not track the true surface, even
where positions ARE the GT surface. See MyResearchVault/Stage0-Surface-Prior-Art.md.

GPU float32 (not CPU float64): an earlier CPU/float64 attempt at this same test took over two
hours without finishing at 328k primitives / 4.4M edges and was killed; this version completes
in seconds on GPU with float32, which is more than adequate precision for this comparison.
"""
import argparse, json, numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from powerfoam.facet_normal import facet_normal_loss, facet_normal_target


def get_normals(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-20)


def angular_error_deg(n_est, n_gt):
    c = (n_est * n_gt).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(torch.arccos(c)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-contrast", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    d = torch.load(args.recon, map_location="cpu")
    points = d["points"].float().cuda()
    radii = F.softplus(d["radii"].float(), beta=100).cuda()
    density = F.softplus(d["density"].float(), beta=100).cuda()  # these checkpoints trained w/ softplus
    adjacency, offsets = d["adjacency"].cuda(), d["adjacency_offsets"].cuda()
    n = points.shape[0]
    print(f"primitives: {n:,}  edges: {adjacency.shape[0]:,}")

    gt_xyz = torch.from_numpy(np.load(f"{args.gt_dir}/coord.npy")).float()
    gt_nrm = torch.from_numpy(np.load(f"{args.gt_dir}/normal.npy")).float()
    gt_nrm = gt_nrm / gt_nrm.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    _, idx = cKDTree(points.cpu().numpy()).query(gt_xyz.numpy(), k=1, workers=-1)
    idx = torch.from_numpy(idx).long().cuda()
    gt_nrm = gt_nrm.cuda()

    # How many primitives even HAVE a resolvable target, at this min_contrast?
    _, included = facet_normal_target(points, density, radii, adjacency, offsets,
                                       min_contrast=args.min_contrast)
    print(f"primitives with resolvable target: {int(included.sum()):,} / {n:,} "
          f"({100*included.float().mean():.1f}%)")

    q = torch.randn(n, 4, dtype=torch.float32, device="cuda", requires_grad=True)
    with torch.no_grad():
        q /= q.norm(dim=-1, keepdim=True)

    err0 = angular_error_deg(get_normals(q.detach())[idx], gt_nrm).numpy()
    print(f"BEFORE optimization: median {np.median(err0):.2f} deg  mean {err0.mean():.2f} deg  "
          f"(random baseline check, should be ~59-60)")

    opt = torch.optim.Adam([q], lr=args.lr)
    history = []
    for step in range(args.steps):
        opt.zero_grad()
        normals = get_normals(q)
        loss = facet_normal_loss(normals, points, density, radii, adjacency, offsets,
                                  min_contrast=args.min_contrast)
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            with torch.no_grad():
                err = angular_error_deg(get_normals(q)[idx], gt_nrm).numpy()
            print(f"  step {step:5d}  loss {loss.item():.5f}  median_err {np.median(err):.2f} deg")
            history.append(dict(step=step, loss=float(loss.item()), median_err_deg=float(np.median(err))))

    err_final = angular_error_deg(get_normals(q.detach())[idx], gt_nrm).numpy()
    result = dict(
        num_primitives=n, num_edges=int(adjacency.shape[0]),
        pct_with_target=float(100 * included.float().mean()),
        err_before_median=float(np.median(err0)), err_before_mean=float(err0.mean()),
        err_after_median=float(np.median(err_final)), err_after_mean=float(err_final.mean()),
        err_after_within10=float((err_final < 10).mean() * 100),
        err_after_within30=float((err_final < 30).mean() * 100),
        history=history,
    )
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nAFTER: median {result['err_after_median']:.2f} deg  mean {result['err_after_mean']:.2f} deg  "
          f"<10deg {result['err_after_within10']:.2f}%  <30deg {result['err_after_within30']:.2f}%")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
