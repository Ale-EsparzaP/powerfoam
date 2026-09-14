"""Does restricting the true power-diagram facet graph to the alpha complex
(`power_adjacency.build_power_adjacency(..., alpha_complex=True)`) remove the
long-range "sliver" edges that `truefrozen_facet_mechanism.py` found driving
the unanimous KNN-beats-true-facet-adjacency result?

TWO ZERO-TRAINING WAYS TO RECOVER SURFACE ORIENTATION FROM A REAL POWERFOAM CHECKPOINT.

IDEA 1 (texel-height slope). PowerFoam's own description: detail sites carry "displacement
values pushing surfaces locally along a dipole axis", with the dipole face a "macro-scale
geometry proxy". So per primitive, in its own orthonormal frame (t, b, n), each texel k sits at
(u_k, v_k) with displacement h_k along n -- all in units of the cell radius, since scene.py
scales both by radii. If the dipole plane is tilted w.r.t. the true surface, the fitted heights
must carry a systematic slope. Fit h = a*u + b*v + c; the surface's tangents are (1,0,a),(0,1,b),
so its normal is proportional to (-a,-b,1) in that frame, i.e.  n' ~ n - a*t - b*bb.

NOT CHEATING: texel_sites/texel_height are nn.Parameters initialised from noise/zeros and trained
ONLY by the photometric losses. No depth, normal, or mesh ground truth ever touches them. GT
normals appear here solely as the scoring side.

IDEA 2 (neighbourhood aggregation). If per-primitive orientation is a true normal plus zero-mean
noise, averaging over geometric neighbours cuts the error. Normals are UNORIENTED (error is
arccos|n.n_gt|), so naive vector averaging of sign-flipped neighbours cancels; aggregate instead
via the principal eigenvector of sum_j w_j n_j n_j^T, which is sign-invariant.

RESULT (2026-09-14, 3 scenes): idea 1 makes error WORSE in 3/3 scenes (the texel-height field
encodes appearance/parallax, not surface orientation). Idea 2 helps in 3/3 but by far less than
a zero-mean-noise model predicts (measured: 50.0->46.0 deg vs a predicted ~50->13 deg for 14
neighbours), proving the dipole's deviation from truth is SYSTEMATIC, not noise -- no post-hoc
denoiser can recover it. See MyResearchVault/Dipole-Denoising-Probe.md for the full write-up.

Scoring matches the coauthor's protocol: for each GT point take its nearest primitive's normal and
report median arccos|n_est . n_gt| in degrees, against a 60 deg unoriented-random baseline.
"""
import argparse, json, numpy as np, torch
from scipy.spatial import cKDTree


def frame_from_quaternions(q):
    """(n, t, b) exactly as PowerfoamScene.get_normals/get_tangents build them."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    t = torch.stack([2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)], -1)
    b = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)], -1)
    unit = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    return unit(n), unit(t), unit(b)


def texel_slope_normal(n, t, b, texel_sites, texel_height, ridge=1e-6):
    """IDEA 1. Least-squares h ~ a*u + b*v + c per primitive; returns corrected normal + fit info."""
    u, v = texel_sites[..., 0], texel_sites[..., 1]                 # (N,S) radius units
    A = torch.stack([u, v, torch.ones_like(u)], -1)                 # (N,S,3)
    M = A.transpose(1, 2) @ A                                       # (N,3,3)
    rhs = (A.transpose(1, 2) @ texel_height.unsqueeze(-1))          # (N,3,1)
    eye = torch.eye(3, dtype=M.dtype).expand_as(M)
    coef = torch.linalg.solve(M + ridge * eye, rhs).squeeze(-1)     # (N,3)
    a, bb = coef[:, 0], coef[:, 1]
    n_new = n - a.unsqueeze(-1) * t - bb.unsqueeze(-1) * b
    n_new = n_new / n_new.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    # conditioning of the 2x2 (u,v) scatter -- a degenerate texel layout cannot define a slope
    S = M[:, :2, :2]
    ev = torch.linalg.eigvalsh(S)
    cond_ok = (ev[:, 0] > 1e-8)
    tilt_deg = torch.rad2deg(torch.atan(torch.sqrt(a**2 + bb**2)))
    return n_new, cond_ok, tilt_deg


def aggregate(nrm, adjacency, offsets, weights=None, include_self=True):
    """IDEA 2. Sign-invariant neighbourhood aggregation: principal eigenvector of sum w n n^T."""
    N = nrm.shape[0]
    counts = (offsets[1:] - offsets[:-1]).long()
    src = torch.repeat_interleave(torch.arange(N, dtype=torch.long), counts)
    dst = adjacency.long()
    w = torch.ones(N, dtype=nrm.dtype) if weights is None else weights
    outer = nrm.unsqueeze(-1) * nrm.unsqueeze(-2)                   # (N,3,3)
    S = torch.zeros(N, 3, 3, dtype=nrm.dtype)
    S.index_add_(0, src, outer[dst] * w[dst].view(-1, 1, 1))
    if include_self:
        S += outer * w.view(-1, 1, 1)
    evals, evecs = torch.linalg.eigh(S)                             # ascending
    return evecs[:, :, -1], counts


def angular_error_deg(n_est, n_gt):
    """Unoriented angular error; 60 deg is the random baseline in 3D."""
    c = (n_est * n_gt).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(torch.arccos(c))


def report(name, err, out):
    q = np.percentile(err, [25, 50, 75])
    row = dict(median_deg=float(q[1]), p25=float(q[0]), p75=float(q[2]),
               mean_deg=float(err.mean()), within10=float((err < 10).mean() * 100),
               within30=float((err < 30).mean() * 100), n=int(err.size))
    out[name] = row
    print(f"  {name:<34} median {row['median_deg']:6.2f}  mean {row['mean_deg']:6.2f}  "
          f"<10deg {row['within10']:5.2f}%  <30deg {row['within30']:5.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    d = torch.load(args.recon, map_location="cpu")
    pts = d["points"].float()
    n, t, b = frame_from_quaternions(d["quaternions"].float())
    print(f"primitives: {pts.shape[0]:,}   texels/primitive: {d['texel_sites'].shape[1]}")

    gt_xyz = torch.from_numpy(np.load(f"{args.gt_dir}/coord.npy")).float()
    gt_nrm = torch.from_numpy(np.load(f"{args.gt_dir}/normal.npy")).float()
    gt_nrm = gt_nrm / gt_nrm.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    print(f"GT points: {gt_xyz.shape[0]:,}")

    # Frame sanity check: an unfrozen arm must still live in the GT coordinate frame.
    lo_p, hi_p = pts.min(0).values, pts.max(0).values
    lo_g, hi_g = gt_xyz.min(0).values, gt_xyz.max(0).values
    print(f"  primitive bbox {lo_p.tolist()} .. {hi_p.tolist()}")
    print(f"  GT        bbox {lo_g.tolist()} .. {hi_g.tolist()}")
    dist, idx = cKDTree(pts.numpy()).query(gt_xyz.numpy(), k=1, workers=-1)
    print(f"  GT->nearest-primitive distance: median {np.median(dist):.4f}  p90 {np.percentile(dist,90):.4f}")
    idx = torch.from_numpy(idx).long()

    results, fits = {}, {}
    print("\nangular error vs GT normals (unoriented; 60 deg = random):")

    # --- baselines -------------------------------------------------------
    rnd = torch.randn_like(n); rnd = rnd / rnd.norm(dim=-1, keepdim=True)
    report("random control", angular_error_deg(rnd[idx], gt_nrm).numpy(), results)
    err_dipole = angular_error_deg(n[idx], gt_nrm).numpy()
    report("dipole (raw, instrument check)", err_dipole, results)

    # --- idea 1 ----------------------------------------------------------
    n1, cond_ok, tilt = texel_slope_normal(n, t, b, d["texel_sites"].float(), d["texel_height"].float())
    fits["cond_ok_frac"] = float(cond_ok.float().mean())
    fits["tilt_deg_median"] = float(tilt.median())
    fits["texel_height_absmedian"] = float(d["texel_height"].float().abs().median())
    print(f"  [idea1] well-conditioned fits {fits['cond_ok_frac']*100:.2f}%   "
          f"median implied tilt {fits['tilt_deg_median']:.2f} deg   "
          f"median |texel_height| {fits['texel_height_absmedian']:.5f}")
    report("idea1 texel-slope corrected", angular_error_deg(n1[idx], gt_nrm).numpy(), results)

    # --- idea 2 ----------------------------------------------------------
    adj, off = d["adjacency"], d["adjacency_offsets"]
    n2, counts = aggregate(n, adj, off)
    print(f"  [idea2] neighbours/primitive: median {int(counts.median())}  mean {float(counts.float().mean()):.1f}")
    report("idea2 aggregated (raw dipole)", angular_error_deg(n2[idx], gt_nrm).numpy(), results)

    # --- 1 + 2 -----------------------------------------------------------
    n12, _ = aggregate(n1, adj, off)
    report("idea1+2 aggregated corrected", angular_error_deg(n12[idx], gt_nrm).numpy(), results)

    results["_fit_diagnostics"] = fits
    results["_meta"] = dict(recon=args.recon, gt_dir=args.gt_dir,
                            num_primitives=int(pts.shape[0]), num_gt=int(gt_xyz.shape[0]),
                            gt_to_prim_median=float(np.median(dist)))
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
