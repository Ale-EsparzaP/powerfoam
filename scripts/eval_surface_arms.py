"""Score the 2x2 surface arms on cheap, checkpoint-only geometry metrics.

WHAT THIS IS AND IS NOT. These metrics come straight out of model.pt and cost no rendering, so
they are fast enough to run on every arm. But `Experiment-F-scannet.md` records a case where
cell-centre DRIFT and extracted-mesh CD-L1 moved in OPPOSITE directions (drift 3.28 -> 2.81 while
CD-L1 went 6.15 -> 8.08 cm). So drift here is a PROXY ONLY. No geometry claim may rest on it --
CD-L1 from the TSDF/marching-cubes surface (eval_surface_chamfer.py) is required before any arm
is called better.

Metrics per arm:
  normals   median unoriented angular error of the dipole normal vs ScanNet normal.npy
            (nearest primitive per GT point; 60 deg = random baseline)
  drift     distance from primitive centres to the nearest GT point, in metres AND in cell radii
            (the coauthor's units: 0.06 radii frozen vs 3.49 at plr 1e-3 -- the number the
            position-LR fix is supposed to move)
  opacity   per-cell alpha = 1 - exp(-sigma * 2r), and its BIMODALITY -- the property VoroTracing
            attributes to exp density ("cells either near-transparent or near-opaque"). Reported
            as the fraction in the middle band, which should COLLAPSE if the claim holds.
  planarity local PCA planarity over primitive centres (lambda0/sum lambda); the measured foam
            value is 3-6% of cells below 0.05 vs 22-25% for Gaussian means.
Photometric PSNR/SSIM/LPIPS are read from the arm's own metrics.txt (written by train.py).
"""
import argparse, json, os, re, numpy as np, torch
import torch.nn.functional as F
from scipy.spatial import cKDTree


def frame_from_quaternions(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = torch.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-20)


def read_metrics_txt(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"Average (\w+):\s+([-\d.]+)", line.strip())
        if m:
            out[m.group(1).lower()] = float(m.group(2))
    return out


def planarity_fraction(pts_np, k=16, thresh=0.05, sample=40000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts_np), min(sample, len(pts_np)), replace=False)
    _, nb = cKDTree(pts_np).query(pts_np[idx], k=k, workers=-1)
    nbr = pts_np[nb]                                    # (S,k,3)
    nbr = nbr - nbr.mean(1, keepdims=True)
    cov = np.einsum("ski,skj->sij", nbr, nbr) / k
    ev = np.linalg.eigvalsh(cov)                        # ascending
    planar = ev[:, 0] / np.clip(ev.sum(1), 1e-20, None)
    return float((planar < thresh).mean() * 100), float(np.median(planar))


def score(recon, gt_dir, activation):
    d = torch.load(recon, map_location="cpu")
    pts = d["points"].float()
    raw_density, raw_radii = d["density"].float(), d["radii"].float()
    radii = F.softplus(raw_radii, beta=100)
    # density activation must match how the arm was TRAINED, or opacity is meaningless
    sigma = torch.exp(raw_density) if activation == "exp" else F.softplus(raw_density, beta=100)

    gt_xyz = torch.from_numpy(np.load(f"{gt_dir}/coord.npy")).float()
    gt_nrm = torch.from_numpy(np.load(f"{gt_dir}/normal.npy")).float()
    gt_nrm = gt_nrm / gt_nrm.norm(dim=-1, keepdim=True).clamp_min(1e-20)

    # normals: nearest primitive per GT point
    _, idx = cKDTree(pts.numpy()).query(gt_xyz.numpy(), k=1, workers=-1)
    n = frame_from_quaternions(d["quaternions"].float())[torch.from_numpy(idx).long()]
    cos = (n * gt_nrm).sum(-1).abs().clamp(max=1.0)
    err = torch.rad2deg(torch.arccos(cos)).numpy()

    # drift: primitive centre -> nearest GT point (metres, and in units of its own radius)
    dist, _ = cKDTree(gt_xyz.numpy()).query(pts.numpy(), k=1, workers=-1)
    drift_r = dist / np.clip(radii.numpy(), 1e-12, None)

    alpha = (1.0 - torch.exp(-sigma * 2.0 * radii)).numpy()
    planar_pct, planar_med = planarity_fraction(pts.numpy())

    return dict(
        num_primitives=int(pts.shape[0]),
        normal_median_deg=float(np.median(err)),
        normal_mean_deg=float(err.mean()),
        normal_within10_pct=float((err < 10).mean() * 100),
        drift_median_m=float(np.median(dist)),
        drift_median_radii=float(np.median(drift_r)),
        alpha_median=float(np.median(alpha)),
        alpha_mid_band_pct=float((((alpha > 0.1) & (alpha < 0.9)).mean()) * 100),
        alpha_near_opaque_pct=float((alpha > 0.9).mean() * 100),
        alpha_near_transparent_pct=float((alpha < 0.1).mean() * 100),
        planar_below0p05_pct=planar_pct,
        planarity_median=planar_med,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=os.path.expanduser("~/powerfoam/output"))
    ap.add_argument("--gt-root", default="/home/rajehyl/scannet_gt")
    ap.add_argument("--pattern", default="surf2x2_")
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    rows = {}
    for name in sorted(os.listdir(a.output_root)):
        if not name.startswith(a.pattern):
            continue
        ck = os.path.join(a.output_root, name, "model.pt")
        if not os.path.exists(ck):
            print(f"skip (no model.pt yet): {name}"); continue
        m = re.match(rf"{a.pattern}(scene\d+_\d+)_plr(\S+?)_(softplus|exp)$", name)
        if not m:
            print(f"skip (unparsed name): {name}"); continue
        scene, plr, act = m.groups()
        gt = f"{a.gt_root}/train/{scene}"
        if not os.path.isdir(gt):
            gt = f"{a.gt_root}/val/{scene}"
        r = score(ck, gt, act)
        r.update(scene=scene, points_lr_init=plr, density_activation=act,
                 **read_metrics_txt(os.path.join(a.output_root, name, "metrics.txt")))
        rows[name] = r
        print(f"{name:<44} normals {r['normal_median_deg']:6.2f}  drift {r['drift_median_radii']:6.2f}r  "
              f"alpha-mid {r['alpha_mid_band_pct']:5.1f}%  psnr {r.get('psnr', float('nan')):.2f}")

    with open(a.json_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {a.json_out}  ({len(rows)} arms)")
    print("NOTE: drift is a PROXY. CD-L1 from the extracted surface is required for any geometry claim.")


if __name__ == "__main__":
    main()
