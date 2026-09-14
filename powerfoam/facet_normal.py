"""Facet-normal coupling: the orientation term missing from PowerFoam's training objective.

WHY THIS EXISTS. The only normal-related term in the renderer's loss is `normal_err`
(rasterize.py, built into the forward kernel as `sum_i alpha_i * T_i * max(0, dot(n_i, d))^2`).
That is a pure BACKFACE penalty -- it constrains the free per-primitive quaternion normal to the
camera-facing hemisphere and says nothing else. Measured consequence (MyResearchVault/
Dipole-Denoising-Probe.md): these normals sit at ~50 deg mean angular error against a 60 deg
random baseline, and the error is flat across alpha/distance bins -- the signature of a missing
term, not an undersupervised one. Post-hoc denoising (texel-height slope, neighbourhood
aggregation) was tried and both failed or under-delivered, confirming the deviation from truth is
systematic, not noise -- it has to be fixed at training time.

THE SIGNAL THAT ALREADY EXISTS AND IS UNUSED. For a power diagram the facet between adjacent
cells i, j is exactly perpendicular to (p_j - p_i) -- radii shift the plane's OFFSET, never its
ORIENTATION. So wherever a cell sits next to emptier space, the tessellation already specifies
the true interface orientation, with no learned parameter involved. This module builds a target
normal per primitive from that fact and pulls the free quaternion normal toward it.

TARGET CONSTRUCTION. For primitive i with neighbours j (from the checkpoint's own adjacency):
  u_ij = (p_j - p_i) / ||p_j - p_i||                          unit facet direction
  a_k  = 1 - exp(-sigma_k * 2 * r_k)                          per-cell opacity proxy
  w_ij = relu(a_i - a_j)                                      occupancy contrast, OUTWARD only
  t_i  = sum_j w_ij * u_ij                                     the target (unnormalized)
Primitives with ||t_i|| below `min_contrast` have no resolvable interface (uniform neighbourhood)
and are EXCLUDED, not zero-padded -- a zero-vector target normalized to garbage would inject pure
noise into the loss.

LOSS. `mean(1 - |dot(n_i, normalize(t_i))|)` over included primitives. The absolute value makes
this an AXIS loss, not a signed loss: it says "n_i's line must match the facet", nothing about
which way it points. Sign is left to the existing backface term so the two do not fight.

GRADIENT ROUTING. By default (`grad_to_geometry=False`) the target t_i is built under
`torch.no_grad()`, so gradients flow only into the quaternions -- the axis is treated as a fixed
target for this step, which is the stable, conservative choice. Setting `grad_to_geometry=True`
also lets the loss move `points`/`radii`/`density`, i.e. the tessellation itself can reshape to
make its own facets more orientation-consistent, not just have the normal chase them.

STATUS (2026-09-14): implemented and unit-tested (exact analytic recovery on synthetic
configurations, including a two-cell dipole and a 3x3 slab interface), but an isolated test on
REAL trained checkpoints found this target does NOT recover true orientation -- the occupancy
pattern in a trained density field does not reliably track the true surface, even on frozen/
GT-position geometry (median error stays at or above the 60 deg random baseline). See
MyResearchVault/Stage0-Surface-Prior-Art.md for the full negative result. Kept because the
construction and tests are real and correct -- the idea failed on real data, not the
implementation -- and because the mechanism may still be reusable if a better occupancy signal
is found later.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _csr_src_dst(adjacency: torch.Tensor, adjacency_offsets: torch.Tensor):
    """Expand CSR adjacency (adjacency_offsets has N+1 entries) into (src, dst) edge index pairs."""
    n = adjacency_offsets.numel() - 1
    counts = (adjacency_offsets[1:] - adjacency_offsets[:-1]).long()
    src = torch.repeat_interleave(
        torch.arange(n, dtype=torch.long, device=adjacency.device), counts
    )
    dst = adjacency.long()
    return src, dst


def facet_normal_target(
    points: torch.Tensor,
    density: torch.Tensor,
    radii: torch.Tensor,
    adjacency: torch.Tensor,
    adjacency_offsets: torch.Tensor,
    min_contrast: float = 1e-3,
    grad_to_geometry: bool = False,
):
    """Build the per-primitive target normal from occupancy-contrast-weighted facet directions.

    Returns (target, included_mask): `target` is (N, 3), meaningful only where
    `included_mask` is True (excluded rows are left as zero and MUST be masked out by the
    caller -- they are not a valid unit direction).
    """
    ctx = torch.enable_grad() if grad_to_geometry else torch.no_grad()
    with ctx:
        n = points.shape[0]
        src, dst = _csr_src_dst(adjacency, adjacency_offsets)

        alpha = 1.0 - torch.exp(-density * 2.0 * radii)  # (N,) opacity proxy, one per primitive

        diff = points[dst] - points[src]  # (E, 3)
        dist = diff.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u = diff / dist  # (E, 3) unit facet direction, src -> dst

        w = F.relu(alpha[src] - alpha[dst])  # (E,) outward occupancy contrast, >=0

        target = torch.zeros(n, 3, dtype=points.dtype, device=points.device)
        target.index_add_(0, src, w.unsqueeze(-1) * u)

        included = target.norm(dim=-1) >= min_contrast
    return target, included


def facet_normal_loss(
    normals: torch.Tensor,
    points: torch.Tensor,
    density: torch.Tensor,
    radii: torch.Tensor,
    adjacency: torch.Tensor,
    adjacency_offsets: torch.Tensor,
    min_contrast: float = 1e-3,
    grad_to_geometry: bool = False,
):
    """mean(1 - |dot(n_i, normalize(t_i))|) over primitives with a resolvable target.

    Returns a scalar tensor. If NO primitive has a resolvable target (degenerate/empty scene),
    returns a zero tensor that still participates correctly in autograd (connected to `normals`
    with zero coefficient) rather than a disconnected python float.
    """
    target, included = facet_normal_target(
        points, density, radii, adjacency, adjacency_offsets,
        min_contrast=min_contrast, grad_to_geometry=grad_to_geometry,
    )
    if not bool(included.any()):
        return (normals.sum() * 0.0)

    t_hat = F.normalize(target[included], dim=-1)
    n_i = normals[included]
    cos = (n_i * t_hat).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - cos).mean()
