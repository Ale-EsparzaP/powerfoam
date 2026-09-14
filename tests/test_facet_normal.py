"""Real correctness tests for powerfoam/facet_normal.py -- the orientation term that couples the
free per-primitive quaternion normal to the tessellation's own facet geometry. Each test builds a
tiny synthetic configuration where the target normal is known analytically, not just plausible."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from powerfoam.facet_normal import facet_normal_loss, facet_normal_target


def csr(pairs, n):
    """Build (adjacency, adjacency_offsets) CSR from a plain list of (src, dst) edges."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for s, d in pairs:
        buckets[s].append(d)
    offsets = [0]
    flat = []
    for i in range(n):
        flat.extend(buckets[i])
        offsets.append(len(flat))
    return torch.tensor(flat, dtype=torch.int32), torch.tensor(offsets, dtype=torch.int32)


def opacity_to_density(alpha, radius):
    """Invert alpha = 1 - exp(-sigma * 2r) to get the raw density (already-activated sigma)
    facet_normal_target expects, so a test can specify opacity directly and know it round-trips."""
    return -math.log(max(1e-12, 1.0 - alpha)) / (2.0 * radius)


def test_two_cell_dipole_target_is_exact():
    """One opaque cell at the origin, one transparent neighbour at +x. The target at cell 0
    must point exactly toward the empty neighbour: +x."""
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = torch.tensor([0.1, 0.1])
    density = torch.tensor([opacity_to_density(0.95, 0.1), opacity_to_density(0.05, 0.1)])
    adjacency, offsets = csr([(0, 1), (1, 0)], n=2)

    target, included = facet_normal_target(points, density, radii, adjacency, offsets)
    assert bool(included[0]) and not bool(included[1])  # cell 1 sees LESS opaque neighbour -> relu(0)=0

    t_hat = F.normalize(target[0:1], dim=-1)[0]
    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(t_hat, expected, atol=1e-6), t_hat

    # A primitive whose normal already equals the target must score ~zero loss.
    normals = torch.zeros(2, 3)
    normals[0] = expected
    loss = facet_normal_loss(normals, points, density, radii, adjacency, offsets)
    assert loss.item() < 1e-6, loss.item()


def test_slab_interface_normal_matches_slab_normal():
    """A 3x3 grid of opaque cells in the z=0 plane, each with one transparent neighbour directly
    above it (z=+1). Every occupied cell's target must be +z, to numerical tolerance -- this is
    the test that proves the construction recovers a real, extended surface, not just a toy pair."""
    occ_xy = [(x, y) for x in range(3) for y in range(3)]
    points_list = [[x, y, 0.0] for x, y in occ_xy] + [[x, y, 1.0] for x, y in occ_xy]
    points = torch.tensor(points_list, dtype=torch.float32)
    n_occ = len(occ_xy)
    n = points.shape[0]

    radii = torch.full((n,), 0.3)
    density = torch.empty(n)
    density[:n_occ] = opacity_to_density(0.97, 0.3)
    density[n_occ:] = opacity_to_density(0.02, 0.3)

    # Each occupied cell i is adjacent only to its empty counterpart i+n_occ (directly above).
    pairs = []
    for i in range(n_occ):
        pairs.append((i, i + n_occ))
        pairs.append((i + n_occ, i))
    adjacency, offsets = csr(pairs, n=n)

    target, included = facet_normal_target(points, density, radii, adjacency, offsets)
    assert included[:n_occ].all()
    assert not included[n_occ:].any()  # empty cells see a LESS opaque neighbour -> excluded

    t_hat = F.normalize(target[:n_occ], dim=-1)
    expected = torch.tensor([0.0, 0.0, 1.0]).expand_as(t_hat)
    assert torch.allclose(t_hat, expected, atol=1e-5), t_hat


def test_sign_invariance():
    """Flipping a primitive's normal to -n must leave the loss unchanged (axis-only loss)."""
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = torch.tensor([0.1, 0.1])
    density = torch.tensor([opacity_to_density(0.9, 0.1), opacity_to_density(0.1, 0.1)])
    adjacency, offsets = csr([(0, 1), (1, 0)], n=2)

    normals_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals_neg = normals_pos.clone()
    normals_neg[0] = -normals_neg[0]

    loss_pos = facet_normal_loss(normals_pos, points, density, radii, adjacency, offsets)
    loss_neg = facet_normal_loss(normals_neg, points, density, radii, adjacency, offsets)
    assert torch.allclose(loss_pos, loss_neg, atol=1e-7), (loss_pos.item(), loss_neg.item())


def test_uniform_neighbourhood_is_excluded():
    """A primitive whose neighbours all share its own opacity has no resolvable interface and
    must be EXCLUDED (not silently included with a zero/garbage target)."""
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radii = torch.full((4,), 0.1)
    density = torch.full((4,), opacity_to_density(0.5, 0.1))  # ALL cells equally opaque
    adjacency, offsets = csr([(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)], n=4)

    target, included = facet_normal_target(points, density, radii, adjacency, offsets)
    assert not bool(included[0]), "uniform-opacity neighbourhood must be excluded, not included"
    assert int(included.sum()) == 0


def test_off_by_default_no_op():
    """With facet_normal_weight = 0.0 the loss must not be computed at all in train.py's own
    code path (the config default). This test checks the CONFIG default directly, since the
    actual skip lives in train.py's `if args.facet_normal_weight > 0.0:` guard."""
    from configs import Params
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(Params)}
    assert fields["facet_normal_weight"].default == 0.0
    assert fields["facet_normal_grad_to_geometry"].default is False


def test_gradient_matches_finite_difference():
    """Finite-difference the loss w.r.t. a normal vector's free parameters (using an
    unnormalized direction, matching how PowerfoamScene.get_normals() always renormalizes)
    against autograd, on a small configuration."""
    torch.manual_seed(0)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]], dtype=torch.float64)
    radii = torch.tensor([0.1, 0.1], dtype=torch.float64)
    density = torch.tensor(
        [opacity_to_density(0.9, 0.1), opacity_to_density(0.1, 0.1)], dtype=torch.float64
    )
    adjacency, offsets = csr([(0, 1), (1, 0)], n=2)

    raw = torch.tensor([0.3, 0.9, -0.4], dtype=torch.float64, requires_grad=True)

    def loss_fn(raw_vec):
        n0 = raw_vec / raw_vec.norm()
        normals = torch.stack([n0, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)])
        return facet_normal_loss(normals, points, density, radii, adjacency, offsets)

    loss = loss_fn(raw)
    loss.backward()
    analytic = raw.grad.clone()

    eps = 1e-6
    numeric = torch.zeros(3, dtype=torch.float64)
    for k in range(3):
        d = torch.zeros(3, dtype=torch.float64)
        d[k] = eps
        with torch.no_grad():
            lp = loss_fn(raw + d)
            lm = loss_fn(raw - d)
        numeric[k] = (lp - lm) / (2 * eps)

    assert torch.allclose(analytic, numeric, atol=1e-5), (analytic, numeric)
