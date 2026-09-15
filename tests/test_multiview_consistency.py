"""Real correctness tests for powerfoam/multiview_consistency.py -- the multi-view planar-patch
photometric consistency loss ("idea C" of the surface-aware-training project plan).

Everything here runs on CPU only (no GPU/CUDA, no Warp kernel is ever invoked by this module).

Test #1 (round-trip projection) is the single most load-bearing test in this file: every other
test depends on `project_point` actually being the exact numerical inverse of the ray-casting
formula `unproject_pixel` implements, since the whole loss is built by unprojecting a reference
pixel, moving along its tangent plane, and projecting back into a second camera.

Test #2/#3 build a synthetic scene entirely by hand (a real, tilted plane with a smooth analytic
texture, two cameras placed and oriented so the correspondence check can be made essentially
EXACT rather than merely approximate -- see `_make_scene`'s docstring for why): with the TRUE
depth/normal the loss is near zero; with wrong depth/normal it is not.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from powerfoam.multiview_consistency import (
    project_point,
    unproject_pixel,
    sample_bilinear,
    planar_patch_warp,
    ncc,
    multiview_planar_ncc_loss,
)


class _Camera:
    """Minimal duck-typed stand-in for `powerfoam.camera.TorchCamera` -- only the attributes
    `multiview_consistency.py` actually reads (`.eye`, `.right`, `.up`, `.width`, `.height`).
    Deliberately does NOT import `powerfoam.camera` (which unconditionally imports `open3d` at
    module scope) so this test file has no dependency beyond torch."""

    def __init__(self, eye, right, up, width, height):
        self.eye = eye
        self.right = right
        self.up = up
        self.width = width
        self.height = height


def _normalize(v):
    return v / v.norm()


def _make_camera_basis(forward_dir, right_scale, up_scale, dtype):
    """Build a (right, up) pair such that `forward = normalize(cross(up, right))` equals
    `forward_dir` exactly (verified numerically while writing this test: dot(recovered_forward,
    forward_dir) == 1.0 to double precision)."""
    forward_dir = _normalize(forward_dir)
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    if abs(torch.dot(world_up, forward_dir)) > 0.95:
        world_up = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    right_hat = _normalize(torch.cross(forward_dir, world_up, dim=-1))
    up_hat = _normalize(torch.cross(right_hat, forward_dir, dim=-1))
    return right_hat * right_scale, up_hat * up_scale


def _tangent_basis(normal, helper):
    helper = helper - torch.dot(helper, normal) * normal
    e1 = _normalize(helper)
    e2 = torch.cross(normal, e1, dim=-1)
    return e1, e2


def _make_scene(dtype=torch.float64, H=64, W=80):
    """A tilted plane with a smooth analytic texture, seen by two cameras.

    GEOMETRY. The plane has an arbitrary, non-axis-aligned unit normal `N` and passes through
    `Q0`. `ref_camera` is placed FRONTO-PARALLEL to the plane (`forward = N`, i.e. the plane's
    outward-facing normal, which is oriented toward the camera, is `-forward`) -- this is not just
    a convenience: for a fronto-parallel view, `right`/`up` already lie exactly IN the tangent
    plane (they are, by the camera model, orthogonal to `forward`, and the plane here is exactly
    orthogonal to `forward` too), so `planar_patch_warp`'s internal Gram-Schmidt basis becomes a
    no-op (`e1 = normalize(right)` exactly) and its patch-grid world positions become an EXACT
    (not first-order-approximate) linear reparameterization of the reference camera's own
    neighbouring pixels. That is what lets test #2 check `planar_patch_warp` to tight tolerance
    with no curvature error to account for. `src_camera` is a second camera offset sideways and
    re-oriented to still look at roughly the same patch of plane, NOT fronto-parallel to it --
    only the reference side needs the fronto-parallel property for this test's exactness.

    TEXTURE. The "ground truth" is a smooth function `sin(3u)*cos(2.5v)` of PLANE-LOCAL
    coordinates (u, v), defined via a basis (`U`, `V`) that is fixed once, independent of
    anything either camera or `planar_patch_warp` itself computes. `ref_image`/`src_image` are
    built by, for every pixel of each camera, intersecting that pixel's own ray with the plane
    (bypassing any renderer entirely) and evaluating the texture at the intersection's (u, v).
    A smooth trig function (not a checkerboard) is used deliberately: a checkerboard's
    discontinuities would fail bilinear-sampling gradient checks and inflate interpolation error
    past any tight tolerance purely from aliasing, independent of whether the geometry is right.

    Returns a dict with camera objects, `ref_image`/`src_image` (H, W, 3), `ref_depth`/`ref_normal`
    (the TRUE values, i.e. exactly what a perfect renderer would have produced for this scene),
    and the plane's own `(Q0, N, U, V)` for tests that want to perturb depth/normal directly.
    """
    N = _normalize(torch.tensor([0.2, 0.4, 1.0], dtype=dtype))
    Q0 = torch.tensor([0.3, -0.1, 2.0], dtype=dtype)

    world_helper = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    if abs(torch.dot(world_helper, N)) > 0.9:
        world_helper = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    U, V = _tangent_basis(N, world_helper)

    def texture(u, v):
        return torch.sin(3.0 * u) * torch.cos(2.5 * v)

    def plane_uv(points):
        rel = points - Q0
        return (rel * U).sum(-1), (rel * V).sum(-1)

    def ray_plane_intersect(cam, rows, cols):
        eye, up, right = cam.eye, cam.up, cam.right
        forward = torch.cross(up, right, dim=-1)
        forward = forward / forward.norm()
        x = 2.0 * cols / (float(cam.width) - 1.0) - 1.0
        y = 1.0 - 2.0 * rows / (float(cam.height) - 1.0)
        ray_unnorm = x[..., None] * right + y[..., None] * up + forward[None, :]
        denom = (ray_unnorm * N).sum(-1)
        t = torch.dot(Q0 - eye, N) / denom
        points = eye + t[..., None] * ray_unnorm
        depth = t * torch.norm(ray_unnorm, dim=-1)
        return points, depth

    ref_eye = Q0 - 2.5 * N
    ref_right, ref_up = _make_camera_basis(N, 0.35, 0.28, dtype)
    ref_camera = _Camera(ref_eye, ref_right, ref_up, W, H)

    src_eye = Q0 - 2.0 * N + 0.6 * U + 0.3 * V
    src_right, src_up = _make_camera_basis(_normalize(Q0 - src_eye), 0.4, 0.3, dtype)
    src_camera = _Camera(src_eye, src_right, src_up, W, H)

    rows_grid, cols_grid = torch.meshgrid(
        torch.arange(H, dtype=dtype), torch.arange(W, dtype=dtype), indexing="ij"
    )

    ref_points, ref_depth = ray_plane_intersect(ref_camera, rows_grid, cols_grid)
    u_ref, v_ref = plane_uv(ref_points)
    ref_image = texture(u_ref, v_ref).unsqueeze(-1).repeat(1, 1, 3)
    ref_normal = (-N).expand(H, W, 3).clone()  # oriented toward the (fronto-parallel) camera

    src_points, _src_depth = ray_plane_intersect(src_camera, rows_grid, cols_grid)
    u_src, v_src = plane_uv(src_points)
    src_image = texture(u_src, v_src).unsqueeze(-1).repeat(1, 1, 3)

    assert bool((ref_depth > 0).all()) and bool((_src_depth > 0).all())

    return dict(
        ref_camera=ref_camera,
        src_camera=src_camera,
        ref_image=ref_image,
        src_image=src_image,
        ref_depth=ref_depth,
        ref_normal=ref_normal,
        Q0=Q0,
        N=N,
        U=U,
        V=V,
    )


def _tilt_normal(normal, angle_deg, axis):
    """Rodrigues rotation of `normal` (..., 3) by `angle_deg` about unit-ish `axis` (3,)."""
    angle = torch.tensor(angle_deg * math.pi / 180.0, dtype=normal.dtype)
    axis = (axis / axis.norm()).expand_as(normal)
    return (
        normal * torch.cos(angle)
        + torch.cross(axis, normal, dim=-1) * torch.sin(angle)
        + axis * (axis * normal).sum(-1, keepdim=True) * (1 - torch.cos(angle))
    )


# ---------------------------------------------------------------------------------------------
# 1. Round-trip projection is the exact inverse of ray-casting -- the load-bearing test.
# ---------------------------------------------------------------------------------------------


def test_project_point_is_exact_inverse_of_unproject_pixel():
    torch.manual_seed(0)
    dtype = torch.float32
    eye = torch.tensor([0.1, -0.2, 0.3], dtype=dtype)
    right = torch.tensor([0.6, 0.05, -0.05], dtype=dtype)
    up = torch.tensor([0.02, 0.5, 0.03], dtype=dtype)
    width, height = 64, 48
    cam = _Camera(eye, right, up, width, height)

    n = 2000
    rows = torch.randint(0, height, (n,)).to(dtype)
    cols = torch.randint(0, width, (n,)).to(dtype)
    depth = torch.rand(n, dtype=dtype) * 4.0 + 0.5  # positive depths only

    points = unproject_pixel(cam, rows, cols, depth)
    x, y, depth_rec, valid = project_point(cam, points)

    assert bool(valid.all()), "every unprojected-then-reprojected point must be in front and in-bounds"

    col_rec = (x + 1.0) * (float(width) - 1.0) / 2.0
    row_rec = (1.0 - y) * (float(height) - 1.0) / 2.0

    assert torch.allclose(row_rec, rows, atol=1e-4), (row_rec - rows).abs().max().item()
    assert torch.allclose(col_rec, cols, atol=1e-4), (col_rec - cols).abs().max().item()
    assert torch.allclose(depth_rec, depth, atol=1e-4), (depth_rec - depth).abs().max().item()


def test_project_point_marks_behind_camera_as_invalid():
    dtype = torch.float32
    eye = torch.zeros(3, dtype=dtype)
    right = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    cam = _Camera(eye, right, up, 32, 32)

    # forward = normalize(cross(up, right)) = normalize(cross([0,1,0],[1,0,0])) = [0,0,-1] here,
    # so a point at +z (behind the eye relative to that forward direction) must be invalid.
    behind = torch.tensor([[0.0, 0.0, 5.0]], dtype=dtype)
    _x, _y, _d, valid = project_point(cam, behind)
    assert not bool(valid.any())


def test_project_point_marks_out_of_fov_as_invalid():
    dtype = torch.float32
    eye = torch.zeros(3, dtype=dtype)
    right = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    cam = _Camera(eye, right, up, 32, 32)

    # z = -1 is in FRONT of this camera (forward = [0,0,-1]); a huge x offset pushes it outside
    # the [-1, 1] normalized FOV without also being behind the camera.
    far_off_axis = torch.tensor([[50.0, 0.0, -1.0]], dtype=dtype)
    _x, _y, _d, valid = project_point(cam, far_off_axis)
    assert not bool(valid.any())


# ---------------------------------------------------------------------------------------------
# 2. Fronto-parallel two-camera analytic-texture test.
# ---------------------------------------------------------------------------------------------


def test_planar_patch_warp_reproduces_source_patch_at_true_geometry():
    scene = _make_scene()
    pr = torch.tensor([20, 32, 40, 45])
    pc = torch.tensor([25, 40, 55, 30])

    ref_patch, src_patch, valid = planar_patch_warp(
        scene["ref_camera"], scene["src_camera"], scene["ref_depth"], scene["ref_normal"],
        scene["ref_image"], scene["src_image"], pr, pc, patch_radius=2,
    )
    assert bool(valid.all()), "chosen centres should all be comfortably interior/valid"
    diff = (ref_patch[valid] - src_patch[valid]).abs()
    assert diff.max().item() < 1e-3, diff.max().item()


def test_multiview_loss_near_zero_at_true_geometry():
    scene = _make_scene()
    loss = multiview_planar_ncc_loss(
        scene["ref_camera"], scene["ref_image"], scene["ref_depth"], scene["ref_normal"],
        scene["src_camera"], scene["src_image"], patch_radius=2, num_patches=300, min_depth=1e-3,
        generator=torch.Generator().manual_seed(0),
    )
    assert loss.item() < 1e-3, loss.item()


# ---------------------------------------------------------------------------------------------
# 3. Loss increases under wrong geometry.
# ---------------------------------------------------------------------------------------------


def test_loss_increases_with_depth_error():
    scene = _make_scene()
    gen = lambda: torch.Generator().manual_seed(7)
    loss_true = multiview_planar_ncc_loss(
        scene["ref_camera"], scene["ref_image"], scene["ref_depth"], scene["ref_normal"],
        scene["src_camera"], scene["src_image"], patch_radius=2, num_patches=300, min_depth=1e-3,
        generator=gen(),
    )
    wrong_depth = scene["ref_depth"] * 1.05
    loss_wrong = multiview_planar_ncc_loss(
        scene["ref_camera"], scene["ref_image"], wrong_depth, scene["ref_normal"],
        scene["src_camera"], scene["src_image"], patch_radius=2, num_patches=300, min_depth=1e-3,
        generator=gen(),
    )
    assert loss_true.item() < 1e-3
    assert loss_wrong.item() > 10.0 * max(loss_true.item(), 1e-6)
    assert loss_wrong.item() > 1e-2, loss_wrong.item()


def test_loss_increases_with_normal_tilt():
    scene = _make_scene()
    gen = lambda: torch.Generator().manual_seed(7)
    loss_true = multiview_planar_ncc_loss(
        scene["ref_camera"], scene["ref_image"], scene["ref_depth"], scene["ref_normal"],
        scene["src_camera"], scene["src_image"], patch_radius=2, num_patches=300, min_depth=1e-3,
        generator=gen(),
    )
    tilted_normal = _tilt_normal(scene["ref_normal"], 15.0, scene["U"])
    loss_wrong = multiview_planar_ncc_loss(
        scene["ref_camera"], scene["ref_image"], scene["ref_depth"], tilted_normal,
        scene["src_camera"], scene["src_image"], patch_radius=2, num_patches=300, min_depth=1e-3,
        generator=gen(),
    )
    assert loss_true.item() < 1e-3
    assert loss_wrong.item() > 10.0 * max(loss_true.item(), 1e-6)
    assert loss_wrong.item() > 1e-3, loss_wrong.item()


# ---------------------------------------------------------------------------------------------
# 4. NCC invariance properties.
# ---------------------------------------------------------------------------------------------


def test_ncc_identical_patches_gives_zero_loss():
    torch.manual_seed(0)
    a = torch.randn(6, 5, 5, 3)
    loss = 1.0 - ncc(a, a.clone())
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6), loss


def test_ncc_invariant_to_independent_affine_map():
    torch.manual_seed(0)
    a = torch.randn(6, 5, 5, 3)
    b = torch.randn(6, 5, 5, 3)
    scale = torch.tensor([2.0, 0.5, 3.0, 1.2, 7.0, 0.1]).view(6, 1, 1, 1)
    shift = torch.tensor([1.0, -2.0, 0.5, 0.3, -4.0, 2.2]).view(6, 1, 1, 1)
    b_affine = b * scale + shift  # a>0 scale + arbitrary shift, per-patch independent
    assert torch.allclose(ncc(a, b), ncc(a, b_affine), atol=1e-5)


def test_ncc_own_negation_saturates_near_max_loss():
    torch.manual_seed(0)
    a = torch.randn(4, 5, 5, 3)
    a_centered = a - a.mean(dim=(1, 2, 3), keepdim=True)
    negated = a.mean(dim=(1, 2, 3), keepdim=True) - a_centered  # mean-preserving sign flip
    loss = 1.0 - ncc(a, negated)
    assert torch.allclose(loss, 2.0 * torch.ones_like(loss), atol=1e-5), loss


# ---------------------------------------------------------------------------------------------
# 5. Gradient check.
# ---------------------------------------------------------------------------------------------


def test_gradient_matches_finite_difference():
    scene = _make_scene(H=20, W=24)
    ref_depth = (scene["ref_depth"] * 1.01).clone().requires_grad_(True)
    ref_normal = scene["ref_normal"].clone().requires_grad_(True)

    def loss_fn(depth, normal):
        return multiview_planar_ncc_loss(
            scene["ref_camera"], scene["ref_image"], depth, normal,
            scene["src_camera"], scene["src_image"], patch_radius=1, num_patches=6,
            min_depth=1e-3, generator=torch.Generator().manual_seed(3),
        )

    loss = loss_fn(ref_depth, ref_normal)
    loss.backward()
    g_depth = ref_depth.grad.clone()
    g_normal = ref_normal.grad.clone()

    eps = 1e-6

    depth_idx = (g_depth.abs() > 0).nonzero(as_tuple=False)
    assert depth_idx.shape[0] > 0, "expected at least one sampled patch to touch ref_depth"
    for k in range(min(6, depth_idx.shape[0])):
        i, j = depth_idx[k].tolist()
        dp = ref_depth.detach().clone(); dp[i, j] += eps
        dm = ref_depth.detach().clone(); dm[i, j] -= eps
        with torch.no_grad():
            lp = loss_fn(dp, ref_normal.detach())
            lm = loss_fn(dm, ref_normal.detach())
        numeric = ((lp - lm) / (2 * eps)).item()
        analytic = g_depth[i, j].item()
        assert abs(numeric - analytic) < 1e-5, (i, j, analytic, numeric)

    normal_idx = (g_normal.abs() > 0).nonzero(as_tuple=False)
    assert normal_idx.shape[0] > 0, "expected at least one sampled patch to touch ref_normal"
    for k in range(min(6, normal_idx.shape[0])):
        i, j, c = normal_idx[k].tolist()
        npp = ref_normal.detach().clone(); npp[i, j, c] += eps
        npm = ref_normal.detach().clone(); npm[i, j, c] -= eps
        with torch.no_grad():
            lp = loss_fn(ref_depth.detach(), npp)
            lm = loss_fn(ref_depth.detach(), npm)
        numeric = ((lp - lm) / (2 * eps)).item()
        analytic = g_normal[i, j, c].item()
        assert abs(numeric - analytic) < 1e-5, (i, j, c, analytic, numeric)


# ---------------------------------------------------------------------------------------------
# 6. Degenerate case returns a disconnected zero.
# ---------------------------------------------------------------------------------------------


def test_degenerate_all_invalid_depth_returns_disconnected_zero():
    """Mirrors `normal_consistency.py`'s own degenerate-case test: when nothing qualifies (here,
    every pixel has non-positive depth), the loss must be a plain zero NOT connected to
    `ref_depth`'s/`ref_normal`'s autograd graph -- never a `.sum() * 0.0`-style "connected" zero,
    which this project has already found crashes against the renderer's real Warp-backed output
    (see this module's and `normal_consistency.py`'s docstrings)."""
    H, W = 16, 16
    ref_depth = torch.zeros(H, W, requires_grad=True)  # depth <= 0 everywhere -> nothing valid
    ref_normal = torch.zeros(H, W, 3, requires_grad=True)
    ref_normal.data[..., 2] = 1.0
    ref_image = torch.zeros(H, W, 3)
    src_image = torch.zeros(H, W, 3)

    cam = _Camera(
        torch.tensor([0.0, 0.0, 5.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        W, H,
    )

    loss = multiview_planar_ncc_loss(
        cam, ref_image, ref_depth, ref_normal, cam, src_image,
        patch_radius=1, num_patches=10, min_depth=1e-4,
    )
    assert loss.item() == 0.0
    assert loss.grad_fn is None
    assert not loss.requires_grad

    # still safe to fold into a larger loss sum that DOES require grad
    total = ref_depth.sum() * 0.0 + loss
    total.backward()  # must not raise
