"""Multi-view planar-patch photometric consistency (PGSR-style) -- "idea C" of the
surface-aware-training project plan.

WHY THIS EXISTS. `normal_consistency.py` couples the renderer's own composited normal to an
external orientation estimate; `facet_normal.py` couples the free per-primitive normal to the
tessellation's own facet geometry (found NOT to help, see that module's docstring). Neither one
checks the depth/normal pair against a genuinely INDEPENDENT signal: a second camera's own pixels.
This module supplies that signal. The idea (PGSR/photometric-planar-consistency style): if the
rendered depth+normal at a reference pixel truly describe the local surface, then a small patch
around that pixel, warped through the LOCAL TANGENT PLANE they define into a nearby source view,
must photometrically match what that source view actually observed there. If depth or normal are
wrong, the warp lands on the wrong source pixels and the patches stop matching -- an unbiased,
purely geometric, multi-view signal that does not depend on any single view's own rendering being
self-consistent (unlike `normal_consistency.py`, which only ever checks a view against itself).

POWERFOAM'S CAMERA MODEL IS NOT A STANDARD PINHOLE (K, R, t). Per `powerfoam/geometry.py`'s
`normals_from_depth` (the reference this module was built against): a camera exposes `.eye`
(3-vector), `.up`/`.right` (NON-unit 3-vectors -- their norm encodes tan(fov/2) for that axis),
and integer `.height`/`.width`. `forward = normalize(cross(up, right))`. Pixel (row i, col j) maps
to normalized coords `x = 2*j/(W-1) - 1`, `y = 1 - 2*i/(H-1)` (note the y-flip), non-unit ray
`x*right + y*up + forward`, normalized to a unit ray; a 3D point at scalar depth `d` along that
ray is `eye + d*unit_ray`. `unproject_pixel` below is exactly this formula, factored out (without
touching `normals_from_depth` itself, which is left untouched per this task's constraints).

THE INVERSE (`project_point`) DOES NOT EXIST ELSEWHERE IN THIS CODEBASE and is derived here, not
assumed from a pinhole K-matrix (there isn't one). For target camera and world point P: let
`v = P - eye`. Solve `M @ [a, b, c]^T = v` where `M = [right | up | forward]` (columns). Then
`x = a/c`, `y = b/c`, valid only if `c > 0` (in front of the camera) and `(x, y)` within
`[-1, 1]`. Since M depends only on the camera (not on P), it is inverted ONCE per call and
applied to a whole batch of points via a single batched matvec -- there is no need to solve a
fresh 3x3 system per point. The along-ray depth of a valid point is simply `||v||`, because
`P = eye + depth * unit_ray` by construction whenever P truly lies on the ray through (x, y), so
`||P - eye|| = depth` exactly (unit_ray has unit norm) -- no dependence on x, y, or c is needed
for that part of the computation.

`project_point` is verified to be the EXACT numerical inverse of the ray-casting formula by
`tests/test_multiview_consistency.py`'s round-trip test -- the single most load-bearing test in
that file, since every other test here (patch warping, the loss, its gradient) depends on
`project_point` and `unproject_pixel` actually being inverses of one another.

TANGENT-PLANE PATCH CONSTRUCTION (`planar_patch_warp`). At a reference pixel with depth `d` and
normal `n`, the 3D point `q = unproject_pixel(ref_camera, row, col, d)` and `n` define a plane.
Two orthonormal in-plane basis vectors are built by Gram-Schmidt: project `ref_camera.right` (its
own horizontal axis) onto the plane orthogonal to `n`, normalize to get `e1`, then
`e2 = cross(n, e1)`. Using the camera's own right axis as the Gram-Schmidt seed (rather than an
arbitrary world axis) is deliberate: when a patch happens to be fronto-parallel to the reference
camera (`n == -forward`), `right` is already exactly in-plane, so `e1 == normalize(right)` and
`e2 == normalize(up)` with NO Gram-Schmidt correction needed, and the tangent-plane patch grid
becomes an EXACT (not first-order-approximate) reparameterization of the reference camera's own
neighbouring pixels -- this is what `tests/test_multiview_consistency.py`'s fronto-parallel test
exploits to check `planar_patch_warp` to tight (<1e-3, in practice much better) tolerance with
zero curvature error. For an oblique (non-fronto) patch this is only a first-order approximation,
which is exactly why `patch_radius` should stay small (a handful of pixels) in real use.

Physical patch spacing (`patch_step`) defaults to "roughly one reference-pixel's world-space
footprint at that depth": at the patch centre, `d0 = depth / ||x*right + y*up + forward||` is the
perpendicular distance implied by the pixel's own ray obliquity, and one column/row step in world
units is `d0 * (2/(W-1)) * ||right||` / `d0 * (2/(H-1)) * ||up||` respectively. This is exact for
a fronto-parallel patch (constant `d0` across the whole image, zero perspective curvature) and a
reasonable local linearization otherwise. A caller may override it with a fixed `patch_step` in
world units instead.

`sample_bilinear` mind the `grid_sample` coordinate convention: `align_corners=True` matches this
project's own `x = 2*j/(W-1) - 1` pixel mapping along the width axis directly, but the HEIGHT axis
needs its sign flipped (`grid_y = -y_norm`) because this project's `y` runs the opposite way from
`grid_sample`'s: `y = 1` here means row 0 (top), while `grid_sample`'s `align_corners=True`
convention puts row 0 at `grid_y = -1`. Getting this backwards would silently mirror every warped
patch vertically and was checked explicitly against the round-trip test on real (non-symmetric)
pixel coordinates while writing this module.

NCC, NOT MSE. `ncc` (zero-mean-normalized cross-correlation) is used instead of raw photometric
MSE specifically because two real cameras see different exposure/white-balance/vignetting even
of a genuinely matching surface patch; NCC is invariant to any independent per-patch affine
`a*x + b` (`a > 0`) intensity map, so it grades geometric (mis)match without conflating it with
photometric nuisance variation the way MSE would.

DEGENERATE CASE, MATCHING THE PATTERN THIS PROJECT ALREADY LEARNED THE HARD WAY IN
`normal_consistency.py`: if zero sampled patches end up valid (nothing to compare, e.g. all depth
invalid), `multiview_planar_ncc_loss` returns a plain torch.zeros((), ...) that is NOT connected
to `ref_depth`'s/`ref_normal`'s autograd graph via any `.sum() * 0.0`-style expression. That
pattern's backward is `expand()` -- a zero-stride broadcast view -- which the renderer's
Warp-backed custom autograd function (`RasterGradFn.backward`) rejects outright with
`RuntimeError: ... source inner strides are not contiguous` once `ref_depth`/`ref_normal` are the
renderer's own composited outputs (as `normal_consistency.py`'s docstring documents in detail,
from a real crash reproduced by exactly this branch). A disconnected zero is correct regardless:
there is genuinely no pixel pair to learn a correction from in this case.

This module itself performs no rendering and imports neither `warp` nor `powerfoam.camera`; it
only assumes a duck-typed camera object with `.eye`/`.right`/`.up`/`.width`/`.height`, so every
test in the paired test file runs on CPU with no GPU and no Warp kernel involved.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def unproject_pixel(camera, row, col, depth):
    """3D world point at pixel (row, col) and scalar `depth` along the unit ray, using the SAME
    formula as `geometry.py`'s `normals_from_depth` (that function is left untouched; this is a
    clean, non-`.cuda()`-hardcoded, CPU-or-GPU factoring of the same math for reuse here).

    `row`, `col`, `depth` may be any mutually-broadcastable shape; the result has that
    broadcast shape with a trailing size-3 dimension. Device/dtype follow `camera.eye`, never a
    hardcoded `.cuda()` -- unlike `normals_from_depth`, this runs equally on CPU or GPU.
    """
    eye = camera.eye
    up = camera.up
    right = camera.right
    dtype = eye.dtype
    device = eye.device

    row = torch.as_tensor(row, dtype=dtype, device=device)
    col = torch.as_tensor(col, dtype=dtype, device=device)
    depth = torch.as_tensor(depth, dtype=dtype, device=device)
    row, col, depth = torch.broadcast_tensors(row, col, depth)

    forward = torch.cross(up, right, dim=-1)
    forward = forward / torch.norm(forward)

    x = 2.0 * col / (float(camera.width) - 1.0) - 1.0
    y = 1.0 - 2.0 * row / (float(camera.height) - 1.0)

    ray_unnorm = x[..., None] * right + y[..., None] * up + forward
    ray_dir = ray_unnorm / torch.norm(ray_unnorm, dim=-1, keepdim=True)
    return eye + depth[..., None] * ray_dir


def project_point(camera, points, front_eps: float = 0.0, bound_eps: float = 1e-5):
    """Exact inverse of `unproject_pixel` / the ray-casting formula in `normals_from_depth`.

    For world point P, v = P - eye = a*right + b*up + c*forward (solved as a single batched
    matvec against the camera-only, point-independent M^{-1} -- no per-point 3x3 solve needed).
    Then x = a/c, y = b/c (valid only for c > front_eps, i.e. in front of the camera, and
    (x, y) in [-1 - bound_eps, 1 + bound_eps]); depth along the ray is exactly ||v|| (since a
    genuinely on-ray point satisfies P = eye + depth * unit_ray with unit_ray of unit norm,
    independent of x, y, c).

    `bound_eps` absorbs float32 rounding at the exact image border: a point unprojected from a
    border pixel (row/col 0 or W-1/H-1, i.e. x or y exactly +-1) and projected back can land at
    +-1.0000001 in float32 purely from roundoff, which a bound of exactly [-1, 1] would wrongly
    reject as invalid -- confirmed empirically (border pixels flip in/out of a strict [-1, 1]
    check depending on rounding direction) while writing this module's round-trip test.

    Fully differentiable w.r.t. `points` (needed for `multiview_planar_ncc_loss`'s gradient
    check) and never touches `.cuda()` -- works on whatever device `points`/`camera.eye` are on.

    Returns (x_norm, y_norm, depth_along_ray, valid_mask), each of shape `points.shape[:-1]`.
    """
    eye = camera.eye
    up = camera.up
    right = camera.right
    dtype = eye.dtype
    device = eye.device

    points = torch.as_tensor(points, dtype=dtype, device=device)
    forward = torch.cross(up, right, dim=-1)
    forward = forward / torch.norm(forward)

    M = torch.stack([right, up, forward], dim=1)  # columns: right, up, forward
    M_inv = torch.linalg.inv(M)

    v = points - eye
    abc = torch.einsum("ij,...j->...i", M_inv, v)
    a, b, c = abc[..., 0], abc[..., 1], abc[..., 2]

    depth = torch.norm(v, dim=-1)

    eps = 1e-8
    sign_c = torch.where(c == 0, torch.ones_like(c), torch.sign(c))
    c_safe = sign_c * c.abs().clamp_min(eps)
    x = a / c_safe
    y = b / c_safe

    valid = (
        (c > front_eps)
        & (x >= -1.0 - bound_eps) & (x <= 1.0 + bound_eps)
        & (y >= -1.0 - bound_eps) & (y <= 1.0 + bound_eps)
    )
    return x, y, depth, valid


def sample_bilinear(
    image: torch.Tensor, x_norm: torch.Tensor, y_norm: torch.Tensor, bound_eps: float = 1e-5
):
    """Bilinearly sample an (H, W, C) image at continuous normalized [-1, 1] coordinates.

    Uses `F.grid_sample(..., align_corners=True)`, which matches this project's own
    `x = 2*j/(W-1) - 1` pixel convention along the width axis directly. The height axis is
    flipped (`grid_y = -y_norm`) to reconcile this project's `y = 1 - 2*i/(H-1)` (row 0 -> y=+1)
    with `grid_sample`'s convention (row 0 -> grid_y=-1) -- see the module docstring.

    Out-of-bounds coordinates are NOT silently clamped/zero-padded into the loss: `grid_sample`'s
    own zero-padding is used only to keep the op numerically well-defined, and a separate,
    explicit `valid` mask (True iff both coordinates are within [-1, 1]) is returned so a caller
    can exclude them.
    """
    if image.dim() != 3:
        raise ValueError(f"expected an (H, W, C) image, got shape {tuple(image.shape)}")
    H, W, C = image.shape
    dtype = image.dtype
    device = image.device

    x_norm = torch.as_tensor(x_norm, dtype=dtype, device=device)
    y_norm = torch.as_tensor(y_norm, dtype=dtype, device=device)
    orig_shape = x_norm.shape

    xf = x_norm.reshape(-1)
    yf = y_norm.reshape(-1)
    grid = torch.stack([xf, -yf], dim=-1).view(1, 1, -1, 2)

    img = image.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    sampled = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.view(C, -1).permute(1, 0).reshape(*orig_shape, C)

    valid = (
        (xf >= -1.0 - bound_eps) & (xf <= 1.0 + bound_eps)
        & (yf >= -1.0 - bound_eps) & (yf <= 1.0 + bound_eps)
    )
    valid = valid.reshape(orig_shape)
    return sampled, valid


def planar_patch_warp(
    ref_camera,
    src_camera,
    ref_depth: torch.Tensor,
    ref_normal: torch.Tensor,
    ref_image: torch.Tensor,
    src_image: torch.Tensor,
    pixel_rows: torch.Tensor,
    pixel_cols: torch.Tensor,
    patch_radius: int,
    patch_step: float | None = None,
):
    """For each of N chosen reference-view pixel centres, build a `(2r+1)x(2r+1)` patch of 3D
    points lying on the LOCAL TANGENT PLANE defined by that pixel's depth+normal, project them
    into `src_camera`, and bilinearly sample `src_image` there. Also gathers the reference-image
    patch directly at the integer pixel grid (no warp needed -- it is already in the reference
    camera's own pixel grid).

    Deviates from the task sketch's exact argument list by also taking `ref_image`/`src_image`
    (needed to actually produce both patches) and `patch_step` (needed to control the physical
    patch footprint, as the task text itself calls for) -- see the module-level docstring for the
    reasoning behind the Gram-Schmidt seed and the default `patch_step` heuristic.

    Returns `(ref_patch, src_patch, valid)` of shape `(N, 2r+1, 2r+1, C)`, `(N, 2r+1, 2r+1, C)`,
    `(N,)`. `valid[k]` is True iff pixel k's reference depth is positive AND every pixel of its
    patch lands in-bounds in both the reference image and (after projection) the source image,
    AND every one of its tangent-plane points is in front of the source camera.
    """
    device = ref_depth.device
    dtype = ref_depth.dtype
    H, W = ref_depth.shape[-2], ref_depth.shape[-1]

    pixel_rows = torch.as_tensor(pixel_rows, dtype=torch.long, device=device)
    pixel_cols = torch.as_tensor(pixel_cols, dtype=torch.long, device=device)
    N = pixel_rows.shape[0]

    depth_c = ref_depth[pixel_rows, pixel_cols]  # (N,)
    normal_c = ref_normal[pixel_rows, pixel_cols]  # (N, 3)
    normal_unit = normal_c / normal_c.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    row_f = pixel_rows.to(dtype)
    col_f = pixel_cols.to(dtype)
    center_point = unproject_pixel(ref_camera, row_f, col_f, depth_c)  # (N, 3)

    right = ref_camera.right.to(dtype=dtype, device=device)
    up = ref_camera.up.to(dtype=dtype, device=device)
    forward = torch.cross(up, right, dim=-1)
    forward = forward / torch.norm(forward)

    right_hat = right / torch.norm(right)
    up_hat = up / torch.norm(up)

    # Gram-Schmidt: seed the in-plane basis from the reference camera's own horizontal axis
    # (falling back to its vertical axis when that axis is nearly parallel to the normal, i.e.
    # a near-edge-on view), then orthogonalize against the normal.
    helper = right_hat.unsqueeze(0).expand(N, -1).clone()
    dot_helper_normal = (normal_unit * helper).sum(-1)
    needs_fallback = dot_helper_normal.abs() > 0.9
    if bool(needs_fallback.any()):
        alt = up_hat.unsqueeze(0).expand(N, -1)
        helper = torch.where(needs_fallback.unsqueeze(-1), alt, helper)

    e1 = helper - (normal_unit * helper).sum(-1, keepdim=True) * normal_unit
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    e2 = torch.cross(normal_unit, e1, dim=-1)

    # Physical patch spacing: default to ~one reference-pixel's world footprint at this depth.
    x_c = 2.0 * col_f / (float(W) - 1.0) - 1.0
    y_c = 1.0 - 2.0 * row_f / (float(H) - 1.0)
    ray_unnorm_c = x_c[:, None] * right + y_c[:, None] * up + forward
    ray_unnorm_c_norm = torch.norm(ray_unnorm_c, dim=-1).clamp_min(1e-8)
    d0 = depth_c / ray_unnorm_c_norm

    if patch_step is None:
        step_x = d0 * (2.0 / (float(W) - 1.0)) * torch.norm(right)
        step_y = d0 * (2.0 / (float(H) - 1.0)) * torch.norm(up)
    else:
        step_x = torch.full_like(depth_c, float(patch_step))
        step_y = step_x

    r = int(patch_radius)
    offs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    di, dj = torch.meshgrid(offs, offs, indexing="ij")  # (ps, ps): di=row offset, dj=col offset

    # world_point(di, dj) = centre + dj*step_x*e1 - di*step_y*e2 -- the sign on the `di` (row)
    # term accounts for this project's y = 1 - 2*row/(H-1) convention (increasing row -> DEcreasing
    # y); see the module docstring's fronto-parallel derivation for why this is exact there.
    world_points = (
        center_point[:, None, None, :]
        + dj[None, :, :, None] * step_x[:, None, None, None] * e1[:, None, None, :]
        - di[None, :, :, None] * step_y[:, None, None, None] * e2[:, None, None, :]
    )  # (N, ps, ps, 3)

    x_src, y_src, _depth_src, front_valid = project_point(src_camera, world_points)
    src_patch, sample_valid = sample_bilinear(src_image, x_src, y_src)

    di_long = di.to(torch.long)
    dj_long = dj.to(torch.long)
    row_idx = pixel_rows[:, None, None] + di_long[None, :, :]
    col_idx = pixel_cols[:, None, None] + dj_long[None, :, :]
    in_bounds = (row_idx >= 0) & (row_idx < H) & (col_idx >= 0) & (col_idx < W)
    row_idx_c = row_idx.clamp(0, H - 1)
    col_idx_c = col_idx.clamp(0, W - 1)
    ref_patch = ref_image[row_idx_c, col_idx_c]  # (N, ps, ps, C)

    patch_ok = in_bounds & front_valid & sample_valid  # (N, ps, ps)
    valid = patch_ok.reshape(N, -1).all(dim=1) & (depth_c > 0)

    return ref_patch, src_patch, valid


def ncc(patch_a: torch.Tensor, patch_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean-normalized cross-correlation between same-shaped batches of patches, `(N, ...)`.

    Equivalent to `cosine_similarity` of each patch's own mean-centered, flattened values -- which
    is exactly why this (and not raw MSE) is used as the photometric term: it is invariant to each
    patch independently undergoing `a*x + b` for `a > 0` (the additive term cancels in the
    mean-centering, the positive scale cancels in the cosine normalization), so lighting/exposure
    differences between the two views cannot masquerade as -- or hide -- a real geometric error.
    Returns a value in [-1, 1] per patch.
    """
    a = patch_a.reshape(patch_a.shape[0], -1)
    b = patch_b.reshape(patch_b.shape[0], -1)
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return F.cosine_similarity(a, b, dim=-1, eps=eps)


def multiview_planar_ncc_loss(
    ref_camera,
    ref_image: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normal: torch.Tensor,
    src_camera,
    src_image: torch.Tensor,
    patch_radius: int = 2,
    patch_step: float | None = None,
    num_patches: int = 512,
    min_depth: float = 1e-4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """`mean(1 - ncc(ref_patch, src_patch))` over up to `num_patches` randomly sampled reference
    pixels whose depth exceeds `min_depth` and whose full `(2*patch_radius+1)` patch fits inside
    the reference frame, restricted further to whichever of those patches turn out fully valid
    after warping (in-bounds in the source view too, and in front of the source camera).

    DEGENERATE CASE (no reference pixel qualifies, or none of the sampled patches end up valid
    after warping): returns a plain, GRADIENT-DISCONNECTED `torch.zeros((), ...)` -- never a zero
    built from `ref_depth`/`ref_normal`/`ref_image` via e.g. `.sum() * 0.0`. See the module
    docstring: that pattern's backward is a zero-stride `expand()` view, which this project's
    Warp-backed renderer output has already been observed to reject outright once `ref_depth`/
    `ref_normal` are the renderer's own composited tensors. A disconnected zero is correct here
    regardless -- there is no valid patch pair to learn a correction from.
    """
    H, W = ref_depth.shape[-2], ref_depth.shape[-1]
    device = ref_depth.device
    r = int(patch_radius)

    if H <= 2 * r or W <= 2 * r:
        return torch.zeros((), dtype=ref_depth.dtype, device=device)

    valid_center = ref_depth > min_depth

    row_idx = torch.arange(H, device=device)
    col_idx = torch.arange(W, device=device)
    row_ok = (row_idx >= r) & (row_idx <= H - 1 - r)
    col_ok = (col_idx >= r) & (col_idx <= W - 1 - r)
    border_mask = row_ok[:, None] & col_ok[None, :]

    candidate_mask = valid_center & border_mask
    candidate_rows, candidate_cols = torch.nonzero(candidate_mask, as_tuple=True)
    n_candidates = candidate_rows.shape[0]
    if n_candidates == 0:
        return torch.zeros((), dtype=ref_depth.dtype, device=device)

    n_sample = min(int(num_patches), n_candidates)
    if generator is not None:
        perm = torch.randperm(n_candidates, generator=generator)
    else:
        perm = torch.randperm(n_candidates)
    perm = perm.to(device)
    sel = perm[:n_sample]
    sel_rows = candidate_rows[sel]
    sel_cols = candidate_cols[sel]

    ref_patch, src_patch, valid = planar_patch_warp(
        ref_camera,
        src_camera,
        ref_depth,
        ref_normal,
        ref_image,
        src_image,
        sel_rows,
        sel_cols,
        patch_radius,
        patch_step=patch_step,
    )

    if not bool(valid.any()):
        return torch.zeros((), dtype=ref_depth.dtype, device=device)

    ncc_vals = ncc(ref_patch[valid], src_patch[valid])
    return (1.0 - ncc_vals).mean()
