"""Standalone synthetic validation for `powerfoam/multiview_consistency.py` ("idea C" of the
surface-aware-training project plan: PGSR-style multi-view planar-patch NCC consistency).

Not part of the installable package and not exercised by pytest -- a hand-run report script, in
the same spirit as `feature-foam-lifting/scripts/*.py`. Builds the SAME hand-constructed,
renderer-independent two-camera tilted-plane scene as `tests/test_multiview_consistency.py`
(construction duplicated here rather than factored into a shared helper -- it is ~40 lines and
this script has no other reason to import the test file), then:

  1. reports the loss at ground-truth depth/normal (should be ~0),
  2. sweeps depth error and normal-tilt error and reports the loss at each (should increase
     monotonically with the error magnitude),
  3. runs plain Adam gradient descent from a perturbed depth/normal (+20% depth error, 30 degree
     normal tilt), using ONLY this loss as the objective, and reports whether it recovers the
     true depth/normal.

Runs on CPU only; no GPU, no Warp, no PowerFoam checkpoint involved.
"""
import math

import torch

from powerfoam.multiview_consistency import multiview_planar_ncc_loss


class _Camera:
    def __init__(self, eye, right, up, width, height):
        self.eye = eye
        self.right = right
        self.up = up
        self.width = width
        self.height = height


def _normalize(v):
    return v / v.norm()


def _make_camera_basis(forward_dir, right_scale, up_scale, dtype):
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


def make_scene(dtype=torch.float64, H=64, W=80):
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
    ref_normal = (-N).expand(H, W, 3).clone()

    src_points, _ = ray_plane_intersect(src_camera, rows_grid, cols_grid)
    u_src, v_src = plane_uv(src_points)
    src_image = texture(u_src, v_src).unsqueeze(-1).repeat(1, 1, 3)

    return dict(
        ref_camera=ref_camera, src_camera=src_camera,
        ref_image=ref_image, src_image=src_image,
        ref_depth=ref_depth, ref_normal=ref_normal,
        Q0=Q0, N=N, U=U, V=V,
    )


def tilt_normal(normal, angle_deg, axis):
    angle = torch.tensor(angle_deg * math.pi / 180.0, dtype=normal.dtype)
    axis = (axis / axis.norm()).expand_as(normal)
    return (
        normal * torch.cos(angle)
        + torch.cross(axis, normal, dim=-1) * torch.sin(angle)
        + axis * (axis * normal).sum(-1, keepdim=True) * (1 - torch.cos(angle))
    )


def loss_at(scene, ref_depth, ref_normal, seed=0, num_patches=400):
    gen = torch.Generator().manual_seed(seed)
    return multiview_planar_ncc_loss(
        scene["ref_camera"], scene["ref_image"], ref_depth, ref_normal,
        scene["src_camera"], scene["src_image"],
        patch_radius=2, num_patches=num_patches, min_depth=1e-3, generator=gen,
    )


def main():
    torch.manual_seed(0)
    scene = make_scene()
    true_depth = scene["ref_depth"]
    true_normal = scene["ref_normal"]

    print("=" * 70)
    print("1) Loss at ground-truth depth/normal")
    print("=" * 70)
    l0 = loss_at(scene, true_depth, true_normal, seed=0)
    print(f"loss(true depth, true normal) = {l0.item():.8f}")

    print()
    print("=" * 70)
    print("2) Depth-error sweep (normal held at ground truth)")
    print("=" * 70)
    print(f"{'depth error':>12} | {'loss':>12}")
    for frac in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
        d = true_depth * (1.0 + frac)
        l = loss_at(scene, d, true_normal, seed=1)
        print(f"{frac * 100:>10.1f}% | {l.item():>12.6f}")

    print()
    print("=" * 70)
    print("3) Normal-tilt sweep (depth held at ground truth)")
    print("=" * 70)
    print(f"{'tilt (deg)':>12} | {'loss':>12}")
    for deg in [0.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
        n = tilt_normal(true_normal, deg, scene["U"])
        l = loss_at(scene, true_depth, n, seed=1)
        print(f"{deg:>12.1f} | {l.item():>12.6f}")

    print()
    print("=" * 70)
    print("4) Gradient-descent recovery from a perturbed depth/normal")
    print("=" * 70)
    init_depth = (true_depth * 1.20).clone().requires_grad_(True)
    init_normal = tilt_normal(true_normal, 30.0, scene["U"]).clone().requires_grad_(True)

    def relative_depth_error(d):
        return ((d - true_depth).abs() / true_depth).mean().item()

    def normal_angle_error_deg(n):
        n_hat = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        t_hat = true_normal / true_normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (n_hat * t_hat).sum(-1).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cos)).mean().item()

    print(f"before: mean relative depth error = {relative_depth_error(init_depth.detach()):.4f}, "
          f"mean normal angular error = {normal_angle_error_deg(init_normal.detach()):.2f} deg")
    print(f"initial loss = {loss_at(scene, init_depth.detach(), init_normal.detach(), seed=2).item():.6f}")

    depth_param = init_depth.detach().clone().requires_grad_(True)
    normal_param = init_normal.detach().clone().requires_grad_(True)
    # Separate param groups (both lr=0.02 here) mainly so a caller could easily give depth/normal
    # different step sizes; normal is renormalized to unit length after every step since Adam
    # has no notion that it lives on a sphere.
    optimizer = torch.optim.Adam(
        [{"params": [depth_param], "lr": 0.02}, {"params": [normal_param], "lr": 0.02}]
    )

    n_steps = 500
    for step in range(n_steps):
        optimizer.zero_grad()
        gen = torch.Generator().manual_seed(1000 + step)
        loss = multiview_planar_ncc_loss(
            scene["ref_camera"], scene["ref_image"], depth_param, normal_param,
            scene["src_camera"], scene["src_image"],
            patch_radius=2, num_patches=400, min_depth=1e-3, generator=gen,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            normal_param /= normal_param.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: loss={loss.item():.6f}  "
                  f"rel_depth_err={relative_depth_error(depth_param.detach()):.4f}  "
                  f"normal_err_deg={normal_angle_error_deg(normal_param.detach()):.2f}")

    final_loss = loss_at(scene, depth_param.detach(), normal_param.detach(), seed=2)
    print()
    print(f"after {n_steps} steps: mean relative depth error = "
          f"{relative_depth_error(depth_param.detach()):.4f}, mean normal angular error = "
          f"{normal_angle_error_deg(normal_param.detach()):.2f} deg")
    print(f"final loss = {final_loss.item():.6f}")

    print()
    print("-" * 70)
    print("Diagnostic: does it keep improving with more steps? (not part of the spec's 200-500")
    print("step request, but the loss above plateaus noisily while the geometric errors are")
    print("still trending down -- run further to see whether that trend is real.)")
    print("-" * 70)
    n_extra = 1500
    for step in range(n_steps, n_steps + n_extra):
        optimizer.zero_grad()
        gen = torch.Generator().manual_seed(1000 + step)
        loss = multiview_planar_ncc_loss(
            scene["ref_camera"], scene["ref_image"], depth_param, normal_param,
            scene["src_camera"], scene["src_image"],
            patch_radius=2, num_patches=400, min_depth=1e-3, generator=gen,
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            normal_param /= normal_param.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if (step - n_steps) % 200 == 0 or step == n_steps + n_extra - 1:
            print(f"  step {step:4d}: loss={loss.item():.6f}  "
                  f"rel_depth_err={relative_depth_error(depth_param.detach()):.4f}  "
                  f"normal_err_deg={normal_angle_error_deg(normal_param.detach()):.2f}")
    print()
    print(f"after {n_steps + n_extra} steps total: mean relative depth error = "
          f"{relative_depth_error(depth_param.detach()):.4f}, mean normal angular error = "
          f"{normal_angle_error_deg(normal_param.detach()):.2f} deg")


if __name__ == "__main__":
    main()
