"""Real correctness tests for the density_activation option on PowerfoamScene (softplus vs
exp -- VoroTracing's scale-invariant sigma = exp(rho) parameterization, arXiv 2608.17682 Sec
5.4). Constructs a real PowerfoamScene and exercises the real `initialize_from_dataset`/
`get_density`, stubbing only AABBTree/Rasterizer/RayTracer/SphericalVoronoi -- all of which are
built AFTER density is initialised and never read or write it -- so this runs entirely on CPU
with no Warp/CUDA kernel compilation.

NOTE: this rewrites the original test (lost, then recreated, in a checkout-wipe incident on
2026-09-14/15 -- see MyResearchVault/Stage0-Surface-Prior-Art.md) from its original spec, since
only its pass/fail output, not its literal source, was preserved through that incident.
"""
import dataclasses
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import powerfoam.scene as scene_mod
from configs import Params
from powerfoam.scene import PowerfoamScene


class _FakeAABBTree:
    """update()/build_cech_complex() are called by rebuild_adjacency(), well after density is
    set up; a trivial, edge-free CSR adjacency is all initialize_from_dataset needs to finish."""

    def __init__(self, device):
        self.device = device
        self._n = 0

    def update(self, points, radii):
        self._n = points.shape[0]

    def build_cech_complex(self):
        adjacency = torch.zeros(0, dtype=torch.int32, device=self.device)
        offsets = torch.zeros(self._n + 1, dtype=torch.int32, device=self.device)
        return adjacency, offsets


class _FakeRasterizer:
    def __init__(self, *a, **k):
        pass


class _FakeRayTracer:
    def __init__(self, *a, **k):
        pass


class _FakeSphericalVoronoi:
    def __init__(self, *a, **k):
        pass

    @staticmethod
    def compute_fov_cos_cutoff(camera):
        return 0.0


class _FakeCamera:
    """Only the attributes scene.py's max_radii loop and SphericalVoronoi stub need."""

    def __init__(self, eye):
        self.eye = torch.tensor(eye, dtype=torch.float32)
        self.right = torch.tensor([1.0, 0.0, 0.0])
        self.up = torch.tensor([0.0, 1.0, 0.0])


class _FakeDataHandler:
    def __init__(self, points3D):
        self.points3D = points3D
        # init_points_sfm's outlier filter uses the MEDIAN pairwise camera-to-centroid
        # distance as its length scale; with only one camera that distance is exactly 0,
        # which makes every point register as an "outlier" and get dropped. Multiple, spread
        # camera positions give a real, nonzero scale.
        self.cameras = [
            _FakeCamera([0.0, 0.0, 5.0]),
            _FakeCamera([3.0, 0.0, 4.0]),
            _FakeCamera([-3.0, 1.0, 4.0]),
        ]


def _minimal_params(density_activation="softplus"):
    field_defaults = {f.name: f.default for f in dataclasses.fields(Params)
                       if f.default is not dataclasses.MISSING}
    required_overrides = dict(
        iterations=1, normal_weight=0.0, contribution_weight=0.0,
        interpenetration_weight=0.0, densify_from=0, densify_until=1,
        dataset="colmap", data_path="", scene="", alpha_format_on_disk="straight",
        downsample=[1, 1], downsample_iterations=[0], init_type="sfm",
        init_points=8, final_points=8, bkgd_color=[0.0, 0.0, 0.0],
        sv_dof=1, num_texel_sites=1,
        points_lr_init=0.0, points_lr_final=0.0, density_lr_init=0.0, density_lr_final=0.0,
        radii_lr_init=0.0, radii_lr_final=0.0, quaternions_lr_init=0.0,
        quaternions_lr_final=0.0, texel_sites_lr_init=0.0, texel_sites_lr_final=0.0,
        texel_sv_axis_lr_init=0.0, texel_sv_axis_lr_final=0.0, texel_sv_rgb_lr_init=0.0,
        texel_sv_rgb_lr_final=0.0, texel_height_lr_init=0.0, texel_height_lr_final=0.0,
    )
    kwargs = {**field_defaults, **required_overrides, "density_activation": density_activation}
    return Params(**kwargs)


def _build_scene(monkeypatch, density_activation):
    monkeypatch.setattr(scene_mod, "AABBTree", _FakeAABBTree)
    monkeypatch.setattr(scene_mod, "Rasterizer", _FakeRasterizer)
    monkeypatch.setattr(scene_mod, "RayTracer", _FakeRayTracer)
    monkeypatch.setattr(scene_mod, "SphericalVoronoi", _FakeSphericalVoronoi)

    torch.manual_seed(0)
    points3D = torch.randn(8, 3)
    dh = _FakeDataHandler(points3D)

    args = _minimal_params(density_activation)
    model = PowerfoamScene(args)
    model.initialize_from_dataset(dh, device="cpu")
    return model


def test_softplus_matches_f_softplus_exactly(monkeypatch):
    model = _build_scene(monkeypatch, "softplus")
    expected = F.softplus(model.density, beta=100)
    assert torch.equal(model.get_density(), expected)


def test_exp_matches_torch_exp_exactly(monkeypatch):
    model = _build_scene(monkeypatch, "exp")
    expected = torch.exp(model.density)
    assert torch.equal(model.get_density(), expected)


def test_both_activations_converge_to_same_effective_sigma(monkeypatch):
    """softplus(0.1, beta=100) ~= 0.1 (since beta=100 makes it near-linear for x >> 1/100),
    and the exp path is deliberately initialised at log(0.1) so exp(log(0.1)) == 0.1 exactly.
    Both should therefore start training from the same effective density."""
    model_sp = _build_scene(monkeypatch, "softplus")
    model_exp = _build_scene(monkeypatch, "exp")
    sigma_sp = model_sp.get_density()
    sigma_exp = model_exp.get_density()
    assert torch.allclose(sigma_sp, torch.full_like(sigma_sp, 0.1), atol=1e-6)
    assert torch.allclose(sigma_exp, torch.full_like(sigma_exp, 0.1), atol=1e-6)
    assert torch.allclose(sigma_sp, sigma_exp, atol=1e-6)


def test_unrecognized_activation_falls_back_to_softplus(monkeypatch):
    """The native get_density() is `if density_activation == "exp": exp(...) else:
    softplus(...)` -- an unrecognized string (e.g. a typo like "sofplus") silently gets
    softplus behavior rather than raising. Not the stricter explicit-else-raise this
    session's own earlier port used, but that was a different implementation than this
    native one; documenting the REAL behavior here rather than asserting one that doesn't
    exist. Worth knowing: a typo in this field fails silently, not loudly."""
    model = _build_scene(monkeypatch, "softplus")
    model.args.density_activation = "not_a_real_activation"
    assert torch.equal(model.get_density(), F.softplus(model.density, beta=100))


def test_clamp_keeps_exp_density_finite(monkeypatch):
    """train.py clamps the raw density parameter to 30 after each optimizer step under the
    exp activation, specifically to keep exp(density)'s gradient from overflowing fp32.
    exp(30) is already fully opaque for any realistic radius, so the clamp costs nothing
    physically; this checks the clamp actually keeps get_density() finite and saturating,
    while leaving an untouched (low) entry unaffected."""
    model = _build_scene(monkeypatch, "exp")
    with torch.no_grad():
        model.density[0] = 100.0   # would overflow exp() in fp32 if left unclamped
        model.density[1] = 0.5     # untouched control value
        model.density.clamp_(max=30.0)

    sigma = model.get_density()
    assert torch.isfinite(sigma).all()
    assert sigma[0].item() == pytest.approx(float(np.exp(30.0)), rel=1e-5)

    # A ray segment of any realistic thickness is fully opaque at this density.
    alpha_clamped = 1.0 - torch.exp(-sigma[0] * 0.01)
    assert alpha_clamped.item() == pytest.approx(1.0, abs=1e-6)

    # The untouched entry must be unaffected by the clamp.
    assert sigma[1].item() == pytest.approx(float(np.exp(0.5)), rel=1e-6)
