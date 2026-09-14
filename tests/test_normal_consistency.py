"""Real correctness tests for powerfoam/normal_consistency.py -- specifically targeting the
bug this module fixes (magnitude/opacity leaking into a loss meant to grade direction only)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from powerfoam.normal_consistency import depth_normal_consistency_loss


def test_perfect_alignment_gives_zero_loss():
    """A composited normal already pointing exactly along the target must score ~0, regardless
    of magnitude (magnitude = accumulated opacity, which this loss must NOT reward directly)."""
    target = torch.tensor([[0.0, 0.0, 1.0]]).expand(4, 3)
    normal = target.clone() * 0.9  # magnitude 0.9: high but not saturated opacity
    alpha = torch.tensor([0.9, 0.9, 0.9, 0.9])
    valid = torch.ones(4, dtype=torch.bool)
    loss = depth_normal_consistency_loss(normal, alpha, valid, target, min_alpha=0.5)
    assert loss.item() < 1e-5, loss.item()


def test_magnitude_does_not_affect_loss_for_fixed_direction():
    """THE REGRESSION TEST for the bug: two composited normals with the SAME direction but
    DIFFERENT magnitudes (i.e. different accumulated opacity) must give the IDENTICAL loss.
    Under the old F.mse_loss(normal, target) formulation this would NOT hold -- a lower-
    magnitude normal would score a larger (wrong) loss purely from its smaller norm, not from
    any direction error."""
    direction = F.normalize(torch.tensor([[0.3, 0.5, 0.8]]), dim=-1)
    target = F.normalize(torch.tensor([[0.1, 0.2, 0.9]]), dim=-1)  # some fixed, imperfect target
    alpha = torch.tensor([0.9])
    valid = torch.ones(1, dtype=torch.bool)

    loss_full_opacity = depth_normal_consistency_loss(
        direction * 1.0, alpha, valid, target, min_alpha=0.5
    )
    loss_low_opacity = depth_normal_consistency_loss(
        direction * 0.15, alpha, valid, target, min_alpha=0.1
    )
    assert torch.allclose(loss_full_opacity, loss_low_opacity, atol=1e-6), (
        loss_full_opacity.item(), loss_low_opacity.item()
    )


def test_old_mse_formulation_would_have_failed_this_case():
    """Demonstrates the bug directly: construct a normal with PERFECT direction but low
    magnitude, and show the OLD F.mse_loss formulation scores it as clearly wrong (high loss)
    while the FIXED cosine formulation correctly scores it as perfect (near-zero loss)."""
    target = F.normalize(torch.tensor([[0.0, 0.0, 1.0]]), dim=-1)
    low_opacity_but_perfect_direction = target * 0.2  # magnitude 0.2 = low accumulated opacity
    alpha = torch.tensor([0.9])
    valid = torch.ones(1, dtype=torch.bool)

    old_mse = F.mse_loss(low_opacity_but_perfect_direction, target).item()
    new_cosine_loss = depth_normal_consistency_loss(
        low_opacity_but_perfect_direction, alpha, valid, target, min_alpha=0.5
    ).item()

    assert old_mse > 0.1, f"expected the old formulation to score this as clearly wrong, got {old_mse}"
    assert new_cosine_loss < 1e-5, f"expected the fixed formulation to score perfect direction as ~0, got {new_cosine_loss}"


def test_low_opacity_pixels_are_excluded():
    """A pixel below min_alpha must not contribute, even if its normal is maximally wrong."""
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    normal = torch.stack([
        torch.tensor([0.0, 0.0, 1.0]),   # perfect, high alpha -- included
        torch.tensor([0.0, 0.0, -1.0]),  # maximally wrong, LOW alpha -- must be excluded
    ])
    alpha = torch.tensor([0.9, 0.05])
    valid = torch.ones(2, dtype=torch.bool)
    loss = depth_normal_consistency_loss(normal, alpha, valid, target, min_alpha=0.5)
    assert loss.item() < 1e-5, loss.item()


def test_invalid_depth_pixels_are_excluded():
    """A pixel with invalid depth must not contribute even at high opacity."""
    target = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    normal = torch.stack([
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([1.0, 0.0, 0.0]),  # wrong, but depth invalid here
    ])
    alpha = torch.tensor([0.9, 0.9])
    valid = torch.tensor([True, False])
    loss = depth_normal_consistency_loss(normal, alpha, valid, target, min_alpha=0.5)
    assert loss.item() < 1e-5, loss.item()


def test_no_qualifying_pixels_returns_disconnected_zero():
    """When nothing qualifies (e.g. very early training, nothing opaque yet), the function
    must return a plain zero NOT connected to `normal`'s graph. `normal` is the renderer's
    own Warp-backed composited output; a "connected" zero via `normal.sum() * 0.0` uses
    `.sum()`'s expand()-based backward, a zero-stride broadcast VIEW that the renderer's
    custom backward kernel rejects as non-contiguous (reproduced as a real crash during
    integration testing -- this is not a hypothetical). A disconnected zero is correct
    regardless: there is no pixel to learn an orientation from in this case, so no gradient
    should reach `normal` here. The result can still always be added into a caller's loss
    sum unconditionally -- it just contributes nothing, to numerics or to gradients."""
    normal = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3)
    alpha = torch.tensor([0.01, 0.01, 0.01])
    valid = torch.ones(3, dtype=torch.bool)
    loss = depth_normal_consistency_loss(normal, alpha, valid, target, min_alpha=0.5)
    assert loss.item() == 0.0
    assert not loss.requires_grad
    # still safe to add into a larger loss sum that DOES require grad
    total = normal.sum() * 0.0 + loss
    total.backward()  # must not raise (this whole expression is never fed to the renderer)


def test_gradient_matches_finite_difference():
    torch.manual_seed(0)
    normal = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    target = F.normalize(torch.randn(5, 3, dtype=torch.float64), dim=-1)
    alpha = torch.tensor([0.9, 0.9, 0.6, 0.9, 0.2], dtype=torch.float64)
    valid = torch.tensor([True, True, True, False, True])

    def loss_fn(n):
        return depth_normal_consistency_loss(n, alpha, valid, target, min_alpha=0.5)

    loss = loss_fn(normal)
    loss.backward()
    analytic = normal.grad.clone()

    eps = 1e-6
    numeric = torch.zeros_like(normal)
    flat_n, flat_g = normal.detach().view(-1), numeric.view(-1)
    for k in range(flat_n.numel()):
        d = torch.zeros_like(flat_n); d[k] = eps
        with torch.no_grad():
            lp = loss_fn((flat_n + d).view_as(normal))
            lm = loss_fn((flat_n - d).view_as(normal))
        flat_g[k] = (lp - lm) / (2 * eps)

    assert torch.allclose(analytic, numeric, atol=1e-5), (analytic, numeric)
