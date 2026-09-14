"""Per-primitive class colours for the 3D segmentation figure.

Shared by render_seg_powerfoam.py and gsplat_baseline/render_seg_gsplat.py so
the two arms cannot drift: a chair must be the same hue in both renders, or the
figure is unreadable as a comparison.

Colour follows the CLASS NAME, never its rank in whatever class list was passed.
If it followed rank, dropping one class from the set would repaint every class
after it and two panels of the same figure would disagree.

Background classes are deliberately NOT given a hue. They are the bulk of a room
and colouring them buries the objects; sending them to grey at reduced opacity
is what makes the foreground legible, and is what the ScanNet++ version of this
figure wants ("take out the ones that can be background like walls").

Primitives with no usable feature go to grey too, explicitly rather than by
accident: 17.0% of the gs_frozen scene0062_00 features are zero-norm
(unobserved), and handing those to argmax would assign each an arbitrary class
and speckle the render with false positives.

The eight hues are categorical slots 1-8 of the project palette, validated for
CVD separation (worst adjacent pair dE 9.1 protan / 5.8 tritan, against a floor
of 8 with secondary encoding). Three of them fall below 3:1 contrast on a light
surface, so any figure using this MUST carry a named class legend -- that is the
required relief, not an optional nicety.
"""
from __future__ import annotations

import sys

# powerfoam may import feature_foam_lifting (the boundary rule runs the other
# way: the solver repo must never import powerfoam). scene.py does the same.
# Our src must come FIRST: the shared conda env has an editable install of an
# older feature_foam_lifting that has no group.py, and appending lets that one
# win (see CLAUDE.md, "prepend your own src so your checkout wins").
_OURS = "/home/x_pelcasae/feature-foam-lifting/src"
if _OURS in sys.path:
    sys.path.remove(_OURS)
sys.path.insert(0, _OURS)
for _m in [k for k in list(sys.modules) if k.startswith("feature_foam_lifting")]:
    del sys.modules[_m]
if "/home/rajehyl/powerfoam" not in sys.path:
    sys.path.append("/home/rajehyl/powerfoam")

from run_cluster_classify_eval import pool_classify_broadcast

# Slots 1-8, in fixed order. Never cycle: a 9th foreground class folds to grey
# rather than reusing a hue, because two classes sharing a colour is worse than
# one class being unlabelled.
_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

BACKGROUND = {"wall", "floor", "ceiling"}

# Open-vocabulary mode needs distractors. With a closed label set, argmax over
# the set is well posed. With two or three free-text queries it is not: every
# primitive in the scene would be forced into the nearest query, so a room with
# one chair comes out entirely "chair". Scoring the queries against negatives
# and keeping only primitives whose argmax IS a query is the standard fix
# (LERF / LangSplat do the same), and it is what makes "desaturate everything
# that doesn't belong to the class" mean anything on ScanNet++ and LERF, where
# there is no wall/floor class to send to grey.
DEFAULT_NEGATIVES = ("wall", "floor", "ceiling", "room", "object", "texture",
                     "surface", "background")
GREY = (0.60, 0.60, 0.585)


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def class_colour_map(class_names, only=None):
    """{class name -> rgb}. Background, overflow, and anything outside `only`
    map to grey.

    `only` is what makes the figure readable. Colouring all eight foreground
    classes at once produces a patchwork: the classifier runs at roughly 40
    mIoU on this data and each of the K groups is classified independently, so
    neighbouring groups on one wall disagree and the render reads as noise
    rather than as segmentation. Restricting to one or two query classes is the
    standard form for this kind of figure and is what "desaturate everything
    that does not belong to a class" actually asks for.
    """
    fg = [c for c in class_names if c not in BACKGROUND]
    if only:
        fg = [c for c in fg if c in set(only)]
    out = {c: GREY for c in class_names}
    for i, c in enumerate(fg[:len(_HUES)]):
        out[c] = _hex_to_rgb(_HUES[i])
    return out


def legend_entries(class_names, present=None, only=None):
    """(name, rgb) for the foreground classes that earned a hue, in slot order.

    `present` restricts to classes actually rendered, so the legend never
    advertises a class the figure does not show.
    """
    cmap = class_colour_map(class_names, only=only)
    fg = [c for c in class_names if c not in BACKGROUND]
    if only:
        fg = [c for c in fg if c in set(only)]
    fg = fg[:len(_HUES)]
    return [(c, cmap[c]) for c in fg if present is None or c in present]


def query_colours(features, queries, device, negatives=DEFAULT_NEGATIVES,
                  num_groups=320, margin=0.0):
    """Open-vocabulary variant: colour only what matches a free-text query.

    Returns (colours (P,3), matched-query index per primitive with -1 for
    "no query", info). Groups are formed exactly as in `primitive_colours`
    (feature-space k-means, no graph) so the two modes and the two
    representations stay comparable.

    `margin` requires the winning query to beat the best negative by this much
    in cosine before a group is coloured; 0.0 means a plain argmax over
    queries + negatives. Raise it to shrink the highlighted region when a query
    bleeds into the background.
    """
    import torch
    from feature_foam_lifting.group import feature_space_groups

    queries = list(queries)
    negs = [n for n in negatives if n not in queries]
    f = features.to(device).float()
    usable = f.norm(dim=-1) > 0
    unit = torch.nn.functional.normalize(f, dim=-1)
    text = _text(tuple(queries + negs), device)

    colours = torch.empty((f.shape[0], 3), dtype=torch.float32, device=device)
    colours[:] = torch.tensor(GREY, dtype=torch.float32, device=device)
    match = torch.full((f.shape[0],), -1, dtype=torch.long, device=device)
    if not usable.any():
        return colours, match, {"num_primitives": int(f.shape[0]), "matched": 0}

    labels, _sz, _rj = feature_space_groups(unit, num_groups, usable,
                                            num_iters=15, seed=0)
    keep = labels >= 0
    pooled = torch.zeros(num_groups, unit.shape[1], device=device)
    pooled.index_add_(0, labels[keep], unit[keep])
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = pooled @ text.T                                  # (G, Q + N)
    q_best, q_arg = sim[:, :len(queries)].max(dim=-1)
    n_best = sim[:, len(queries):].max(dim=-1).values if negs else \
        torch.full_like(q_best, -1.0)
    won = q_best > (n_best + margin)

    hues = [_hex_to_rgb(h) for h in _HUES]
    palette = torch.tensor([hues[i % len(hues)] for i in range(len(queries))],
                           dtype=torch.float32, device=device)
    g_match = torch.where(won, q_arg, torch.full_like(q_arg, -1))
    match[keep] = g_match[labels[keep]]
    got = match >= 0
    colours[got] = palette[match[got]]
    return colours, match, {
        "num_primitives": int(f.shape[0]),
        "unusable_features": int((~usable).sum()),
        "matched": int(got.sum()),
        "groups_won": int(won.sum()),
        "num_groups": num_groups,
        "queries": queries, "negatives": negs, "margin": margin,
        "per_query": {q: int((match == i).sum()) for i, q in enumerate(queries)},
    }


def query_legend(queries):
    return [(q, _hex_to_rgb(_HUES[i % len(_HUES)])) for i, q in enumerate(queries)]


def primitive_colours(features, class_names, device, num_groups=320,
                      pooled=True, hubness_correct=True, only=None):
    """(P, 3) float RGB per primitive, plus a stats dict.

    `features` is the solved (P, 512) field for THIS arm -- never the other
    arm's, since the two have different primitive counts and orderings.

    `pooled=True` is the method, and it matters enormously for how the figure
    looks. Classifying each primitive independently is the weakest variant: on
    scene0062_00 it labelled the walls "toilet" and "sofa" and put toilet at
    26,731 primitives against wall's 15,202, in a room whose GT is ~48% wall.
    Grouping first and classifying the pooled group centroid -- which is what
    the paper's pipeline actually does -- averages 512-d features over hundreds
    of primitives before the argmax, and that is where the denoising comes from.

    The grouping used is feature-space k-means with no neighbour graph
    (`feature_space_groups`). Two reasons: it was the strongest arm we measured
    on GT-label purity, and it needs no adjacency, so the SAME grouping runs on
    3DGS, which has no adjacency to offer. A figure whose two panels used
    different grouping methods would not be a comparison of representations.
    """
    import torch
    from evaluate_point_cloud_miou import classify_primitives

    f = features.to(device).float()
    usable = f.norm(dim=-1) > 0
    cmap = class_colour_map(class_names, only=only)
    palette = torch.tensor([cmap[c] for c in class_names], dtype=torch.float32,
                           device=device)

    colours = torch.empty((f.shape[0], 3), dtype=torch.float32, device=device)
    colours[:] = torch.tensor(GREY, dtype=torch.float32, device=device)
    cls = torch.full((f.shape[0],), -1, dtype=torch.long, device=device)
    text = _text(class_names, device)
    unit = torch.nn.functional.normalize(f, dim=-1)

    if usable.any():
        if pooled:
            from feature_foam_lifting.group import feature_space_groups
            labels, _sizes, _rej = feature_space_groups(
                unit, num_groups, usable, num_iters=15, seed=0)
            # Unassigned primitives keep -1 and stay grey; pooling needs a
            # contiguous label space, so classify on the assigned subset.
            keep = labels >= 0
            c_all = pool_classify_broadcast(labels[keep], unit[keep],
                                            num_groups, text)
            cls[keep] = c_all
        else:
            cls[usable] = classify_primitives(f[usable], text,
                                              hubness_correct=hubness_correct)
        got = cls >= 0
        colours[got] = palette[cls[got]]

    hued = {c for c, v in cmap.items() if v != GREY}
    bg = {i for i, c in enumerate(class_names) if c not in hued}
    is_bg = torch.zeros_like(usable)
    for i in bg:
        is_bg |= cls == i
    is_fg = (cls >= 0) & ~is_bg
    return colours, cls, {
        "num_primitives": int(f.shape[0]),
        "unusable_features": int((~usable).sum()),
        "unlabelled": int((cls < 0).sum()),
        "background": int(is_bg.sum()),
        "foreground": int(is_fg.sum()),
        "grouping": f"feature_space_groups(K={num_groups})" if pooled else "per-primitive",
        "highlighted_classes": sorted(hued),
        "per_class": {c: int((cls == i).sum()) for i, c in enumerate(class_names)},
    }


_TEXT_CACHE = {}


def _text(class_names, device):
    """Text embeddings, cached: building them loads the whole CLIP text tower,
    which costs more than every render in this figure combined."""
    key = (tuple(class_names), str(device))
    if key not in _TEXT_CACHE:
        from evaluate_point_cloud_miou import embed_class_names
        _TEXT_CACHE[key] = embed_class_names(list(class_names), device)
    return _TEXT_CACHE[key]
