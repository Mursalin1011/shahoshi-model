"""Which model output is which head.

Separated from `shahoshi.quantize` so it imports without TensorFlow and can be
unit-tested. That separation is the point: this logic failing is silent, and it
already cost one full training run.
"""

from __future__ import annotations


def assign_roles(
    widths: list[int],
    n_classes: int,
    embed_dim: int,
    with_fall_head: bool = False,
) -> dict[str, int]:
    """Map each model output to its role, by width. Pure, so it is testable.

    **This is not a nicety.** TFLite does not preserve the Keras output order:
    the converter emits outputs in its own graph order, so the softmax may land
    at index 1 and the embedding at index 0. Reading them positionally produced
    an int8 accuracy of 0.0196 against a float accuracy of 0.9024 -- argmax over
    a 64-wide embedding returns 0..63, which almost never equals a label in
    0..5. The model was fine; the indexing was wrong.

    Roles are therefore identified by the one property the converter cannot
    scramble -- how wide each output is. Ambiguity raises rather than guessing.

    Parameters
    ----------
    widths : list of int
        Last-dimension size of each model output, in the converter's order.

    Returns
    -------
    dict mapping "probs" / "embedding" / "fall" to an index into `widths`.
    """
    wanted: dict[str, int] = {"probs": n_classes, "embedding": embed_dim}
    if with_fall_head:
        wanted["fall"] = 1

    dupes = [w for w in set(wanted.values()) if list(wanted.values()).count(w) > 1]
    if dupes:
        raise ValueError(
            f"two heads share a width {dupes}, so outputs cannot be told apart. "
            f"Change embed_dim so it differs from n_classes "
            f"(n_classes={n_classes}, embed_dim={embed_dim})."
        )
    if len(widths) != len(wanted):
        raise ValueError(
            f"model has {len(widths)} outputs {widths} but {len(wanted)} were "
            f"expected {wanted}. Was the model built with a different "
            f"with_fall_head setting than the one passed here?"
        )

    roles: dict[str, int] = {}
    remaining = list(enumerate(widths))
    for role, width in wanted.items():
        matches = [i for i, w in remaining if w == width]
        if not matches:
            raise ValueError(
                f"no output of width {width} for role {role!r}; model outputs "
                f"are {widths}. Check n_classes and embed_dim against the model."
            )
        if len(matches) > 1:
            raise ValueError(
                f"{len(matches)} outputs of width {width} both match role "
                f"{role!r}; cannot disambiguate. Model outputs are {widths}."
            )
        roles[role] = matches[0]
        remaining = [(i, w) for i, w in remaining if i != matches[0]]

    return roles
