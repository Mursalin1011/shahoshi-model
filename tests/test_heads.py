"""Tests for output-role resolution.

`assign_roles` is factored out of the TensorFlow code precisely so this can be
tested without TF, because getting it wrong is silent: a float accuracy of 0.9024
alongside an int8 accuracy of 0.0196, from a model that was entirely fine and an
indexing assumption that was not.
"""

import pytest

from shahoshi.heads import assign_roles

N_CLASSES = 6
EMBED_DIM = 64


class TestAssignRoles:
    def test_declared_order(self):
        """Keras order: probs then embedding."""
        assert assign_roles([N_CLASSES, EMBED_DIM], N_CLASSES, EMBED_DIM) == {
            "probs": 0,
            "embedding": 1,
        }

    def test_converter_reordered_the_heads(self):
        """The bug. TFLite emitted the embedding first, so reading position 0 as
        the softmax gave argmax over 64 values against labels in 0..5."""
        assert assign_roles([EMBED_DIM, N_CLASSES], N_CLASSES, EMBED_DIM) == {
            "probs": 1,
            "embedding": 0,
        }

    def test_resolution_is_order_independent(self):
        """Whatever order the converter picks, roles land on the right widths."""
        for widths, expect in (
            ([6, 64], {"probs": 0, "embedding": 1}),
            ([64, 6], {"probs": 1, "embedding": 0}),
        ):
            roles = assign_roles(widths, N_CLASSES, EMBED_DIM)
            assert widths[roles["probs"]] == N_CLASSES
            assert widths[roles["embedding"]] == EMBED_DIM
            assert roles == expect

    @pytest.mark.parametrize(
        "widths",
        [[6, 64, 1], [64, 6, 1], [1, 6, 64], [64, 1, 6], [1, 64, 6], [6, 1, 64]],
    )
    def test_three_heads_in_any_order(self, widths):
        roles = assign_roles(widths, N_CLASSES, EMBED_DIM, with_fall_head=True)
        assert widths[roles["probs"]] == N_CLASSES
        assert widths[roles["embedding"]] == EMBED_DIM
        assert widths[roles["fall"]] == 1
        assert len(set(roles.values())) == 3       # no index reused

    def test_fall_head_absent_when_not_requested(self):
        roles = assign_roles([6, 64], N_CLASSES, EMBED_DIM, with_fall_head=False)
        assert "fall" not in roles

    def test_rejects_ambiguous_equal_widths(self):
        """If the embedding were as wide as the class count, width could not tell
        the heads apart -- so it refuses instead of picking one."""
        with pytest.raises(ValueError, match="share a width"):
            assign_roles([6, 6], n_classes=6, embed_dim=6)

    def test_rejects_wrong_output_count(self):
        with pytest.raises(ValueError, match="outputs"):
            assign_roles([6], N_CLASSES, EMBED_DIM)
        with pytest.raises(ValueError, match="with_fall_head"):
            assign_roles([6, 64, 1], N_CLASSES, EMBED_DIM, with_fall_head=False)

    def test_rejects_missing_expected_width(self):
        """A model built with embed_dim=32 read as if it were 64 must fail loudly
        rather than assign the softmax to the embedding role."""
        with pytest.raises(ValueError, match="no output of width 64"):
            assign_roles([6, 32], N_CLASSES, EMBED_DIM)

    def test_error_names_the_actual_widths(self):
        """The message has to carry the evidence, or diagnosing this costs a run."""
        with pytest.raises(ValueError) as exc:
            assign_roles([6, 32], N_CLASSES, EMBED_DIM)
        assert "[6, 32]" in str(exc.value)

    def test_a_width_one_class_count_still_resolves(self):
        """Binary-only model: n_classes=1 collides with a fall head, so a
        two-output build must still be unambiguous."""
        assert assign_roles([64, 1], n_classes=1, embed_dim=64) == {
            "probs": 1,
            "embedding": 0,
        }
