"""Keras model definitions. Importing this module requires TensorFlow.

Kept in a subpackage so that everything else in `shahoshi` -- loaders, signal
conditioning, splits, augmentation, scoring, export -- stays importable and
unit-testable without a TensorFlow install.
"""

from . import movement

__all__ = ["movement"]
