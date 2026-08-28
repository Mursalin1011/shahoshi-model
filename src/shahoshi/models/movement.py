"""The movement branch: one trunk, two or three heads, one TFLite file.

Two deliberate departures from the pre-refactor notebook.

**No dilated convolutions.** Dilation gave a wide receptive field cheaply, but
TFLite lowers a dilated depthwise convolution into
``SPACE_TO_BATCH_ND -> DEPTHWISE_CONV_2D -> BATCH_TO_SPACE_ND``. Those two extra
ops have to be registered in the firmware's resolver, and both of them compute
nothing: they only rearrange memory, at the cost of a copy over the whole
tensor plus a scratch buffer in an arena that is already the scarce resource.
Replacing dilation with one more stride-2 pooling stage buys the same global
context -- global average pooling sees the whole sequence regardless -- with an
op set that is all arithmetic. `shahoshi.export.resolver_source` generates the resolver
from the converted model's actual op list, so this can never drift out of sync
with the firmware again.

**One model, not two.** The notebook exported a classifier (46.8 KB) and a
separate embedding model (44.6 KB) that contained a byte-identical copy of the
same trunk: ~45 KB of wasted flash and a second tensor arena, on a part where
both are scarce. A single model with two outputs costs nothing extra.

The fall head is off by default because Stage 0 has no fall data. It exists now
so that adding SisFall is a config change rather than a rebuild, and so the
masked-loss machinery it needs is written and reviewed before it is urgent.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L

from ..datasets.base import N_CHANNELS, N_CLASSES

WIN = 128


def _relu(name: str, bounded: bool):
    """ReLU, optionally capped at 6.

    An unbounded ReLU is fine in float and expensive in int8. TFLite sizes
    each activation scale from the min/max observed during calibration, so a
    long tail spends the int8 range on outliers and squeezes the bulk of the
    distribution into a handful of the 127 levels. Capping at 6 is what
    MobileNet does, for exactly this reason.

    Use quantize.activation_ranges() to see whether it is needed here rather
    than assuming: the column to read is levels_at_p99.
    """
    return L.ReLU(max_value=6.0, name=name) if bounded else L.ReLU(name=name)


def _ds_block(x, filters: int, kernel: int, name: str, bounded: bool = False):
    """Depthwise-separable convolution: depthwise, pointwise, BN and ReLU on each.

    BatchNorm folds into the preceding convolution at conversion time and costs
    nothing on device.
    """
    x = L.DepthwiseConv1D(kernel, padding="same", use_bias=False, name=f"{name}_dw")(x)
    x = L.BatchNormalization(name=f"{name}_bn1")(x)
    x = _relu(f"{name}_r1", bounded)(x)
    x = L.Conv1D(filters, 1, use_bias=False, name=f"{name}_pw")(x)
    x = L.BatchNormalization(name=f"{name}_bn2")(x)
    x = _relu(f"{name}_r2", bounded)(x)
    return x


def build(
    n_classes: int = N_CLASSES,
    win: int = WIN,
    channels: int = N_CHANNELS,
    embed_dim: int = 64,
    width: float = 1.0,
    dropout: float = 0.3,
    with_fall_head: bool = False,
    bounded_relu: bool = False,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the movement model, as a training view and an export view.

    Returns
    -------
    (train_model, export_model)
        Two Keras models over the *same layer objects*, so they share one set of
        weights: training `train_model` is what updates `export_model`.

        `train_model` outputs only the supervised heads --
        ``probs`` (and ``fall`` when enabled). `export_model` additionally
        outputs ``embedding``.

    Why two views rather than one model with three outputs: Keras requires a
    loss for every output of a compiled model, and the embedding has no target.
    Working around that with a zero loss means the converter still sees a
    dangling output during training and the training logs carry a meaningless
    loss term. Two views keeps the training graph honest and the exported graph
    complete.

    Export output order is fixed and load-bearing -- ``probs``, ``embedding``,
    then ``fall`` -- because the firmware indexes outputs positionally and
    `shahoshi.quantize.predict` returns them in that order.
    """
    w = lambda c: max(8, int(c * width))  # noqa: E731

    inp = L.Input((win, channels), name="imu")

    x = L.Conv1D(w(24), 9, strides=2, padding="same", use_bias=False, name="stem")(inp)
    x = L.BatchNormalization(name="stem_bn")(x)
    x = _relu("stem_relu", bounded_relu)(x)               # 64 x 24

    x = _ds_block(x, w(32), 5, "b1", bounded_relu)
    x = L.MaxPooling1D(2, name="p1")(x)                    # 32 x 32
    x = _ds_block(x, w(48), 5, "b2", bounded_relu)
    x = L.MaxPooling1D(2, name="p2")(x)                    # 16 x 48
    x = _ds_block(x, w(64), 5, "b3", bounded_relu)
    x = L.MaxPooling1D(2, name="p3")(x)                    #  8 x 64
    x = _ds_block(x, w(64), 3, "b4", bounded_relu)                       #  8 x 64

    x = L.GlobalAveragePooling1D(name="gap")(x)
    # The embedding is also int8-quantized on export and feeds the Mahalanobis
    # scorer, so its activation range matters for the same reason.
    emb = L.Dense(embed_dim, name="embedding_dense")(x)
    emb = _relu("embedding", bounded_relu)(emb)

    dropped = L.Dropout(dropout, name="drop")(emb)
    probs = L.Dense(n_classes, activation="softmax", name="probs")(dropped)

    train_outputs = [probs]
    export_outputs = [probs, emb]
    if with_fall_head:
        fall = L.Dense(1, activation="sigmoid", name="fall")(dropped)
        train_outputs.append(fall)
        export_outputs.append(fall)

    # A single-element output list would make Keras hand back a list from
    # predict(); unwrap it so the common (no fall head) case behaves like an
    # ordinary single-output model.
    train_model = tf.keras.Model(
        inp, train_outputs if len(train_outputs) > 1 else train_outputs[0],
        name="movement_train",
    )
    export_model = tf.keras.Model(inp, export_outputs, name="movement")
    return train_model, export_model


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def masked_sparse_ce(class_weights=None, ignore_label: int = -1):
    """Sparse categorical cross-entropy that skips `ignore_label` rows.

    Class weighting lives *inside* the loss rather than being passed as Keras's
    `class_weight=`, and that is the fix for a specific bug in the pre-refactor
    baseline. `class_weight=` reweights the training loss but not the validation
    loss, so the two are computed on different scales; `val_loss` then diverged
    from epoch 5 onward while `val_accuracy` kept climbing, and
    `EarlyStopping(monitor="val_loss", restore_best_weights=True)` restored the
    epoch-5 weights. The shipped model was a 5-epoch model.

    Weighting inside the loss makes train and validation loss directly
    comparable again, so `val_loss` is a usable monitor. We still prefer macro-F1
    (see `MacroF1`) because it is the metric we actually care about under class
    imbalance, but the underlying inconsistency is gone either way.
    """
    cw = None if class_weights is None else tf.constant(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
        mask = tf.cast(tf.not_equal(y_true, ignore_label), tf.float32)
        safe = tf.maximum(y_true, 0)

        ce = tf.keras.losses.sparse_categorical_crossentropy(safe, y_pred)
        weight = mask if cw is None else mask * tf.gather(cw, safe)
        # Normalize by summed weight, not by batch size, so a batch that happens
        # to be mostly masked does not contribute a near-zero gradient.
        return tf.reduce_sum(ce * weight) / tf.maximum(tf.reduce_sum(weight), 1e-6)

    loss.__name__ = "masked_sparse_ce"
    return loss


def masked_binary_ce(positive_weight: float = 1.0, ignore_label: int = -1):
    """Binary cross-entropy that skips `ignore_label` rows.

    This is what makes the multi-task design worth the complexity. UCI and
    MotionSense windows carry no information about falls in either direction, and
    SisFall's transition ADLs ("sit down, then stand up") map onto no single
    activity label while being unambiguously not-falls. Masking lets each window
    train exactly the head it has evidence for.
    """
    def loss(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(y_pred, [-1])
        mask = tf.cast(tf.not_equal(y_true, float(ignore_label)), tf.float32)
        safe = tf.clip_by_value(y_true, 0.0, 1.0)

        bce = tf.keras.losses.binary_crossentropy(safe[:, None], y_pred[:, None])
        weight = mask * (1.0 + (positive_weight - 1.0) * safe)
        return tf.reduce_sum(bce * weight) / tf.maximum(tf.reduce_sum(weight), 1e-6)

    loss.__name__ = "masked_binary_ce"
    return loss


def class_weights_from(labels, n_classes: int = N_CLASSES, ignore_label: int = -1):
    """Inverse-frequency class weights over labelled windows only.

    Absent classes get weight 0 rather than infinity, and the weights are
    normalized to mean 1 over present classes so the loss scale does not shift
    when the class mix changes between experiments.
    """
    import numpy as np

    labels = np.asarray(labels)
    labels = labels[labels != ignore_label]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)

    present = counts > 0
    if not present.any():
        raise ValueError("no labelled windows to weight")

    w = np.zeros(n_classes, dtype=np.float64)
    w[present] = counts[present].sum() / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    return w.astype(np.float32)


def compile_model(
    train_model: tf.keras.Model,
    lr: float = 1e-3,
    class_weights=None,
    with_fall_head: bool = False,
    fall_positive_weight: float = 1.0,
) -> tf.keras.Model:
    """Wire the masked losses onto the training view.

    Kept here rather than in the notebook so the loss configuration -- the part
    that silently changed the meaning of the pre-refactor baseline's early
    stopping -- lives in one reviewed place.
    """
    act_loss = masked_sparse_ce(class_weights)
    if with_fall_head:
        losses = [act_loss, masked_binary_ce(fall_positive_weight)]
        # The fall head is the one the alert path consumes, so it is weighted up
        # relative to the auxiliary activity task.
        weights = [1.0, 2.0]
    else:
        losses, weights = act_loss, None

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=losses,
        loss_weights=weights,
    )
    return train_model


# ---------------------------------------------------------------------------
# metrics / callbacks
# ---------------------------------------------------------------------------

class MacroF1(tf.keras.callbacks.Callback):
    """Compute macro-F1 on a validation set and put it in `logs`.

    Must be listed *before* EarlyStopping and ReduceLROnPlateau in the callbacks
    list, because those read `logs` and Keras runs callbacks in order.

    Macro-F1 rather than accuracy: accuracy is dominated by the majority classes,
    and the classes at risk here (the rare ones, and later the falls) are exactly
    the ones accuracy is insensitive to.
    """

    def __init__(
        self,
        x,
        y,
        n_classes: int = N_CLASSES,
        ignore_label: int = -1,
        name: str = "val_macro_f1",
        batch_size: int = 256,
    ):
        super().__init__()
        self.x, self.y = x, y
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.name = name
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        import numpy as np

        logs = logs if logs is not None else {}
        out = self.model.predict(self.x, batch_size=self.batch_size, verbose=0)
        probs = out[0] if isinstance(out, (list, tuple)) else out
        pred = probs.argmax(axis=1)

        keep = self.y != self.ignore_label
        y_true, y_pred = self.y[keep], pred[keep]

        f1s = []
        for c in range(self.n_classes):
            tp = float(((y_pred == c) & (y_true == c)).sum())
            fp = float(((y_pred == c) & (y_true != c)).sum())
            fn = float(((y_pred != c) & (y_true == c)).sum())
            if tp + fn == 0:
                continue  # class absent from this split: no opinion, not a zero
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else 2 * tp / denom)

        logs[self.name] = float(np.mean(f1s)) if f1s else 0.0


def default_callbacks(
    val_x,
    val_y_act,
    n_classes: int = N_CLASSES,
    patience: int = 20,
    monitor: str = "val_macro_f1",
) -> list:
    """MacroF1 first, then the callbacks that read its output.

    `restore_best_weights=True` with a maximized metric, so the weights that
    survive are the ones that scored best on the metric we care about -- not,
    as before, the ones that scored best on an incomparable validation loss.
    """
    return [
        MacroF1(val_x, val_y_act, n_classes=n_classes),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.5, patience=max(3, patience // 3),
            min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
    ]
