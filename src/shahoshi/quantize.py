"""Full-integer quantization and int8 inference. Requires TensorFlow.

int8 in, int8 out, weights and activations both, so no float kernels are linked
into the firmware at all.

The representative dataset determines activation ranges and is the usual cause
of a bad quantization result. Draw it from *training* data, and draw it broadly:
`representative_dataset` stratifies by class so a few hundred windows cover every
class rather than several hundred windows of whichever class happens to be most
common. In the pre-refactor baseline int8 cost 2.8 accuracy points, which is
above the 0-2 that a well-calibrated conversion should cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from .heads import assign_roles


def representative_dataset(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n: int = 512,
    n_classes: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Pick `n` diverse windows for activation-range calibration.

    Stratified by `y` when given: a representative set drawn uniformly at random
    from an imbalanced dataset under-represents the rare classes, and those are
    precisely the ones whose activations get clipped.
    """
    X = np.asarray(X)
    rng = np.random.default_rng(seed)

    if y is None:
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx]

    y = np.asarray(y)
    classes = [c for c in np.unique(y) if c >= 0]
    if n_classes is not None:
        classes = [c for c in classes if c < n_classes]
    if not classes:
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx]

    per_class = max(1, n // len(classes))
    picks = []
    for c in classes:
        pool = np.where(y == c)[0]
        picks.append(rng.choice(pool, min(per_class, len(pool)), replace=False))
    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return X[idx[:n]]


def to_int8(
    model: tf.keras.Model,
    rep_x: np.ndarray,
    path: str | Path | None = None,
    output_int8: bool = True,
) -> bytes:
    """Convert a Keras model to a fully int8-quantized TFLite blob.

    Falls back to the SavedModel route when `from_keras_model` refuses, which
    Keras 3 (TF >= 2.16) frequently does. The SavedModel path is the supported
    route and produces an identical graph, so the fallback is not a compromise.

    Parameters
    ----------
    output_int8 : bool
        Quantize the *outputs* to int8 as well as the weights and activations.
        Turning this off leaves a trailing DEQUANTIZE so outputs come back as
        float32, while every kernel stays integer. That matters more than it
        sounds: an int8 softmax output carries about 1/256 of resolution, and
        when two classes are close the argmax can flip on a rounding boundary.
        `diagnose` measures whether that is happening here.
    """
    rep_x = np.asarray(rep_x, dtype=np.float32)

    def configure(conv: tf.lite.TFLiteConverter) -> bytes:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8 if output_int8 else tf.float32
        conv.representative_dataset = lambda: ([rep_x[i : i + 1]] for i in range(len(rep_x)))
        return conv.convert()

    try:
        blob = configure(tf.lite.TFLiteConverter.from_keras_model(model))
    except Exception as exc:  # noqa: BLE001 - the converter raises many types
        print(f"  from_keras_model failed ({type(exc).__name__}), using SavedModel route")
        import shutil
        import tempfile

        sm = Path(tempfile.gettempdir()) / f"sm_{model.name}"
        shutil.rmtree(sm, ignore_errors=True)
        model.export(str(sm))
        blob = configure(tf.lite.TFLiteConverter.from_saved_model(str(sm)))

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(blob)
        print(f"  {Path(path).name:<28s} {len(blob) / 1024:7.1f} KB")
    return blob


def interpreter(blob: bytes) -> tf.lite.Interpreter:
    itp = tf.lite.Interpreter(model_content=blob)
    itp.allocate_tensors()
    return itp


def output_roles(
    blob: bytes,
    n_classes: int,
    embed_dim: int,
    with_fall_head: bool = False,
) -> dict[str, int]:
    """Resolve the converted model's outputs to roles, verified by width."""
    details = interpreter(blob).get_output_details()
    widths = [int(d["shape"][-1]) for d in details]
    return assign_roles(widths, n_classes, embed_dim, with_fall_head)


def output_widths(blob: bytes) -> list[int]:
    """Last-dimension size of each output, in the converter's order.

    Worth printing after every conversion: it is the fact that reveals whether
    the converter reordered the heads.
    """
    return [int(d["shape"][-1]) for d in interpreter(blob).get_output_details()]


def predict(blob: bytes, X: np.ndarray, batch_note: bool = False) -> list[np.ndarray]:
    """Run the int8 model over `X`, returning dequantized float outputs.

    Quantizes the input and dequantizes each output using the scale and
    zero-point baked into the model, which is exactly what the firmware does --
    so a discrepancy between this and the device points at the firmware, not at
    the model.

    Returns
    -------
    list of arrays, one per model output, in the model's output order.
    """
    itp = interpreter(blob)
    in_d = itp.get_input_details()[0]
    out_d = itp.get_output_details()

    si, zi = in_d["quantization"]
    if si == 0:
        raise ValueError(
            "input tensor is not quantized -- to_int8() did not produce a "
            "fully-integer model, so the firmware would need float kernels"
        )

    X = np.asarray(X, dtype=np.float32)
    collected: list[list[np.ndarray]] = [[] for _ in out_d]

    for i, x in enumerate(X):
        q = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)
        itp.set_tensor(in_d["index"], q[None])
        itp.invoke()
        for k, d in enumerate(out_d):
            so, zo = d["quantization"]
            raw = itp.get_tensor(d["index"])[0].astype(np.float32)
            collected[k].append((raw - zo) * so if so else raw)
        if batch_note and i and i % 5000 == 0:
            print(f"    {i:,}/{len(X):,}")

    return [np.array(c) for c in collected]


def predict_named(
    blob: bytes,
    X: np.ndarray,
    n_classes: int,
    embed_dim: int,
    with_fall_head: bool = False,
    batch_note: bool = False,
) -> dict[str, np.ndarray]:
    """Run the int8 model and return outputs keyed by role, resolved by width.

    Prefer this over `predict` everywhere. `predict` returns the converter's
    positional order, which is not the order the Keras model declared.
    """
    roles = output_roles(blob, n_classes, embed_dim, with_fall_head)
    outs = predict(blob, X, batch_note=batch_note)
    return {role: outs[idx] for role, idx in roles.items()}


def ops_used(blob: bytes) -> list[str]:
    """Builtin operators the converted model actually uses.

    Read from the model rather than assumed. The pre-refactor firmware sketch
    registered eight ops by hand and the converted model used thirteen,
    including two the sketch had never heard of, which would have failed at
    AllocateTensors() on the device.
    """
    itp = interpreter(blob)
    # _get_ops_details is private but is the only way to enumerate ops from a
    # blob; guard so a TF upgrade degrades to a clear error, not a wrong answer.
    if not hasattr(itp, "_get_ops_details"):
        raise RuntimeError(
            "this TensorFlow build does not expose Interpreter._get_ops_details; "
            "read the op list with `flatc` against schema.fbs instead of guessing"
        )
    return sorted({d["op_name"] for d in itp._get_ops_details()})


def arena_estimate(blob: bytes) -> dict[str, float]:
    """Rough arena figures. A lower bound -- confirm on device.

    TFLM's real figure differs because of alignment, scratch buffers and
    op-specific temporaries. Take `arena_used_bytes()` from the device after the
    first Invoke() and size the arena to that times about 1.1.
    """
    itp = interpreter(blob)
    biggest = 0
    total = 0
    for d in itp.get_tensor_details():
        if d["shape"].size:
            nbytes = int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
            biggest = max(biggest, nbytes)
            total += nbytes
    return {
        "flash_kb": len(blob) / 1024,
        "largest_tensor_kb": biggest / 1024,
        "sum_of_tensors_kb": total / 1024,
    }


def accuracy_delta(
    float_probs: np.ndarray, int8_probs: np.ndarray, y_true: np.ndarray
) -> dict[str, float]:
    """Compare float and int8 predictions on the same windows.

    Always measure after quantizing. A drop beyond about 2 points means the
    representative dataset was not diverse enough -- not that int8 is unsuitable.
    `agreement` isolates that: low agreement with high accuracy on both means the
    two models disagree on genuinely marginal windows, which is benign.
    """
    keep = np.asarray(y_true) >= 0
    y = np.asarray(y_true)[keep]
    pf = np.asarray(float_probs)[keep].argmax(1)
    pq = np.asarray(int8_probs)[keep].argmax(1)
    return {
        "float_accuracy": float((pf == y).mean()),
        "int8_accuracy": float((pq == y).mean()),
        "delta": float((pq == y).mean() - (pf == y).mean()),
        "agreement": float((pf == pq).mean()),
    }


# ---------------------------------------------------------------------------
# diagnosing an int8 accuracy drop
# ---------------------------------------------------------------------------

def diagnose(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    embed_dim: int,
    float_probs: np.ndarray,
    rep_sizes: tuple[int, ...] = (256, 512, 1024, 2048),
    seed: int = 42,
) -> dict:
    """Locate the cause of an int8 accuracy drop by measurement, not guesswork.

    A post-training int8 conversion of a depthwise-separable network should cost
    0-2 accuracy points. A larger drop has three plausible causes, and they call
    for different fixes, so guessing is expensive:

    1. **Calibration** -- the representative set does not cover the activation
       ranges the test data produces, so activations clip. Fix: more and more
       diverse representative windows. Detected by the drop shrinking as
       `rep_sizes` grows.
    2. **Output resolution** -- an int8 softmax has ~1/256 resolution, so the
       argmax flips whenever the top two classes are within one quantization
       step. Fix: float32 outputs, which keeps every kernel integer. Detected by
       the float-output variant recovering most of the gap.
    3. **Weight quantization** -- per-channel weight ranges in the depthwise
       layers are too wide for int8 to represent. Fix: quantization-aware
       training, or architectural changes. Indicated when neither of the above
       moves the number.

    Returns
    -------
    dict with a `sweep` list (one entry per representative-set size), a
    `float_output` entry, `per_class` recall deltas, and a `verdict` string.
    """
    float_pred = np.asarray(float_probs).argmax(1)
    float_acc = float((float_pred == y_test).mean())
    out: dict = {"float_accuracy": float_acc, "sweep": [], "per_class": {}}

    print(f"float32 accuracy: {float_acc:.4f}\n")
    print("1. calibration -- does a bigger representative set help?")
    print(f"   {'rep windows':>12s} {'int8 acc':>10s} {'delta':>9s} {'agreement':>10s}")

    best = None
    for n in rep_sizes:
        if n > len(X_train):
            continue
        rep = representative_dataset(X_train, y_train, n=n, n_classes=n_classes, seed=seed)
        blob = to_int8(model, rep)
        probs = predict_named(blob, X_test, n_classes, embed_dim)["probs"]
        pred = probs.argmax(1)
        row = {
            "rep_size": int(len(rep)),
            "accuracy": float((pred == y_test).mean()),
            "delta": float((pred == y_test).mean() - float_acc),
            "agreement": float((pred == float_pred).mean()),
            "blob": blob,
        }
        out["sweep"].append(row)
        print(f"   {row['rep_size']:>12,} {row['accuracy']:>10.4f} "
              f"{row['delta']:>+9.4f} {row['agreement']:>10.4f}")
        if best is None or row["accuracy"] > best["accuracy"]:
            best = row

    print("\n2. output resolution -- do float32 outputs recover the gap?")
    rep = representative_dataset(X_train, y_train, n=max(rep_sizes),
                                 n_classes=n_classes, seed=seed)
    blob_f = to_int8(model, rep, output_int8=False)
    probs_fo = predict_named(blob_f, X_test, n_classes, embed_dim)["probs"]
    pred_fo = probs_fo.argmax(1)
    out["float_output"] = {
        "accuracy": float((pred_fo == y_test).mean()),
        "delta": float((pred_fo == y_test).mean() - float_acc),
        "agreement": float((pred_fo == float_pred).mean()),
        "blob": blob_f,
    }
    print(f"   int8 weights + float32 outputs: {out['float_output']['accuracy']:.4f} "
          f"({out['float_output']['delta']:+.4f})")

    # Where the loss lands, per class. Diffuse loss points at weights; loss
    # concentrated in one class points at that class sitting near a boundary.
    print("\n3. per-class recall, float vs best int8")
    ref = best if best is not None else out["float_output"]
    best_pred = (predict_named(ref["blob"], X_test, n_classes, embed_dim)["probs"].argmax(1)
                 if "blob" in ref else pred_fo)
    for c in range(n_classes):
        msk = np.asarray(y_test) == c
        if not msk.any():
            continue
        rf = float((float_pred[msk] == c).mean())
        rq = float((best_pred[msk] == c).mean())
        out["per_class"][int(c)] = {"float": rf, "int8": rq, "delta": rq - rf}
        print(f"   class {c}: float {rf:.3f}  int8 {rq:.3f}  ({rq - rf:+.3f})  n={int(msk.sum())}")

    # Verdict, in the order the fixes should be tried.
    sweep_gain = (out["sweep"][-1]["accuracy"] - out["sweep"][0]["accuracy"]
                  if len(out["sweep"]) > 1 else 0.0)
    fo_gain = out["float_output"]["accuracy"] - (best["accuracy"] if best else 0.0)

    if out["float_output"]["delta"] > -0.02:
        verdict = ("output resolution. int8 weights with float32 outputs is within 2 "
                   "points, so the kernels are fine and the int8 softmax was losing "
                   "the argmax. Ship output_int8=False -- every kernel stays integer "
                   "and only a trailing DEQUANTIZE is added.")
    elif sweep_gain > 0.02:
        verdict = ("calibration. The drop shrinks as the representative set grows, so "
                   "raise quantize.n_representative and re-measure before anything else.")
    elif fo_gain > 0.02:
        verdict = ("mostly output resolution, with some calibration loss left over. "
                   "Use float32 outputs and the largest representative set.")
    else:
        verdict = ("weight quantization. Neither calibration nor output dtype moves it, "
                   "so the depthwise per-channel weight ranges are the problem. That "
                   "needs quantization-aware training or an architecture change -- not "
                   "more calibration data.")
    out["verdict"] = verdict
    print(f"\nverdict: {verdict}")
    return out


def set_determinism(seed: int = 42) -> None:
    """Make a training run reproducible, including on GPU.

    `tf.random.set_seed` alone does not do this: cuDNN picks nondeterministic
    kernels for convolution backprop, so two runs of identical code diverge. In
    this project that showed up as held-out accuracy of 0.9024 and 0.8758 from
    the same commit -- a 2.7-point spread that could easily be mistaken for a
    real effect when comparing two experiments.

    Costs some training speed. Worth it: without it, no ablation in this project
    is interpretable.
    """
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
        print(f"  op determinism enabled, seed {seed}")
    except Exception as exc:  # noqa: BLE001 - availability varies by TF build
        print(f"  op determinism unavailable ({type(exc).__name__}: {exc}); "
              f"seeds set, but GPU runs may still vary")
