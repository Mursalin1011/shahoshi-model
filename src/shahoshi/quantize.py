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


def _convert(model: tf.keras.Model, configure) -> bytes:
    """Run `configure` against a converter, falling back to the SavedModel route.

    Keras 3 (TF >= 2.16) frequently refuses `from_keras_model`. Exporting a
    SavedModel first is the supported route and produces an identical graph, so
    the fallback is not a compromise. Shared by every conversion variant.
    """
    try:
        return configure(tf.lite.TFLiteConverter.from_keras_model(model))
    except Exception as exc:  # noqa: BLE001 - the converter raises many types
        print(f"  from_keras_model failed ({type(exc).__name__}), using SavedModel route")
        import shutil
        import tempfile

        sm = Path(tempfile.gettempdir()) / f"sm_{model.name}"
        shutil.rmtree(sm, ignore_errors=True)
        model.export(str(sm))
        return configure(tf.lite.TFLiteConverter.from_saved_model(str(sm)))


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

    blob = _convert(model, configure)

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


def predict(
    blob: bytes,
    X: np.ndarray,
    batch_note: bool = False,
    require_int8: bool = True,
) -> list[np.ndarray]:
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
    int8_in = np.issubdtype(in_d["dtype"], np.integer)
    if require_int8 and (si == 0 or not int8_in):
        raise ValueError(
            "input tensor is not quantized -- to_int8() did not produce a "
            "fully-integer model, so the firmware would need float kernels. "
            "Pass require_int8=False if this is a diagnostic variant."
        )

    X = np.asarray(X, dtype=np.float32)
    collected: list[list[np.ndarray]] = [[] for _ in out_d]

    for i, x in enumerate(X):
        if int8_in:
            q = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)
        else:
            q = x.astype(in_d["dtype"])
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
    require_int8: bool = True,
) -> dict[str, np.ndarray]:
    """Run the int8 model and return outputs keyed by role, resolved by width.

    Prefer this over `predict` everywhere. `predict` returns the converter's
    positional order, which is not the order the Keras model declared.
    """
    roles = output_roles(blob, n_classes, embed_dim, with_fall_head)
    outs = predict(blob, X, batch_note=batch_note, require_int8=require_int8)
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


def activation_ranges(
    model: tf.keras.Model,
    X: np.ndarray,
    sample: int = 1024,
    seed: int = 42,
) -> list[dict]:
    """Per-layer activation percentiles, to expose outlier tails.

    TFLite calibrates activation scales from the min and max seen over the
    representative set. So a layer whose values mostly live in [0, 2] but which
    occasionally spikes to 40 gets a scale sized for 40, and the ordinary values
    are then squeezed into the bottom 5% of the int8 range -- roughly 6 of the
    available 127 levels. That is how a conversion loses 8-12 accuracy points
    while every individual op is behaving correctly.

    It also explains a counter-intuitive observation: enlarging the
    representative set made this model's int8 accuracy *worse* (-0.081 at 512
    windows, -0.123 at 2048). More samples means more opportunities to observe an
    extreme value, a wider calibrated range, and coarser quantization for
    everything else. Under this failure mode, more calibration data actively
    hurts.

    The column to read is `levels_at_p99`: how many of the 127 int8 levels the
    bulk of the distribution actually occupies. Single digits is a problem.
    Unbounded ReLU is the usual cause, and a bounded activation
    (`model.bounded_relu`) is the usual fix.
    """
    acts = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.ReLU)
        or layer.__class__.__name__ in ("Activation", "Dense")
    ]
    if not acts:
        raise ValueError("no activation layers found to probe")

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)

    probe = tf.keras.Model(model.inputs, [layer.output for layer in acts])
    outs = probe.predict(X[idx], batch_size=128, verbose=0)
    if not isinstance(outs, list):
        outs = [outs]

    rows = []
    print(f"{'layer':<18s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} "
          f"{'max/p99.9':>10s} {'levels_at_p99':>14s}")
    for layer, out in zip(acts, outs):
        v = np.abs(np.asarray(out, dtype=np.float64)).ravel()
        p50, p99, p999 = (float(np.percentile(v, q)) for q in (50, 99, 99.9))
        mx = float(v.max())
        ratio = mx / max(p999, 1e-12)
        levels = 127.0 * p99 / max(mx, 1e-12)
        rows.append({
            "layer": layer.name, "p50": p50, "p99": p99, "p99_9": p999,
            "max": mx, "max_over_p999": ratio, "levels_at_p99": levels,
        })
        flag = "  <-- outlier tail" if levels < 32 else ""
        print(f"{layer.name:<18s} {p50:>9.3f} {p99:>9.3f} {p999:>9.3f} {mx:>9.3f} "
              f"{ratio:>10.2f} {levels:>14.1f}{flag}")

    worst = min(rows, key=lambda r: r["levels_at_p99"])
    print(f"\nworst layer: {worst['layer']} -- the bulk of its distribution occupies "
          f"{worst['levels_at_p99']:.1f} of 127 int8 levels")
    if worst["levels_at_p99"] < 32:
        print("This is enough to explain a large int8 drop on its own. Set\n"
              "model.bounded_relu: true to cap activations at 6 and retrain; that is\n"
              "standard practice for int8 targets and costs little float accuracy.")
    else:
        print("No severe outlier tail here, so look at weight quantization instead "
              "(quantize.diagnose reports which).")
    return rows


# ---------------------------------------------------------------------------
# separating weight quantization from activation quantization
# ---------------------------------------------------------------------------

def to_variant(
    model: tf.keras.Model,
    mode: str,
    rep_x: np.ndarray | None = None,
) -> bytes:
    """Convert to one of several quantization variants, for attribution.

    The variants exist because "int8 costs 12 points" is not actionable. Weights
    and activations are quantized by different mechanisms, they fail for
    different reasons, and they have different fixes -- bounded activations or a
    different calibration for one, quantization-aware training for the other.
    Comparing these variants attributes the loss instead of inferring it.

    Modes
    -----
    ``full_int8``
        int8 weights, int8 activations, int8 in/out. The deployment target.
    ``int8_out_float``
        int8 weights and activations, float32 in/out. Isolates output resolution.
    ``dynamic_range``
        **The discriminator.** int8 weights, but activations stay float32 at
        runtime -- no representative dataset is used at all. If this holds
        accuracy, the weights quantize fine and activation quantization is
        responsible. If it drops as far as full int8, the weights are the problem.
    ``float16``
        16-bit weights, float activations. A near-lossless floor; if even this
        drops, something is wrong beyond quantization.
    """
    modes = ("full_int8", "int8_out_float", "dynamic_range", "float16")
    if mode not in modes:
        raise ValueError(f"mode must be one of {modes}; got {mode!r}")
    if mode in ("full_int8", "int8_out_float") and rep_x is None:
        raise ValueError(f"mode {mode!r} needs a representative dataset")

    def configure(conv: tf.lite.TFLiteConverter) -> bytes:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if mode == "float16":
            conv.target_spec.supported_types = [tf.float16]
        elif mode == "dynamic_range":
            pass  # weights-only int8: no representative dataset, float activations
        else:
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8 if mode == "full_int8" else tf.float32
            rep = np.asarray(rep_x, dtype=np.float32)
            conv.representative_dataset = lambda: ([rep[i : i + 1]] for i in range(len(rep)))
        return conv.convert()

    return _convert(model, configure)


def attribute_loss(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    embed_dim: int,
    float_probs: np.ndarray,
    n_representative: int = 512,
    seed: int = 42,
) -> dict:
    """Attribute an int8 accuracy drop to weights or to activations.

    Runs the variants in `to_variant` and reports each one's accuracy against the
    float model. The comparison that matters is `dynamic_range` versus
    `full_int8`: they use identically quantized weights and differ only in
    whether activations are quantized, so the gap between them is the cost of
    activation quantization and the gap from float to `dynamic_range` is the cost
    of weight quantization.

    This replaces a verdict reached by elimination. Ruling out calibration and
    output dtype leaves weights *and* activations both in play, and prescribing
    quantization-aware training for what turns out to be an activation-range
    problem would be an expensive detour.
    """
    y_test = np.asarray(y_test)
    float_pred = np.asarray(float_probs).argmax(1)
    float_acc = float((float_pred == y_test).mean())

    rep = representative_dataset(X_train, y_train, n=n_representative,
                                 n_classes=n_classes, seed=seed)

    results: dict[str, dict] = {}
    plan = [
        ("float16", None),
        ("dynamic_range", None),
        ("int8_out_float", rep),
        ("full_int8", rep),
    ]
    print(f"float32 accuracy: {float_acc:.4f}\n")
    print(f"{'variant':<16s} {'weights':<9s} {'activations':<12s} "
          f"{'accuracy':>9s} {'delta':>9s} {'agreement':>10s}")

    describe = {
        "float16": ("float16", "float32"),
        "dynamic_range": ("int8", "float32"),
        "int8_out_float": ("int8", "int8"),
        "full_int8": ("int8", "int8"),
    }
    for mode, r in plan:
        try:
            blob = to_variant(model, mode, r)
            probs = predict_named(blob, X_test, n_classes, embed_dim,
                                  require_int8=False)["probs"]
        except Exception as exc:  # noqa: BLE001 - converter raises many types
            print(f"{mode:<16s} failed: {type(exc).__name__}: {exc}")
            continue
        pred = probs.argmax(1)
        acc = float((pred == y_test).mean())
        results[mode] = {
            "accuracy": acc,
            "delta": acc - float_acc,
            "agreement": float((pred == float_pred).mean()),
            "size_kb": len(blob) / 1024,
        }
        w, a = describe[mode]
        print(f"{mode:<16s} {w:<9s} {a:<12s} {acc:>9.4f} "
              f"{acc - float_acc:>+9.4f} {results[mode]['agreement']:>10.4f}")

    out: dict = {"float_accuracy": float_acc, "variants": results}

    dr = results.get("dynamic_range")
    fi = results.get("full_int8")
    if dr is None or fi is None:
        out["verdict"] = "inconclusive: a required variant failed to convert"
        print(f"\nverdict: {out['verdict']}")
        return out

    weight_cost = float_acc - dr["accuracy"]        # float -> int8 weights
    activation_cost = dr["accuracy"] - fi["accuracy"]  # + int8 activations
    out["weight_cost"] = weight_cost
    out["activation_cost"] = activation_cost

    print(f"\ncost of quantizing weights     : {weight_cost:+.4f}")
    print(f"cost of quantizing activations : {activation_cost:+.4f}")

    if activation_cost > 2 * max(weight_cost, 1e-6) and activation_cost > 0.02:
        out["verdict"] = (
            f"activation quantization, not weights. int8 weights alone cost "
            f"{weight_cost:.4f} while adding int8 activations costs "
            f"{activation_cost:.4f}. Fix the activation ranges -- set "
            f"model.bounded_relu: true and retrain -- rather than reaching for "
            f"quantization-aware training."
        )
    elif weight_cost > 2 * max(activation_cost, 1e-6) and weight_cost > 0.02:
        out["verdict"] = (
            f"weight quantization. int8 weights alone already cost "
            f"{weight_cost:.4f}, so calibration and activation ranges are not the "
            f"lever. This needs quantization-aware training, a wider model, or "
            f"accepting the loss."
        )
    elif weight_cost + activation_cost <= 0.02:
        out["verdict"] = "no meaningful int8 loss to attribute"
    else:
        out["verdict"] = (
            f"both, comparably: weights {weight_cost:.4f}, activations "
            f"{activation_cost:.4f}. Try bounded activations first since it is "
            f"far cheaper, then quantization-aware training if the remainder "
            f"still matters."
        )
    print(f"\nverdict: {out['verdict']}")
    return out


def weight_ranges(model: tf.keras.Model) -> list[dict]:
    """Per-output-channel weight range spread for each convolution.

    TFLite quantizes convolution weights per output channel, so wide variation
    *between* channels is handled. What it cannot handle is a wide range *within*
    one channel. This reports both, so a claim about "depthwise per-channel
    weight ranges" can be checked rather than asserted.

    Note these are the pre-fold weights. BatchNorm folds a per-channel
    gamma/sqrt(var+eps) multiplier into the preceding convolution at conversion
    time, which can widen the spread considerably, so treat this as indicative.
    """
    rows = []
    print(f"{'layer':<16s} {'kind':<10s} {'channels':>9s} {'max|w|':>10s} "
          f"{'chan spread':>12s} {'worst in-chan':>14s}")
    for layer in model.layers:
        kind = layer.__class__.__name__
        if kind not in ("Conv1D", "DepthwiseConv1D", "Dense"):
            continue
        weights = layer.get_weights()
        if not weights:
            continue
        w = np.asarray(weights[0], dtype=np.float64)
        # Last axis is the output channel for Conv1D/Dense; for DepthwiseConv1D
        # the channel axis is the second-to-last, so flatten all but that.
        axis = -2 if kind == "DepthwiseConv1D" else -1
        w = np.moveaxis(w, axis, 0).reshape(w.shape[axis], -1)

        per_chan_max = np.abs(w).max(axis=1)
        per_chan_max = np.maximum(per_chan_max, 1e-12)
        chan_spread = float(per_chan_max.max() / per_chan_max.min())

        # Within a channel: how much of that channel's int8 range the typical
        # weight uses. Small means the channel is dominated by one large tap.
        p99 = np.percentile(np.abs(w), 99, axis=1)
        worst_in_chan = float((127.0 * p99 / per_chan_max).min())

        rows.append({
            "layer": layer.name, "kind": kind, "channels": int(w.shape[0]),
            "max_abs": float(np.abs(w).max()), "channel_spread": chan_spread,
            "worst_in_channel_levels": worst_in_chan,
        })
        print(f"{layer.name:<16s} {kind:<10s} {w.shape[0]:>9d} "
              f"{np.abs(w).max():>10.4f} {chan_spread:>12.1f}x "
              f"{worst_in_chan:>14.1f}")

    if rows:
        worst = min(rows, key=lambda r: r["worst_in_channel_levels"])
        print(f"\nworst in-channel utilization: {worst['layer']} "
              f"({worst['worst_in_channel_levels']:.1f} of 127 levels)")
        print("Per-channel quantization handles between-channel spread; it is the "
              "in-channel figure that costs accuracy.")
    return rows
