#!/usr/bin/env python3
"""Measure color-resolved D4 observables on saved NQS checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any


REQUESTED_PLATFORM = (
    os.environ.get("MEASURE_JAX_PLATFORM")
    or os.environ.get("JAX_PLATFORMS")
    or os.environ.get("JAX_PLATFORM_NAME")
    or "cuda"
)
os.environ.setdefault("JAX_PLATFORMS", REQUESTED_PLATFORM)
if "," not in REQUESTED_PLATFORM:
    os.environ.setdefault("JAX_PLATFORM_NAME", REQUESTED_PLATFORM)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION", "True")

import flax
import flax.linen as nn
from flax import traverse_util
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

try:
    from .measurement_core import (
        COLOR_NAMES,
        COLORS,
        Geometry,
        build_electric_specs,
        build_magnetic_specs,
        json_default as core_json_default,
        measure_observable_batch,
        operator_config_hash,
        specs_to_manifest,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from measurement_core import (
        COLOR_NAMES,
        COLORS,
        Geometry,
        build_electric_specs,
        build_magnetic_specs,
        json_default as core_json_default,
        measure_observable_batch,
        operator_config_hash,
        specs_to_manifest,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
COLOR_OBSERVABLES = ("A", "B", "W_e", "W_m")
_RANDOM_STATE_REGISTERED = False
_JAX_NAMED_SHAPE_PATCHED = False


class InvalidCheckpointError(ValueError):
    pass


class InvalidMeasurementError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    L: int
    use_small_point_group: bool
    use_color_labels: bool

    @property
    def label(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class CheckpointArchitecture:
    family: str
    n_features: int | None
    n_layers: int | None
    has_mean_field: bool
    has_deep_cnn: bool
    has_color_kernels: bool

    def as_manifest(self) -> dict[str, Any]:
        return asdict(self)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return core_json_default(obj)


def stable_slug(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_")
    return f"{stem}_{digest}"


def parse_L(path: Path) -> int:
    match = re.search(r"_L(\d+)(?:_|$)", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_L(\d+)(?:_|$)", str(path))
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer L from checkpoint path {path}; pass --L")


def looks_color_resolved(path: Path) -> bool:
    text = str(path).lower()
    markers = (
        "hx_red",
        "hy_red",
        "hz_red",
        "hx_green",
        "hy_green",
        "hz_green",
        "hx_blue",
        "hy_blue",
        "hz_blue",
    )
    return any(marker in text for marker in markers)


def resolve_bool_mode(mode: str, *, auto: bool) -> bool:
    if mode == "auto":
        return auto
    if mode == "yes":
        return True
    if mode == "no":
        return False
    raise ValueError(f"unknown mode {mode!r}")


def import_repo_modules(repo: Path):
    repo = repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import global_vars as g  # noqa: PLC0415
    import cnn  # noqa: PLC0415

    return g, cnn


def configure_geometry_modules(g, L: int) -> None:
    g.L = L
    g.update_globals()


def register_random_state(g) -> None:
    global _RANDOM_STATE_REGISTERED
    if _RANDOM_STATE_REGISTERED:
        return

    @nk.hilbert.random.random_state.dispatch
    def random_state(hilb: nk.hilbert.Spin, key, size: int, *, dtype):
        del hilb
        out = jnp.ones((size, g.N), dtype=dtype)
        rs = jax.random.randint(
            key,
            shape=(size, g.transform_matrix.shape[1]),
            minval=0,
            maxval=2,
        )

        @jax.vmap
        def flip(sigma, index):
            b = g.transform_matrix @ index % 2
            return (sigma * (-1) ** b).astype(dtype)

        return flip(out, rs)

    _RANDOM_STATE_REGISTERED = True


def patch_jax_named_shape_pickle() -> None:
    global _JAX_NAMED_SHAPE_PATCHED
    if _JAX_NAMED_SHAPE_PATCHED:
        return
    try:
        from jax._src import core as jax_core

        original_update = jax_core.ShapedArray.update

        def compat_update(self, *args, **kwargs):
            kwargs.pop("named_shape", None)
            return original_update(self, *args, **kwargs)

        jax_core.ShapedArray.update = compat_update
    except Exception:
        pass
    _JAX_NAMED_SHAPE_PATCHED = True


def load_checkpoint_params(path: Path):
    patch_jax_named_shape_pickle()
    with path.open("rb") as handle:
        return pickle.load(handle)


def _flat_params(params) -> dict[tuple[str, ...], Any]:
    return traverse_util.flatten_dict(flax.core.unfreeze(params))


def _param_shape(flat: dict[tuple[str, ...], Any], key: tuple[str, ...]) -> tuple[int, ...] | None:
    value = flat.get(key)
    shape = getattr(value, "shape", None)
    return tuple(int(x) for x in shape) if shape is not None else None


def _deep_cnn_kernel_keys(flat: dict[tuple[str, ...], Any], branch: str) -> list[tuple[str, ...]]:
    keys = []
    for key in flat:
        if len(key) < 3 or key[0] != branch or key[-1] != "W":
            continue
        layer_name = key[-2]
        if layer_name == "input_conv" or str(layer_name).startswith("residual_conv_"):
            keys.append(key)
        elif str(layer_name).startswith("conv2_"):
            keys.append(key)
    return sorted(keys)


def _infer_deep_cnn_shape(flat: dict[tuple[str, ...], Any]) -> tuple[tuple[int, ...] | None, int | None]:
    keys = _deep_cnn_kernel_keys(flat, "deep_CNN_0")
    if not keys:
        return None, None

    first_key = next((key for key in keys if key[-2] == "input_conv"), keys[0])
    first_shape = _param_shape(flat, first_key)
    current_style_layers = [
        key for key in keys if key[-2] == "input_conv" or str(key[-2]).startswith("residual_conv_")
    ]
    if current_style_layers:
        return first_shape, len(current_style_layers)
    return first_shape, len(keys)


def validate_checkpoint_params(params, label: str) -> None:
    bad = []
    for key, value in _flat_params(params).items():
        arr = np.asarray(value)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan or n_inf:
            bad.append(("/".join(key), n_nan, n_inf, int(arr.size)))

    if not bad:
        return

    details = ", ".join(
        f"{name}: nan={n_nan}, inf={n_inf}, size={size}"
        for name, n_nan, n_inf, size in bad[:4]
    )
    if len(bad) > 4:
        details += f", ... {len(bad) - 4} more tensors"
    raise InvalidCheckpointError(f"{label} contains non-finite checkpoint parameters ({details})")


def infer_checkpoint_architecture(params) -> CheckpointArchitecture:
    flat = _flat_params(params)
    has_mean_field = any(key[:1] in (("mean_field0",), ("mean_field1",)) for key in flat)
    has_deep_cnn = any(key[:1] in (("deep_CNN_0",), ("deep_CNN_1",)) for key in flat)
    first_kernel_shape, n_layers = _infer_deep_cnn_shape(flat)
    n_features = first_kernel_shape[-1] if first_kernel_shape is not None else None
    has_color_kernels = bool(first_kernel_shape is not None and len(first_kernel_shape) == 4)

    if has_deep_cnn and has_mean_field:
        family = "cnn_symmetric_mean_field"
    elif has_deep_cnn:
        family = "legacy_cnn_symmetric"
    elif has_mean_field:
        family = "mean_field_symmetric"
    else:
        raise ValueError("Cannot infer checkpoint architecture from parameter tree")

    return CheckpointArchitecture(
        family=family,
        n_features=n_features,
        n_layers=n_layers,
        has_mean_field=has_mean_field,
        has_deep_cnn=has_deep_cnn,
        has_color_kernels=has_color_kernels,
    )


def infer_checkpoint_info(path: Path, args, architecture: CheckpointArchitecture) -> CheckpointInfo:
    L = args.L if args.L is not None else parse_L(path)
    auto_color_labels = architecture.has_color_kernels or looks_color_resolved(path)
    use_color_labels = resolve_bool_mode(args.color_labels, auto=auto_color_labels)
    if architecture.has_color_kernels and not use_color_labels:
        raise ValueError("checkpoint uses color-resolved kernels; --color-labels no is incompatible")
    use_small_point_group = resolve_bool_mode(
        args.small_point_group,
        auto=use_color_labels or looks_color_resolved(path),
    )
    return CheckpointInfo(
        path=path.resolve(),
        L=L,
        use_small_point_group=use_small_point_group,
        use_color_labels=use_color_labels,
    )


def make_legacy_cnn_symmetric(cnn, architecture: CheckpointArchitecture, info: CheckpointInfo):
    if architecture.n_features is None:
        raise ValueError("Legacy CNN checkpoint is missing deep-CNN feature shape")

    class LegacyCNNSymmetric(nn.Module):
        rotation: bool = True
        use_small_point_group: bool = False
        use_color_labels: bool = False
        n_features: int = architecture.n_features
        n_layers: int = 2 if architecture.n_layers is None else architecture.n_layers

        @nn.compact
        def __call__(self, x):
            rotation_group = cnn.g.small_point_group if self.use_small_point_group else cnn.g.point_group
            if self.rotation:
                x = x[:, rotation_group].reshape((-1, cnn.g.N))

            u0 = cnn.phase2(x)
            y1 = cnn.deep_CNN(
                n_features=self.n_features,
                n_layers=self.n_layers,
                use_color_labels=self.use_color_labels,
            )(u0)

            u1 = jnp.stack(
                (
                    jnp.prod(x[:, cnn.g.left_triangles], axis=-1),
                    jnp.prod(x[:, cnn.g.right_triangles], axis=-1),
                ),
                axis=-1,
            )
            y2 = cnn.deep_CNN(
                n_features=self.n_features,
                n_layers=self.n_layers,
                use_color_labels=self.use_color_labels,
            )(u1)

            out = jnp.sum(cnn.activation4(y1 + y2), axis=[1, 2]) / jnp.sqrt(3)
            if self.rotation:
                out = out.reshape(-1, rotation_group.shape[0])
                out = jnp.log(jnp.mean(jnp.exp(out), axis=-1))
            return out

    return LegacyCNNSymmetric(
        rotation=True,
        use_small_point_group=info.use_small_point_group,
        use_color_labels=info.use_color_labels,
    )


def build_checkpoint_model(cnn, info: CheckpointInfo, architecture: CheckpointArchitecture):
    if architecture.family == "legacy_cnn_symmetric":
        return make_legacy_cnn_symmetric(cnn, architecture, info)

    if architecture.family == "cnn_symmetric_mean_field":
        if architecture.n_features is None:
            raise ValueError("CNN checkpoint is missing deep-CNN feature shape")
        return cnn.CNN_symmetric(
            rotation=True,
            use_small_point_group=info.use_small_point_group,
            use_color_labels=info.use_color_labels,
            n_features=architecture.n_features,
            n_layers=2 if architecture.n_layers is None else architecture.n_layers,
        )

    if architecture.family == "mean_field_symmetric":
        return cnn.MeanField_symmetric(
            rotation=True,
            use_small_point_group=info.use_small_point_group,
            use_color_labels=info.use_color_labels,
        )

    raise ValueError(f"Unsupported checkpoint architecture {architecture.family!r}")


def lift_conv_params(old_value, target_value, g):
    if old_value.shape == target_value.shape:
        return old_value.astype(target_value.dtype)
    if (
        len(target_value.shape) == len(old_value.shape) + 1
        and target_value.shape[1:] == old_value.shape
        and target_value.shape[0] == g.n_unit_cell_colors
    ):
        expanded = jnp.stack([old_value] * g.n_unit_cell_colors, axis=0)
        return expanded.astype(target_value.dtype)
    raise ValueError(f"Cannot adapt parameter shape {old_value.shape} to {target_value.shape}")


def adapt_loaded_params(loaded_params, template_params, g):
    loaded_flat = traverse_util.flatten_dict(flax.core.unfreeze(loaded_params))
    template_flat = traverse_util.flatten_dict(flax.core.unfreeze(template_params))
    adapted_flat = {}
    for key, target_value in template_flat.items():
        if key not in loaded_flat:
            raise KeyError(f"Missing parameter {key} in loaded parameter tree")
        old_value = loaded_flat[key]
        if hasattr(old_value, "shape") and hasattr(target_value, "shape"):
            adapted_flat[key] = lift_conv_params(old_value, target_value, g)
        else:
            adapted_flat[key] = old_value
    return traverse_util.unflatten_dict(adapted_flat)


def build_vstate(
    g,
    cnn,
    info: CheckpointInfo,
    n_chains: int,
    n_samples: int,
    chunk_size: int,
    n_discard: int,
    loaded_params,
    architecture: CheckpointArchitecture,
):
    validate_checkpoint_params(loaded_params, info.label)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.N, inverted_ordering=False)
    rules = cnn.Gauge_trans(0.5)
    sampler = nk.sampler.MetropolisSampler(
        hi,
        rules,
        sweep_size=3 * g.N // 4,
        n_chains=n_chains,
        reset_chains=True,
        dtype=jnp.int64,
    )
    vstate = nk.vqs.MCState(
        sampler,
        model=build_checkpoint_model(cnn, info, architecture),
        chunk_size=chunk_size,
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
    )
    try:
        vstate.parameters = loaded_params
    except Exception:
        vstate.parameters = adapt_loaded_params(loaded_params, vstate.parameters, g)
    return vstate


def log_value_np(vstate, samples: np.ndarray, eval_batch: int) -> np.ndarray:
    out = []
    samples64 = samples.astype(np.int64, copy=False)
    for start in range(0, samples64.shape[0], eval_batch):
        chunk = jnp.asarray(samples64[start : start + eval_batch])
        out.append(np.asarray(vstate.log_value(chunk)))
    return np.concatenate(out, axis=0)


def measure_batch(vstate, geom: Geometry, e_specs, m_specs, eval_batch: int) -> dict[str, np.ndarray]:
    raw = np.asarray(vstate.sample())
    samples = raw.reshape((-1, geom.N)).astype(np.int8)

    def log_value_fn(batch: np.ndarray) -> np.ndarray:
        return log_value_np(vstate, batch, eval_batch)

    return measure_observable_batch(samples, log_value_fn, geom, e_specs, m_specs)


def validate_measurement_arrays(arrays: dict[str, np.ndarray], label: str) -> None:
    bad = []
    for key, value in arrays.items():
        if key == "samples":
            continue
        arr = np.asarray(value)
        n_bad = int((~np.isfinite(arr)).sum())
        if n_bad:
            bad.append((key, n_bad, int(arr.size)))
    if bad:
        details = ", ".join(f"{key}: nonfinite={n_bad}/{size}" for key, n_bad, size in bad)
        raise InvalidMeasurementError(f"{label} produced non-finite measurement values ({details})")


def summarize_values(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals)
    real = np.real(vals)
    imag = np.imag(vals)
    n = vals.shape[0]
    mean = np.mean(vals)
    return {
        "mean": float(np.mean(real)),
        "stderr": float(np.std(real, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "imag_mean": float(np.mean(imag)),
        "imag_stderr": float(np.std(imag, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "phase_pi": float(np.angle(mean) / np.pi),
        "modulus": float(abs(mean)),
    }


def summarize_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": int(arrays["A"].shape[0]),
        "operators": {},
    }
    for key in COLOR_OBSERVABLES:
        arr = np.asarray(arrays[key])
        entry = {}
        for color in COLORS:
            entry[COLOR_NAMES[color]] = summarize_values(arr[:, color])
        entry["rgb"] = summarize_values(np.mean(arr[:, list(COLORS)], axis=1))
        summary["operators"][key] = entry
    return summary


def save_arrays(path: Path, arrays: dict[str, np.ndarray], include_samples: bool) -> None:
    payload = {key: arrays[key] for key in COLOR_OBSERVABLES}
    if include_samples:
        payload["samples"] = arrays["samples"].astype(np.int8)
    np.savez_compressed(path, **payload)


def measure_checkpoint(path: Path, repo: Path, args) -> dict[str, Any]:
    loaded_params = load_checkpoint_params(path)
    architecture = infer_checkpoint_architecture(loaded_params)
    info = infer_checkpoint_info(path, args, architecture)
    g, cnn = import_repo_modules(repo)
    configure_geometry_modules(g, info.L)
    register_random_state(g)

    geom = Geometry.build(info.L)
    e_specs = build_electric_specs(geom, args.max_e_pairs_per_color, args.string_set)
    m_specs = build_magnetic_specs(geom, args.max_m_pairs_per_color, args.string_set)
    operator_manifest = specs_to_manifest(e_specs, m_specs)
    operator_hash = operator_config_hash(operator_manifest)

    n_samples = args.n_samples
    if n_samples % args.n_chains != 0:
        n_samples = int(math.ceil(n_samples / args.n_chains) * args.n_chains)

    vstate = build_vstate(
        g,
        cnn,
        info,
        n_chains=args.n_chains,
        n_samples=n_samples,
        chunk_size=args.chunk_size,
        n_discard=args.n_discard_per_chain,
        loaded_params=loaded_params,
        architecture=architecture,
    )
    arrays = measure_batch(vstate, geom, e_specs, m_specs, args.eval_batch)
    validate_measurement_arrays(arrays, info.label)
    summary = summarize_arrays(arrays)

    result = {
        "checkpoint": {
            "path": str(info.path),
            "L": info.L,
            "use_small_point_group": info.use_small_point_group,
            "use_color_labels": info.use_color_labels,
            "architecture": architecture.as_manifest(),
        },
        "sampling": {
            "platform": REQUESTED_PLATFORM,
            "jax_devices": [str(device) for device in jax.devices()],
            "n_chains": args.n_chains,
            "n_samples": n_samples,
            "n_discard_per_chain": args.n_discard_per_chain,
            "chunk_size": args.chunk_size,
            "eval_batch": args.eval_batch,
            "sampler": "NetKet MetropolisSampler with cnn.Gauge_trans(0.5)",
        },
        "operator_selection": {
            "string_set": args.string_set,
            "max_e_pairs_per_color": args.max_e_pairs_per_color,
            "max_m_pairs_per_color": args.max_m_pairs_per_color,
        },
        "operators": operator_manifest,
        "operator_config_hash": operator_hash,
        "summary": summary,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = stable_slug(path)
    if args.save_arrays:
        arrays_path = args.out_dir / f"{slug}.npz"
        save_arrays(arrays_path, arrays, include_samples=args.save_samples)
        result["arrays_path"] = str(arrays_path)
    json_path = args.out_dir / f"{slug}.json"
    json_path.write_text(json.dumps(result, indent=2, default=json_default) + "\n")
    result["json_path"] = str(json_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="+", type=Path, help="Checkpoint pickle file(s) to measure.")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="Repository root containing global_vars.py and cnn.py.")
    parser.add_argument("--out-dir", type=Path, default=Path("measurements"))
    parser.add_argument("--L", type=int, default=None, help="Lattice linear size. Inferred from the filename when omitted.")
    parser.add_argument("--small-point-group", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--color-labels", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--n-chains", type=int, default=32)
    parser.add_argument("--n-discard-per-chain", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--string-set", choices=("paper", "representative"), default="paper")
    parser.add_argument("--max-e-pairs-per-color", type=int, default=None)
    parser.add_argument("--max-m-pairs-per-color", type=int, default=None)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--save-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    print(f"requested platform: {REQUESTED_PLATFORM}")
    print(f"JAX devices: {jax.devices()}")
    for path in args.checkpoint:
        if not path.is_file():
            raise SystemExit(f"checkpoint does not exist: {path}")
        print(f"measure {path}")
        result = measure_checkpoint(path.resolve(), repo, args)
        compact = {
            "checkpoint": str(path),
            "sample_count": result["summary"]["sample_count"],
            "json_path": result["json_path"],
        }
        print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
