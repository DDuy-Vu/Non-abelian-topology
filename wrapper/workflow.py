"""Checkpointed training workflow for ``Non-abelian-topology``.

This module keeps the custom SR/Heun evolution from ``cnn.evolve``
but wraps it with structured logging, resumable checkpoints,
color-resolved fields, optional NetKet sharding, and post-training D4
measurements.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_RANDOM_STATE_DISPATCH_REGISTERED = False
_JAX_NAMED_SHAPE_PATCHED = False
_TRANSFER_INIT_SCALE = 0.05
_DEFAULT_INITIAL_PARAMS_PATH = "init_params_L6"
NetKetSpin = None


class NonFiniteStepError(ValueError):
    """Raised when one attempted evolution substep produces non-finite data."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class _InterruptedForRequeue(Exception):
    def __init__(self, signum: int):
        super().__init__(f"interrupted by signal {signum}")
        self.signum = int(signum)


@dataclass(frozen=True)
class StageSpec:
    name: str
    n_iter: int
    dt: float
    n_features: int
    n_layers: int


@dataclass(frozen=True)
class FieldSpec:
    hx: float = 0.2
    hy: float = 0.0
    hz: float = 0.2
    hx_red: float | None = None
    hy_red: float | None = None
    hz_red: float | None = None
    hx_green: float | None = None
    hy_green: float | None = None
    hz_green: float | None = None
    hx_blue: float | None = None
    hy_blue: float | None = None
    hz_blue: float | None = None

    @property
    def has_color_fields(self) -> bool:
        return any(
            value is not None
            for key, value in asdict(self).items()
            if "_" in key
        )


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def _tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        with redirect_stdout(_Tee(sys.stdout, handle)), redirect_stderr(_Tee(sys.stderr, handle)):
            yield


def _json_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, complex):
        real = float(value.real)
        imag = float(value.imag)
        return {
            "real": real if np.isfinite(real) else None,
            "imag": imag if np.isfinite(imag) else None,
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(_json_value(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_value(payload), allow_nan=False, sort_keys=True) + "\n")


def _write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        handle.write(value)
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _config_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _output_path(output_filename: str) -> Path:
    output_path = Path(output_filename).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _log_paths(output_path: Path) -> dict[str, Path]:
    base = output_path.with_suffix("")
    return {
        "stdout_log": base.with_name(base.name + ".stdout.log"),
        "diagnostics_log": base.with_name(base.name + ".diagnostics.jsonl"),
        "training_log": base.with_name(base.name + ".training.jsonl"),
        "resume_file": base.with_name(base.name + ".resume.json"),
        "params_file": base.with_name(base.name + ".params.pkl"),
        "state_file": base.with_name(base.name + ".state.mpack"),
        "measurement_summary": base.with_name(base.name + ".measurements.json"),
        "measurement_dir": base.with_name(base.name + ".measurements"),
    }


def _stage_params_file(logs: dict[str, Path], stage_name: str) -> Path:
    base = logs["params_file"].with_suffix("")
    return base.with_name(base.name + f".{stage_name}.pkl")


def _numpy_state_to_json(state) -> dict[str, Any]:
    return {
        "bit_generator": state[0],
        "state": state[1].tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _numpy_state_from_json(payload: dict[str, Any]):
    return (
        payload["bit_generator"],
        np.asarray(payload["state"], dtype=np.uint32),
        int(payload["pos"]),
        int(payload["has_gauss"]),
        float(payload["cached_gaussian"]),
    )


def _append_diagnostic(log_path: Path, event: str, *, started_at: float | None = None, **payload) -> None:
    now = time.time()
    record = {
        "event": event,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "time_epoch": now,
        **payload,
    }
    if started_at is not None:
        record["elapsed_seconds"] = now - started_at
    _append_jsonl(log_path, record)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _stats_mean_real(stats_payload: dict[str, Any] | None) -> float | None:
    if not isinstance(stats_payload, dict):
        return None
    for key in ("mean", "Mean"):
        value = stats_payload.get(key)
        if isinstance(value, dict):
            return _finite_float_or_none(value.get("real"))
        finite = _finite_float_or_none(value)
        if finite is not None:
            return finite
    return None


def _array_nonfinite_counts(value: Any) -> dict[str, int]:
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "S", "O"}:
        return {
            "size": int(arr.size),
            "nan_count": 0,
            "inf_count": 0,
            "nonfinite_count": 0,
        }
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)
    nonfinite_mask = ~np.isfinite(arr)
    return {
        "size": int(arr.size),
        "nan_count": int(nan_mask.sum()),
        "inf_count": int(inf_mask.sum()),
        "nonfinite_count": int(nonfinite_mask.sum()),
    }


def _merge_nonfinite_counts(target: dict[str, Any], counts: dict[str, int]) -> None:
    target["size"] += int(counts.get("size", 0))
    target["nan_count"] += int(counts.get("nan_count", 0))
    target["inf_count"] += int(counts.get("inf_count", 0))
    target["nonfinite_count"] += int(counts.get("nonfinite_count", 0))


def _stats_nonfinite_counts(stats) -> dict[str, Any]:
    out = {
        "attributes": {},
        "size": 0,
        "nan_count": 0,
        "inf_count": 0,
        "nonfinite_count": 0,
    }
    if stats is None:
        return out
    attr_groups = (
        ("mean", "Mean"),
        ("variance", "Variance"),
        ("error_of_mean", "Sigma"),
        ("R_hat",),
        ("tau_corr",),
    )
    for attr_group in attr_groups:
        attr = next((name for name in attr_group if hasattr(stats, name)), None)
        if attr is None:
            continue
        counts = _array_nonfinite_counts(getattr(stats, attr))
        out["attributes"][attr] = counts
        _merge_nonfinite_counts(out, counts)
    return out


def _payload_nonfinite_counts(stats_payload: dict[str, Any] | None) -> dict[str, int]:
    counts = {
        "size": 0,
        "nan_count": 0,
        "inf_count": 0,
        "nonfinite_count": 0,
    }
    if not isinstance(stats_payload, dict):
        return counts
    summary = stats_payload.get("nonfinite_counts")
    if isinstance(summary, dict):
        return {
            "size": int(summary.get("size", 0)),
            "nan_count": int(summary.get("nan_count", 0)),
            "inf_count": int(summary.get("inf_count", 0)),
            "nonfinite_count": int(summary.get("nonfinite_count", 0)),
        }
    if "mean" in stats_payload or "Mean" in stats_payload:
        if _stats_mean_real(stats_payload) is None:
            return {
                "size": 1,
                "nan_count": 0,
                "inf_count": 0,
                "nonfinite_count": 1,
            }
    return counts


def _record_energy_nonfinite_counts(record: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(record, dict) or "energy" not in record:
        return {
            "size": 0,
            "nan_count": 0,
            "inf_count": 0,
            "nonfinite_count": 0,
        }
    return _payload_nonfinite_counts(record.get("energy"))


def _tree_nonfinite_counts(tree) -> dict[str, int]:
    leaves = []

    def collect(x):
        leaves.append(np.asarray(x))
        return x

    import jax

    jax.tree_util.tree_map(collect, tree)
    out = {
        "leaf_count": len(leaves),
        "bad_leaf_count": 0,
        "size": 0,
        "nan_count": 0,
        "inf_count": 0,
        "nonfinite_count": 0,
    }
    for leaf in leaves:
        counts = _array_nonfinite_counts(leaf)
        out["size"] += counts["size"]
        out["nan_count"] += counts["nan_count"]
        out["inf_count"] += counts["inf_count"]
        out["nonfinite_count"] += counts["nonfinite_count"]
        if counts["nonfinite_count"]:
            out["bad_leaf_count"] += 1
    return out


def _emit_workflow_warning(
    *,
    code: str,
    logs: dict[str, Path],
    metadata: dict[str, Any],
    severity: str,
    started_at: float,
    **payload,
) -> dict[str, Any]:
    warning = {
        "code": code,
        "severity": severity,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "time_epoch": time.time(),
        **payload,
    }
    warnings = metadata.setdefault("warnings", [])
    warnings.append(warning)
    metadata["warning_count"] = len(warnings)
    metadata["health_status"] = "warning"
    _append_diagnostic(
        logs["diagnostics_log"],
        "workflow_warning",
        started_at=started_at,
        **warning,
    )
    return warning


def _update_stage_health_counters(stage_meta: dict[str, Any], record: dict[str, Any]) -> None:
    if record.get("accepted", True):
        stage_meta["accepted_iterations"] = int(stage_meta.get("accepted_iterations", 0)) + 1
        stage_meta["consecutive_skipped_iterations"] = 0
        return

    stage_meta["skipped_iterations"] = int(stage_meta.get("skipped_iterations", 0)) + 1
    stage_meta["consecutive_skipped_iterations"] = int(stage_meta.get("consecutive_skipped_iterations", 0)) + 1
    skip_reasons = stage_meta.setdefault("skip_reasons", {})
    reason = str(record.get("skip_reason", "unknown"))
    skip_reasons[reason] = int(skip_reasons.get(reason, 0)) + 1


def _emit_iteration_nonfinite_warnings(
    *,
    logs: dict[str, Path],
    metadata: dict[str, Any],
    record: dict[str, Any],
    stage_meta: dict[str, Any],
    started_at: float,
) -> None:
    completed = int(stage_meta.get("completed_iterations", 0))
    skipped = int(stage_meta.get("skipped_iterations", 0))
    consecutive = int(stage_meta.get("consecutive_skipped_iterations", 0))

    if not record.get("accepted", True):
        if record.get("skip_reason") == "nan_update_k2":
            step_name = "step2"
        elif record.get("skip_reason") == "nonfinite_update_k2":
            step_name = "step2"
        else:
            step_name = "step1"
        step_info = record.get(step_name, {})
        _emit_workflow_warning(
            code="nonfinite_update_skipped",
            logs=logs,
            metadata=metadata,
            severity="warning",
            started_at=started_at,
            source="canonical_skipped_update",
            stage=record.get("stage"),
            phase_iteration=record.get("phase_iteration"),
            completed_iterations=completed,
            skipped_iterations=skipped,
            consecutive_skipped_iterations=consecutive,
            skip_reason=record.get("skip_reason"),
            step=step_name,
            update_size=step_info.get("update_size"),
            update_nan_count=step_info.get("update_nan_count"),
            update_inf_count=step_info.get("update_inf_count"),
            update_nonfinite_count=step_info.get("update_nonfinite_count"),
            skip_reasons=stage_meta.get("skip_reasons", {}),
        )

    for retry_event in record.get("retry_events", []):
        _emit_workflow_warning(
            code="nonfinite_substep_retried",
            logs=logs,
            metadata=metadata,
            severity="warning",
            started_at=started_at,
            source="adaptive_substep_retry",
            stage=record.get("stage"),
            phase_iteration=record.get("phase_iteration"),
            completed_iterations=completed,
            attempt=retry_event.get("attempt"),
            next_step_size=retry_event.get("next_step_size"),
            error=retry_event.get("error"),
            details=retry_event.get("details", {}),
        )

    energy_counts = _record_energy_nonfinite_counts(record)
    if record.get("accepted", True) and energy_counts["nonfinite_count"]:
        _emit_workflow_warning(
            code="accepted_nonfinite_energy",
            logs=logs,
            metadata=metadata,
            severity="error",
            started_at=started_at,
            stage=record.get("stage"),
            phase_iteration=record.get("phase_iteration"),
            completed_iterations=completed,
            energy_nonfinite_counts=energy_counts,
            energy=record.get("energy"),
        )


def _emit_stage_completion_warnings(
    *,
    logs: dict[str, Path],
    metadata: dict[str, Any],
    stage_meta: dict[str, Any],
    stage_name: str,
    started_at: float,
) -> None:
    last_record = stage_meta.get("last_record")
    energy_counts = _record_energy_nonfinite_counts(last_record)
    if energy_counts["nonfinite_count"]:
        _emit_workflow_warning(
            code="nonfinite_stage_final_energy",
            logs=logs,
            metadata=metadata,
            severity="error",
            started_at=started_at,
            stage=stage_name,
            phase_iteration=(last_record or {}).get("phase_iteration"),
            completed_iterations=stage_meta.get("completed_iterations"),
            skipped_iterations=stage_meta.get("skipped_iterations", 0),
            consecutive_skipped_iterations=stage_meta.get("consecutive_skipped_iterations", 0),
            energy_nonfinite_counts=energy_counts,
            energy=(last_record or {}).get("energy"),
        )


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return False


def _count_gpu_id_list(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return 0
    count = 0
    for token in [part.strip() for part in text.split(",") if part.strip()]:
        match = re.fullmatch(r"(\d+)-(\d+)", token)
        if match is None:
            count += 1
        else:
            start, end = int(match.group(1)), int(match.group(2))
            count += abs(end - start) + 1
    return count


def _count_gpu_request(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    match = re.search(r"(\d+)(?!.*\d)", text)
    return None if match is None else int(match.group(1))


def _requested_gpu_count(env: dict[str, str]) -> tuple[int | None, str | None]:
    for key in ("CUDA_VISIBLE_DEVICES", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        count = _count_gpu_id_list(env.get(key))
        if count is not None:
            return count, key
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_GPUS"):
        count = _count_gpu_request(env.get(key))
        if count is not None:
            return count, key
    return None, None


def _gpu_runtime_enabled(env: dict[str, str]) -> bool:
    if str(env.get("JAX_PLATFORMS", "")).strip().lower() == "cpu":
        return False
    if str(env.get("JAX_PLATFORM_NAME", "")).strip().lower() == "cpu":
        return False
    return True


def _resolve_runtime(runtime_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    runtime_kwargs = dict(runtime_kwargs or {})
    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "platform")
    env.setdefault("nk.config.netket_experimental_disable_ode_jit", "True")
    env.setdefault("NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION", "True")
    env.setdefault("NETKET_EXPERIMENTAL_SHARDING_CPU", "0")

    mpl_dir = Path(env.get("TMPDIR", "/tmp")) / f"matplotlib-{env.get('USER', 'user')}"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))

    use_gpu = runtime_kwargs.get("use_gpu")
    jax_platform_name = runtime_kwargs.get("jax_platform_name")
    if jax_platform_name is None and use_gpu is not None:
        jax_platform_name = "gpu" if bool(use_gpu) else "cpu"
    if jax_platform_name is not None:
        env["JAX_PLATFORM_NAME"] = str(jax_platform_name)
        if str(jax_platform_name).lower() == "cpu":
            env["JAX_PLATFORMS"] = "cpu"

    cuda_visible_devices = runtime_kwargs.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    elif str(jax_platform_name).lower() == "cpu":
        env.setdefault("CUDA_VISIBLE_DEVICES", "")

    for key, value in dict(runtime_kwargs.get("extra_env") or {}).items():
        if key == "NETKET_EXPERIMENTAL_SHARDING":
            continue
        env[str(key)] = str(value)

    if str(env.get("JAX_PLATFORM_NAME", "")).lower() == "cpu":
        env["JAX_PLATFORMS"] = "cpu"

    requested_count, requested_source = _requested_gpu_count(env)
    if "use_sharding" in runtime_kwargs:
        sharding_enabled = _is_truthy(runtime_kwargs["use_sharding"])
        sharding_source = "runtime_kwargs.use_sharding"
    elif "NETKET_EXPERIMENTAL_SHARDING" in os.environ:
        sharding_enabled = _is_truthy(os.environ["NETKET_EXPERIMENTAL_SHARDING"])
        sharding_source = "environment.NETKET_EXPERIMENTAL_SHARDING"
    else:
        sharding_enabled = _gpu_runtime_enabled(env) and (requested_count or 0) > 1
        sharding_source = "auto_multi_gpu" if sharding_enabled else "auto_default_off"
    env["NETKET_EXPERIMENTAL_SHARDING"] = "1" if sharding_enabled else "0"

    return {
        "env": env,
        "jax_platform_name": env.get("JAX_PLATFORM_NAME"),
        "requested_gpu_count": requested_count,
        "requested_gpu_source": requested_source,
        "sharding_enabled": bool(sharding_enabled),
        "sharding_source": sharding_source,
    }


def _activate_runtime(resolved_runtime: dict[str, Any]) -> None:
    os.environ.update(dict(resolved_runtime["env"]))


def _patch_numpy_asarray_copy_kw() -> None:
    try:
        np.asarray(0, copy=False)
    except (TypeError, ValueError):
        original_asarray = np.asarray

        def compat_asarray(a, dtype=None, order=None, *, copy=None, like=None):
            if like is not None:
                return np.array(a, dtype=dtype, order=order, like=like)
            return original_asarray(a, dtype=dtype, order=order)

        np.asarray = compat_asarray


def _patch_jax_named_shape_pickle() -> None:
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


def _import_modules() -> dict[str, Any]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    _patch_numpy_asarray_copy_kw()
    _patch_jax_named_shape_pickle()

    import flax
    import jax
    import jax.numpy as jnp
    import netket as nk
    from flax import serialization, traverse_util
    from netket import jax as nkjax
    from netket.operator.spin import sigmax, sigmay, sigmaz
    from netket.optimizer.qgt.qgt_jacobian import QGTJacobian_DefaultConstructor
    from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
    from netket.stats import statistics as mpi_statistics
    from netket.utils import mpi

    g = importlib.import_module("global_vars")
    cnn = importlib.import_module("cnn")
    blurred_sampling = importlib.import_module("wrapper.blurred_sampling")

    return {
        "QGTJacobian_DefaultConstructor": QGTJacobian_DefaultConstructor,
        "blurred_sampling": blurred_sampling,
        "cnn": cnn,
        "convert_tree_to_dense_format": convert_tree_to_dense_format,
        "flax": flax,
        "g": g,
        "jax": jax,
        "jnp": jnp,
        "mpi": mpi,
        "mpi_statistics": mpi_statistics,
        "nk": nk,
        "nkjax": nkjax,
        "serialization": serialization,
        "sigmax": sigmax,
        "sigmay": sigmay,
        "sigmaz": sigmaz,
        "traverse_util": traverse_util,
    }


def _ensure_random_state_dispatch(modules: dict[str, Any]) -> None:
    global _RANDOM_STATE_DISPATCH_REGISTERED
    global NetKetSpin
    if _RANDOM_STATE_DISPATCH_REGISTERED:
        return

    nk = modules["nk"]
    jax = modules["jax"]
    jnp = modules["jnp"]
    g = modules["g"]
    NetKetSpin = nk.hilbert.Spin

    @nk.hilbert.random.random_state.dispatch
    def random_state(hilb: NetKetSpin, key, size: int, *, dtype):
        del hilb
        out = jnp.ones((size, g.N), dtype=dtype)
        rs = jax.random.randint(key, shape=(size, g.transform_matrix.shape[1]), minval=0, maxval=2)

        @jax.vmap
        def flip(sigma, index):
            b = g.transform_matrix @ index % 2
            return (sigma * (-1) ** b).astype(dtype)

        return flip(out, rs)

    _RANDOM_STATE_DISPATCH_REGISTERED = True


def _prepare_globals_for_sharding(g) -> None:
    for name in (
        "plaquette_list",
        "X_list",
        "left_triangles",
        "right_triangles",
        "X_list_r",
        "X_list_g",
        "X_list_b",
        "kernel2",
        "kernel3",
        "point_group",
        "small_point_group",
        "transform_matrix",
        "inverse_matrix",
        "adjacent_matrix",
        "path_matrix",
        "kx",
        "ky",
    ):
        if hasattr(g, name) and getattr(g, name) is not None:
            setattr(g, name, np.asarray(getattr(g, name)))


def _patch_sharding_functions(*, cnn, g, jax, jnp) -> None:
    x_list_host = np.asarray(g.X_list)
    translation_site_host = np.asarray(g.translation_site)
    translation_cell_host = np.asarray(g.translation_cell)
    kx_host = np.asarray(g.kx)
    ky_host = np.asarray(g.ky)
    inverse_matrix_host = np.asarray(g.inverse_matrix)
    transform_matrix_host = np.asarray(g.transform_matrix)
    kernel2_host = np.asarray(g.kernel2)
    kernel3_host = np.asarray(g.kernel3)

    def transition(self, sampler, machine, parameters, state, key, sigmas):
        del sampler, machine, parameters, state
        n_chains = sigmas.shape[0]
        key1, key2, key3 = jax.random.split(key, 3)
        ind_cluster = jax.random.randint(key1, shape=(n_chains,), minval=0, maxval=g.N_plaquette)
        ind_single = jax.random.randint(key2, shape=(n_chains,), minval=0, maxval=g.N)
        ind_which = jax.random.uniform(key3, shape=(n_chains,))

        @jax.vmap
        def flip(sigma, cluster_index, single_index, which):
            a = sigma.at[single_index].set(-sigma[single_index])
            plaquette = jnp.take(x_list_host, cluster_index, axis=0)
            b = sigma.at[plaquette].set(-sigma[plaquette])
            return jnp.where(which < self.plaquette_rate, b, a).astype(jnp.int64)

        return flip(sigmas, ind_cluster, ind_single, ind_which), None

    def state_reposition(s):
        s2 = (1 + s) // 2
        kx = jnp.asarray(kx_host)
        ky = jnp.asarray(ky_host)
        x_shift = jnp.round(jnp.angle(jnp.sum(kx[None, :] * s2, axis=-1)) * g.L / (2 * np.pi), 5)
        y_shift = jnp.round(jnp.angle(jnp.sum(ky[None, :] * s2, axis=-1)) * g.L / (2 * np.pi), 5)
        x_shift = jnp.where(x_shift <= 0, g.L - jnp.ceil(-x_shift), -jnp.ceil(-x_shift)).astype(int) % g.L
        y_shift = jnp.where(y_shift <= 0, g.L - jnp.ceil(-y_shift), -jnp.ceil(-y_shift)).astype(int) % g.L
        dis = y_shift * g.L + x_shift
        translated_sites = jnp.take(translation_site_host, dis, axis=0)
        rows = jnp.arange(s.shape[0])[:, None]
        return s[rows, translated_sites], ((g.L - y_shift) % g.L) * g.L + (g.L - x_shift) % g.L

    def phase2(x0):
        x, shift = state_reposition(x0)
        inverse_matrix = jnp.asarray(inverse_matrix_host)
        transform_matrix = jnp.asarray(transform_matrix_host)
        u = (((1 - x) // 2) @ inverse_matrix.T) % 2
        res = ((1 - x) // 2 + u @ transform_matrix.T) % 2
        a = jnp.sum(res[:, :, jnp.newaxis] * transform_matrix[jnp.newaxis, :, :], axis=1)
        b = jnp.where(a > 3, 1, 0)
        u = (u + b) % 2
        res = ((1 - x) // 2 + u @ transform_matrix.T) % 2
        shift_cells = jnp.take(translation_cell_host, shift, axis=0)
        shift_sites = jnp.take(translation_site_host, shift, axis=0)
        rows = jnp.arange(x.shape[0])[:, None]
        u = u[rows, shift_cells]
        res = res[rows, shift_sites]
        out = jnp.concatenate((u[:, :, None], res.reshape((res.shape[0], res.shape[1] // 3, 3))), axis=-1)
        return 2 * out - 1

    class conv2_sharded(cnn.nn.Module):
        out_features: int
        ker_size: int = 2
        use_color_labels: bool = False
        dtype: type = complex

        @cnn.nn.compact
        def __call__(self, x):
            size = x.shape
            n_kernel = self.ker_size**2
            mask = kernel2_host if self.ker_size == 2 else kernel3_host
            offset_mask = jnp.asarray(mask[:, :, 0], dtype=jnp.int32)

            if self.use_color_labels:
                color_mask = jnp.asarray(mask[:, :, 1], dtype=jnp.int32)
                target_colors = color_mask[:, 0]
                W = jnp.pad(
                    self.param(
                        "W",
                        cnn.nn.initializers.normal(stddev=0.5 / jnp.sqrt(n_kernel * size[-1])),
                        (g.n_unit_cell_colors, n_kernel, size[-1], self.out_features),
                        self.dtype,
                    ),
                    pad_width=((0, 0), (0, 1), (0, 0), (0, 0)),
                    constant_values=0j,
                )
                color_selector = jax.nn.one_hot(color_mask, W.shape[0], dtype=W.dtype)
                offset_selector = jax.nn.one_hot(offset_mask, W.shape[1], dtype=W.dtype)
                kernel = jnp.einsum("ijc,ijk,ckfo->ijfo", color_selector, offset_selector, W)
            else:
                W = jnp.pad(
                    self.param(
                        "W",
                        cnn.nn.initializers.normal(stddev=0.5 / jnp.sqrt(n_kernel * size[-1])),
                        (n_kernel, size[-1], self.out_features),
                        self.dtype,
                    ),
                    pad_width=((0, 1), (0, 0), (0, 0)),
                    constant_values=0j,
                )
                offset_selector = jax.nn.one_hot(offset_mask, W.shape[0], dtype=W.dtype)
                kernel = jnp.einsum("ijk,kfo->ijfo", offset_selector, W)

            y = jax.lax.dot_general(x, kernel, (((1, 2), (0, 2)), ((), ())))

            if self.use_color_labels:
                b = self.param(
                    "b",
                    cnn.nn.initializers.zeros_init(),
                    (g.n_unit_cell_colors, self.out_features),
                    self.dtype,
                )
                target_selector = jax.nn.one_hot(target_colors, b.shape[0], dtype=b.dtype)
                y += (target_selector @ b)[jnp.newaxis, :, :]
            else:
                b = self.param(
                    "b",
                    cnn.nn.initializers.zeros_init(),
                    (self.out_features,),
                    self.dtype,
                )
                y += b[jnp.newaxis, jnp.newaxis, :]

            return y

    cnn.Gauge_trans.transition = transition
    cnn.state_reposition = state_reposition
    cnn.phase2 = phase2
    cnn.conv2 = conv2_sharded


def _stats_payload(stats) -> dict[str, Any]:
    payload = {"repr": repr(stats)}
    for attr in ("mean", "Mean", "variance", "Variance", "error_of_mean", "Sigma", "R_hat", "tau_corr"):
        if hasattr(stats, attr):
            payload[attr] = _json_value(getattr(stats, attr))
    payload["nonfinite_counts"] = _stats_nonfinite_counts(stats)
    return payload


def _stats_are_finite(stats) -> bool:
    counts = _stats_nonfinite_counts(stats)
    return not counts["nonfinite_count"]


def _tree_has_nonfinite(tree) -> bool:
    return _tree_nonfinite_counts(tree)["nonfinite_count"] > 0


def _regularized_eigensolve(matrix, vector, *, jnp) -> Any:
    ev, V = jnp.linalg.eigh(matrix)
    rho = V.conj().T @ vector
    ev_inv = rho / ev
    rel = jnp.abs(ev / ev[-1])
    filt = jnp.where(rel > 1e-5, 1e1, jnp.where(rel > 1e-8, 0.5, 0.01))
    ev_inv_abs = jnp.abs(ev_inv)
    safe_abs = jnp.where(ev_inv_abs > 0, ev_inv_abs, 1.0)
    ev_inv = jnp.where(ev_inv_abs >= filt, filt * ev_inv / safe_abs, ev_inv)
    return V @ ev_inv


def _reinitialize_sampler(vstate, *, modules: dict[str, Any]) -> int:
    seed = int(np.random.randint(0, 2**31 - 1))
    try:
        vstate.sampler_state = vstate.sampler.init_state(
            vstate._sampler_model,
            vstate._sampler_variables,
            seed=seed,
        )
        if hasattr(vstate, "_sampler_state_previous"):
            vstate._sampler_state_previous = vstate.sampler_state
    except Exception:
        vstate.reset()
    vstate.reset()
    try:
        vstate.sample()
    except Exception:
        pass
    return seed


def _flatten_samples(samples, *, modules: dict[str, Any]):
    jax = modules["jax"]
    if samples.ndim >= 3:
        return jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)
    if samples.ndim == 1:
        return samples.reshape((1, -1))
    return samples


def _next_blurred_key(vstate, *, modules: dict[str, Any]):
    jax = modules["jax"]
    nkjax = modules["nkjax"]
    if hasattr(vstate, "_sampler_seed"):
        vstate._sampler_seed, key = jax.random.split(vstate._sampler_seed, 2)
        return key
    return nkjax.PRNGKey(int(np.random.randint(0, 2**31 - 1)))


def _blurred_operator_estimate(
    vstate,
    operator,
    *,
    blurred_sampling_settings: dict[str, Any],
    modules: dict[str, Any],
):
    jnp = modules["jnp"]
    mpi_statistics = modules["mpi_statistics"]
    blurred_sampling = modules["blurred_sampling"]

    chunk_size = blurred_sampling_settings.get("chunk_size")
    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)
    samples = _flatten_samples(vstate.samples, modules=modules)
    key = _next_blurred_key(vstate, modules=modules)
    samples_q, weights, local_values = blurred_sampling.blurred_sample_general(
        samples,
        key,
        vstate.parameters,
        vstate.model_state,
        float(blurred_sampling_settings["q"]),
        vstate._apply_fun,
        operator,
        chunk_size,
    )
    weights = jnp.real(weights.reshape(-1))
    weights = weights / jnp.mean(weights)
    local_values = local_values.reshape(-1)
    stats = mpi_statistics(weights * local_values)
    return {
        "stats": stats,
        "local_values": local_values,
        "weights": weights,
        "samples": samples_q,
        "ess": blurred_sampling.ess_from_weights(weights),
        "chunk_size": chunk_size,
    }


def _expect_operator(
    vstate,
    operator,
    *,
    blurred_sampling_settings: dict[str, Any],
    modules: dict[str, Any],
):
    if _is_truthy(blurred_sampling_settings.get("enabled")):
        data = _blurred_operator_estimate(
            vstate,
            operator,
            blurred_sampling_settings=blurred_sampling_settings,
            modules=modules,
        )
        return data["stats"], data
    return vstate.expect(operator), None


def _blurred_sr_matrix_and_rhs(
    *,
    active_indices,
    data: dict[str, Any],
    energy_stats,
    modules: dict[str, Any],
    rcond: float,
    vstate,
):
    jnp = modules["jnp"]
    nk = modules["nk"]
    mpi = modules["mpi"]
    QGTJacobian_DefaultConstructor = modules["QGTJacobian_DefaultConstructor"]
    g = modules["g"]

    weights = data["weights"].reshape(data["samples"].shape[:-1])
    pdf = weights / weights.size
    qgt = QGTJacobian_DefaultConstructor(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        data["samples"],
        pdf=pdf,
        dense=True,
        mode="holomorphic",
        chunk_size=data["chunk_size"],
    )
    O = qgt.O[:, active_indices] / g.N
    rhs = jnp.sqrt(pdf.reshape(-1)) * (data["local_values"] - energy_stats.mean) / g.N
    Sd = mpi.mpi_sum_jax(O.conj().T @ O)[0] + float(rcond) * jnp.eye(O.shape[1], dtype=O.dtype)
    force = nk.stats.sum(O.conj() * rhs[:, None], axis=0)
    return O, rhs, Sd, force


def _sr_direction(
    vstate,
    hamiltonian,
    active_indices,
    *,
    blurred_sampling_settings: dict[str, Any],
    modules: dict[str, Any],
    rcond: float,
    previous_variance: float | None,
    max_resample_attempts: int,
):
    jnp = modules["jnp"]
    nk = modules["nk"]
    mpi = modules["mpi"]
    mpi_statistics = modules["mpi_statistics"]
    convert_tree_to_dense_format = modules["convert_tree_to_dense_format"]
    nkjax = modules["nkjax"]
    g = modules["g"]

    energy_stats = None
    delta_energy = None
    blurred_data = None
    resample_count = 0
    for attempt in range(max(int(max_resample_attempts), 1)):
        if _is_truthy(blurred_sampling_settings.get("enabled")):
            blurred_data = _blurred_operator_estimate(
                vstate,
                hamiltonian,
                blurred_sampling_settings=blurred_sampling_settings,
                modules=modules,
            )
            energy_stats = blurred_data["stats"]
            delta_energy = blurred_data["local_values"] - energy_stats.mean
        else:
            local_energies = vstate.local_estimators(hamiltonian).reshape(-1)
            energy_stats = mpi_statistics(local_energies)
            delta_energy = local_energies - energy_stats.mean
        variance = float(np.real(np.asarray(energy_stats.Variance)))
        if previous_variance is not None and variance > 5.0 * max(float(previous_variance), 1e-12):
            _reinitialize_sampler(vstate, modules=modules)
            resample_count += 1
            continue
        break

    if energy_stats is None or delta_energy is None or not _stats_are_finite(energy_stats):
        raise NonFiniteStepError(
            "Encountered non-finite local energy statistics.",
            details={"source": "local_energy_statistics", "energy": _stats_payload(energy_stats)},
        )

    if _is_truthy(blurred_sampling_settings.get("enabled")):
        _, _, Sd, force = _blurred_sr_matrix_and_rhs(
            active_indices=active_indices,
            data=blurred_data,
            energy_stats=energy_stats,
            modules=modules,
            rcond=rcond,
            vstate=vstate,
        )
    else:
        qgt = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(mode="holomorphic"))
        O = qgt.O[:, active_indices] / g.N
        rhs = delta_energy / (np.sqrt(vstate.n_samples) * g.N)
        Sd = mpi.mpi_sum_jax(O.conj().T @ O)[0] + float(rcond) * jnp.eye(O.shape[1], dtype=O.dtype)
        force = nk.stats.sum(O.conj() * rhs[:, None], axis=0)
    update_index = _regularized_eigensolve(Sd, force, jnp=jnp)

    update_index_array = np.asarray(update_index)
    update_counts = _array_nonfinite_counts(update_index_array)
    if update_counts["nonfinite_count"]:
        raise NonFiniteStepError(
            "Encountered non-finite SR update.",
            details={"source": "sr_update", "update": update_counts},
        )

    update = jnp.zeros((vstate.n_parameters,), dtype=complex)
    update = update.at[active_indices].set(update_index)
    _, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
    update_tree = nkjax.tree_cast(reassemble(update), vstate.parameters)
    update_tree_counts = _tree_nonfinite_counts(update_tree)
    if update_tree_counts["nonfinite_count"]:
        raise NonFiniteStepError(
            "Encountered non-finite parameter update tree.",
            details={"source": "parameter_update_tree", "tree": update_tree_counts},
        )

    eigvals = np.asarray(jnp.linalg.eigvalsh(Sd))
    finite_eig = np.abs(eigvals[np.isfinite(eigvals)])
    info = {
        "active_parameter_count": int(len(active_indices)),
        "force_l2_norm": float(np.linalg.norm(np.asarray(force))),
        "update_l2_norm": float(np.linalg.norm(update_index_array)),
        "update_max_abs": float(np.max(np.abs(update_index_array))) if update_index_array.size else 0.0,
        "qgt_eig_abs_min": float(np.min(finite_eig)) if finite_eig.size else None,
        "qgt_eig_abs_max": float(np.max(finite_eig)) if finite_eig.size else None,
        "variance_resamples": int(resample_count),
    }
    if blurred_data is not None:
        info["blurred_sampling"] = {
            "enabled": True,
            "q": float(blurred_sampling_settings["q"]),
            "kernel": str(blurred_sampling_settings["kernel"]),
            "ess": _json_value(blurred_data["ess"]),
        }
    return update_tree, energy_stats, info


def _heun_substep(
    *,
    active_indices,
    blurred_sampling_settings: dict[str, Any],
    bp_operator,
    hamiltonian,
    max_resample_attempts: int,
    modules: dict[str, Any],
    previous_variance: float | None,
    rcond: float,
    step_size: float,
    vstate,
):
    jax = modules["jax"]
    old_parameters = vstate.parameters
    lr = -float(step_size)

    k1, pre_energy, step1_info = _sr_direction(
        vstate,
        hamiltonian,
        active_indices,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
        rcond=rcond,
        previous_variance=previous_variance,
        max_resample_attempts=max_resample_attempts,
    )
    vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr * y, old_parameters, k1)

    k2, midpoint_energy, step2_info = _sr_direction(
        vstate,
        hamiltonian,
        active_indices,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
        rcond=rcond,
        previous_variance=previous_variance,
        max_resample_attempts=max_resample_attempts,
    )
    vstate.parameters = jax.tree_util.tree_map(
        lambda x, y1, y2: x + 0.5 * lr * (y1 + y2),
        old_parameters,
        k1,
        k2,
    )

    if _tree_has_nonfinite(vstate.parameters):
        vstate.parameters = old_parameters
        raise NonFiniteStepError(
            "Accepted parameter tree contains non-finite values.",
            details={"source": "accepted_parameter_tree", "tree": _tree_nonfinite_counts(vstate.parameters)},
        )

    post_energy, _ = _expect_operator(
        vstate,
        hamiltonian,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
    )
    post_bp, post_bp_data = _expect_operator(
        vstate,
        bp_operator,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
    )
    if not _stats_are_finite(post_energy):
        vstate.parameters = old_parameters
        raise NonFiniteStepError(
            "Encountered non-finite post-step energy.",
            details={"source": "post_step_energy", "energy": _stats_payload(post_energy)},
        )
    if not _stats_are_finite(post_bp):
        vstate.parameters = old_parameters
        raise NonFiniteStepError(
            "Encountered non-finite post-step Bp.",
            details={"source": "post_step_bp", "bp": _stats_payload(post_bp)},
        )

    payload = {
        "energy": _stats_payload(post_energy),
        "bp": _stats_payload(post_bp),
        "update_reference_energy_pre": _stats_payload(pre_energy),
        "update_reference_energy_midpoint": _stats_payload(midpoint_energy),
        "step1": step1_info,
        "step2": step2_info,
        "variance": float(np.real(np.asarray(post_energy.Variance))),
    }
    if post_bp_data is not None:
        payload["bp_blurred_sampling"] = {
            "enabled": True,
            "q": float(blurred_sampling_settings["q"]),
            "ess": _json_value(post_bp_data["ess"]),
        }
    return payload


def _phase_status(completed_iterations: int, target_iterations: int) -> str:
    return "completed" if int(completed_iterations) >= int(target_iterations) else "running"


def _checkpoint_settings(checkpoint_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "enabled": True,
        "every_iterations": 1,
        "interval_seconds": 30 * 60,
    }
    settings.update(dict(checkpoint_kwargs or {}))
    return settings


def _diagnostics_settings(diagnostics_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "print_every": 1,
    }
    settings.update(dict(diagnostics_kwargs or {}))
    return settings


def _evolution_settings(evolution_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "active_parameter_count": 4000,
        "active_refresh_interval": 10,
        "adaptive_substeps": False,
        "max_step_retries": 8,
        "min_step_fraction": 1e-4,
        "variance_resample_attempts": 5,
        "initial_variance": 1e3,
    }
    settings.update(dict(evolution_kwargs or {}))
    return settings


def _sampler_settings(sampler_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "n_chains": 2**8,
        "sweep_size": None,
        "reset_chains": True,
        "plaquette_rate": 0.5,
    }
    settings.update(dict(sampler_kwargs or {}))
    return settings


def _vstate_settings(vstate_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "n_samples": 2**13,
        "chunk_size": 2**14,
        "n_discard_per_chain": 10,
    }
    settings.update(dict(vstate_kwargs or {}))
    return settings


def _blurred_sampling_settings(blurred_sampling_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    settings = {
        "enabled": False,
        "q": 0.2,
        "kernel": "general",
        "chunk_size": None,
    }
    settings.update(dict(blurred_sampling_kwargs or {}))
    settings["enabled"] = _is_truthy(settings.get("enabled"))
    settings["q"] = float(settings["q"])
    if not 0.0 < float(settings["q"]) < 1.0:
        raise ValueError("blurred_sampling_kwargs['q'] must be in (0, 1).")
    if str(settings["kernel"]).lower() != "general":
        raise ValueError("Only blurred_sampling_kwargs['kernel']='general' is supported.")
    if settings["chunk_size"] is not None:
        settings["chunk_size"] = int(settings["chunk_size"])
    return settings


def _measurement_settings(measurement_kwargs: dict[str, Any] | None, logs: dict[str, Path]) -> dict[str, Any]:
    settings = {
        "enabled": True,
        "out_dir": logs["measurement_dir"],
        "n_samples": 2048,
        "n_chains": 32,
        "n_discard_per_chain": 10,
        "chunk_size": 512,
        "eval_batch": 512,
        "string_set": "paper",
        "max_e_pairs_per_color": None,
        "max_m_pairs_per_color": None,
        "save_arrays": False,
        "save_samples": False,
        "small_point_group": "auto",
        "color_labels": "auto",
    }
    settings.update(dict(measurement_kwargs or {}))
    settings["out_dir"] = Path(settings["out_dir"]).expanduser().resolve()
    return settings


def canonical_training_defaults() -> dict[str, Any]:
    """Return the state-changing defaults used by the original ``GS.py`` workflow."""

    return {
        "hx": 0.2,
        "hy": 0.0,
        "hz": 0.2,
        "hx_red": None,
        "hy_red": None,
        "hz_red": None,
        "hx_green": None,
        "hy_green": None,
        "hz_green": None,
        "hx_blue": None,
        "hy_blue": None,
        "hz_blue": None,
        "Jz": -1.0,
        "Jx": -1.0,
        "staged_training": True,
        "pretrain_n_iter": 150,
        "pretrain_dt": 2.5,
        "pretrain_n_features": None,
        "pretrain_n_layers": 1,
        "n_iter": 600,
        "dt": 10.0,
        "n_features": 6,
        "n_layers": 3,
        "rotation": True,
        "use_small_point_group": None,
        "use_color_labels": None,
        "sampler_kwargs": _sampler_settings(None),
        "vstate_kwargs": _vstate_settings(None),
        "evolution_kwargs": _evolution_settings(None),
        "blurred_sampling_kwargs": _blurred_sampling_settings(None),
        "initial_params_path": _DEFAULT_INITIAL_PARAMS_PATH,
        "seed": None,
        "rcond": 1e-12,
    }


def _stage_specs(
    *,
    staged_training: bool,
    pretrain_n_iter: int,
    pretrain_dt: float,
    pretrain_n_features: int | None,
    pretrain_n_layers: int,
    n_iter: int,
    dt: float,
    n_features: int,
    n_layers: int,
) -> list[StageSpec]:
    stages: list[StageSpec] = []
    if staged_training:
        stages.append(
            StageSpec(
                name="pretrain",
                n_iter=int(pretrain_n_iter),
                dt=float(pretrain_dt),
                n_features=int(pretrain_n_features if pretrain_n_features is not None else max(1, n_features // 2)),
                n_layers=int(pretrain_n_layers),
            )
        )
    stages.append(
        StageSpec(
            name="main",
            n_iter=int(n_iter),
            dt=float(dt),
            n_features=int(n_features),
            n_layers=int(n_layers),
        )
    )
    return stages


def _fresh_metadata(
    *,
    config_hash: str,
    config_payload: dict[str, Any],
    logs: dict[str, Path],
    output_path: Path,
    stages: list[StageSpec],
    seed: int | None,
) -> dict[str, Any]:
    stage_payload = {}
    for stage in stages:
        stage_payload[stage.name] = {
            "status": "pending",
            "completed_iterations": 0,
            "target_iterations": stage.n_iter,
            "dt": stage.dt,
            "n_features": stage.n_features,
            "n_layers": stage.n_layers,
            "gamma": 0.0,
            "last_record": None,
            "accepted_iterations": 0,
            "skipped_iterations": 0,
            "consecutive_skipped_iterations": 0,
            "skip_reasons": {},
        }
    return {
        "schema_version": 1,
        "status": "running",
        "health_status": "ok",
        "warning_count": 0,
        "warnings": [],
        "function": "wrapper.workflow.run",
        "config_hash": config_hash,
        "config": config_payload,
        "output_filename": output_path,
        "stdout_log": logs["stdout_log"],
        "diagnostics_log": logs["diagnostics_log"],
        "training_log": logs["training_log"],
        "resume_file": logs["resume_file"],
        "params_file": logs["params_file"],
        "state_file": logs["state_file"],
        "current_stage": stages[0].name,
        "completed_total_iterations": 0,
        "seed": None if seed is None else int(seed),
        "stages": stage_payload,
    }


def _save_checkpoint(
    *,
    checkpoint_settings: dict[str, Any],
    config_hash: str,
    config_payload: dict[str, Any],
    logs: dict[str, Path],
    metadata: dict[str, Any],
    modules: dict[str, Any],
    output_path: Path,
    stage_name: str,
    status: str,
    vstate,
) -> dict[str, Any]:
    _write_pickle(logs["params_file"], vstate.parameters)
    _write_pickle(_stage_params_file(logs, stage_name), vstate.parameters)
    state_file_status = "not_saved"
    state_file_error = None
    try:
        _write_bytes(logs["state_file"], modules["serialization"].to_bytes(vstate))
        state_file_status = "saved"
    except Exception as exc:
        state_file_status = "failed"
        state_file_error = str(exc)
    saved = dict(metadata)
    saved.update(
        {
            "schema_version": 1,
            "status": status,
            "config_hash": config_hash,
            "config": config_payload,
            "output_filename": output_path,
            "stdout_log": logs["stdout_log"],
            "diagnostics_log": logs["diagnostics_log"],
            "training_log": logs["training_log"],
            "resume_file": logs["resume_file"],
            "params_file": logs["params_file"],
            "state_file": logs["state_file"],
            "state_file_status": state_file_status,
            "state_file_error": state_file_error,
            "checkpoint_settings": checkpoint_settings,
            "last_checkpoint_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "last_checkpoint_time_epoch": time.time(),
            "numpy_random_state": _numpy_state_to_json(np.random.get_state()),
        }
    )
    _write_json(logs["resume_file"], saved)
    return saved


def _should_checkpoint(
    *,
    completed_iterations: int,
    force: bool,
    last_checkpoint_iteration: int,
    last_checkpoint_time: float,
    settings: dict[str, Any],
) -> bool:
    if force:
        return True
    if not settings.get("enabled", True):
        return False
    if completed_iterations <= last_checkpoint_iteration:
        return False
    every_iterations = settings.get("every_iterations")
    if every_iterations is not None and completed_iterations - last_checkpoint_iteration >= int(every_iterations):
        return True
    interval_seconds = settings.get("interval_seconds")
    if interval_seconds is not None and time.time() - last_checkpoint_time >= float(interval_seconds):
        return True
    return False


def _install_signal_handlers():
    requested: dict[str, int | None] = {"signal": None}
    previous: dict[int, Any] = {}

    def handler(signum, frame):
        del frame
        requested["signal"] = int(signum)
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        print(f"Received {name}; checkpoint will be written after the current accepted iteration.")

    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def restore():
        for sig, old in previous.items():
            signal.signal(sig, old)

    return requested, restore


def _param_base_like(target_value, *, jnp, scale=1.0):
    return scale * jnp.array(target_value, dtype=target_value.dtype)


def _lift_param(old_value, target_value, *, g, jnp, scale=1.0):
    if old_value.shape == target_value.shape:
        return old_value.astype(target_value.dtype)

    def copy_overlap(source, target):
        slices = tuple(slice(0, min(source.shape[i], target.shape[i])) for i in range(source.ndim))
        return target.at[slices].set(source[slices])

    adapted = _param_base_like(target_value, jnp=jnp, scale=scale)
    if len(target_value.shape) == len(old_value.shape):
        return copy_overlap(old_value.astype(target_value.dtype), adapted)
    if len(target_value.shape) == len(old_value.shape) + 1 and target_value.shape[0] == g.n_unit_cell_colors:
        expanded = jnp.stack([old_value] * g.n_unit_cell_colors, axis=0).astype(target_value.dtype)
        return copy_overlap(expanded, adapted)
    raise ValueError(f"Cannot adapt parameter shape {old_value.shape} to {target_value.shape}")


def _adapt_params(loaded_params, template_params, *, modules: dict[str, Any], missing_scale=1.0):
    flax = modules["flax"]
    traverse_util = modules["traverse_util"]
    jnp = modules["jnp"]
    g = modules["g"]
    loaded_flat = traverse_util.flatten_dict(flax.core.unfreeze(loaded_params))
    template_flat = traverse_util.flatten_dict(flax.core.unfreeze(template_params))
    adapted_flat = {}
    for key, target_value in template_flat.items():
        if key not in loaded_flat:
            adapted_flat[key] = _param_base_like(target_value, jnp=jnp, scale=missing_scale)
            continue
        old_value = loaded_flat[key]
        if hasattr(old_value, "shape") and hasattr(target_value, "shape"):
            adapted_flat[key] = _lift_param(old_value, target_value, g=g, jnp=jnp, scale=missing_scale)
        else:
            adapted_flat[key] = old_value
    return traverse_util.unflatten_dict(adapted_flat)


def _load_params(path: Path):
    with path.open("rb") as handle:
        params = pickle.load(handle)
    if isinstance(params, dict) and "params" in params:
        return params["params"]
    return params


def _load_params_if_available(path: str | None):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    return _load_params(resolved)


def _build_sampler(
    *,
    hi,
    modules: dict[str, Any],
    sampler_settings: dict[str, Any],
):
    nk = modules["nk"]
    jnp = modules["jnp"]
    cnn = modules["cnn"]
    g = modules["g"]
    sweep_size = sampler_settings["sweep_size"]
    if sweep_size is None:
        sweep_size = 3 * g.N // 4
    return nk.sampler.MetropolisSampler(
        hi,
        cnn.Gauge_trans(float(sampler_settings["plaquette_rate"])),
        sweep_size=int(sweep_size),
        n_chains=int(sampler_settings["n_chains"]),
        reset_chains=bool(sampler_settings["reset_chains"]),
        dtype=jnp.int64,
    )


def _build_vstate(
    *,
    hi,
    modules: dict[str, Any],
    rotation: bool,
    sampler,
    seed: int | None,
    stage: StageSpec,
    use_color_labels: bool,
    use_small_point_group: bool,
    vstate_settings: dict[str, Any],
):
    nk = modules["nk"]
    cnn = modules["cnn"]
    model = cnn.CNN_symmetric(
        rotation=bool(rotation),
        use_small_point_group=bool(use_small_point_group),
        use_color_labels=bool(use_color_labels),
        n_features=int(stage.n_features),
        n_layers=int(stage.n_layers),
    )
    common_kwargs = {
        "chunk_size": int(vstate_settings["chunk_size"]),
        "n_samples": int(vstate_settings["n_samples"]),
        "n_discard_per_chain": int(vstate_settings["n_discard_per_chain"]),
    }
    if seed is None:
        return nk.vqs.MCState(sampler, model=model, **common_kwargs)
    try:
        return nk.vqs.MCState(sampler, model=model, seed=int(seed), **common_kwargs)
    except TypeError:
        return nk.vqs.MCState(sampler, model=model, **common_kwargs)


def _field_spec_from_kwargs(
    *,
    hx: float,
    hy: float,
    hz: float | None,
    hx_red: float | None,
    hy_red: float | None,
    hz_red: float | None,
    hx_green: float | None,
    hy_green: float | None,
    hz_green: float | None,
    hx_blue: float | None,
    hy_blue: float | None,
    hz_blue: float | None,
) -> FieldSpec:
    return FieldSpec(
        hx=float(hx),
        hy=float(hy),
        hz=0.2 if hz is None else float(hz),
        hx_red=None if hx_red is None else float(hx_red),
        hy_red=None if hy_red is None else float(hy_red),
        hz_red=None if hz_red is None else float(hz_red),
        hx_green=None if hx_green is None else float(hx_green),
        hy_green=None if hy_green is None else float(hy_green),
        hz_green=None if hz_green is None else float(hz_green),
        hx_blue=None if hx_blue is None else float(hx_blue),
        hy_blue=None if hy_blue is None else float(hy_blue),
        hz_blue=None if hz_blue is None else float(hz_blue),
    )


def _add_field_term(hamiltonian, coeff: float | None, op_factory, hi, sites):
    if coeff is None or float(coeff) == 0.0:
        return hamiltonian
    return hamiltonian - float(coeff) * sum(op_factory(hi, int(site)) for site in sites)


def _build_hamiltonian(*, field_spec: FieldSpec, hi, Jx: float, Jz: float, modules: dict[str, Any]):
    cnn = modules["cnn"]
    g = modules["g"]
    sigmax = modules["sigmax"]
    sigmay = modules["sigmay"]
    sigmaz = modules["sigmaz"]

    hamiltonian = cnn.Ruby_Hamiltonian(hi, float(Jz), float(Jx), use_cz_ring=True)
    all_sites = range(g.N)
    hamiltonian = _add_field_term(hamiltonian, field_spec.hx, sigmax, hi, all_sites)
    hamiltonian = _add_field_term(hamiltonian, field_spec.hy, sigmay, hi, all_sites)
    hamiltonian = _add_field_term(hamiltonian, field_spec.hz, sigmaz, hi, all_sites)

    color_sites = {
        "red": g.site_list_r,
        "green": g.site_list_g,
        "blue": g.site_list_b,
    }
    color_ops = {"hx": sigmax, "hy": sigmay, "hz": sigmaz}
    for color, sites in color_sites.items():
        if sites is None:
            continue
        for axis, op_factory in color_ops.items():
            coeff = getattr(field_spec, f"{axis}_{color}")
            hamiltonian = _add_field_term(hamiltonian, coeff, op_factory, hi, np.asarray(sites))
    return hamiltonian


def _bp_operator(*, hi, modules: dict[str, Any]):
    g = modules["g"]
    cnn = modules["cnn"]
    return cnn.Ruby_Hamiltonian(hi, 0.0, 1.0 / g.N_plaquette, use_cz_ring=True)


def _choose_active_indices(vstate, *, evolution_settings: dict[str, Any], modules: dict[str, Any]):
    jnp = modules["jnp"]
    explicit = evolution_settings.get("index_range")
    if explicit is not None:
        return jnp.asarray(np.asarray(explicit, dtype=np.int64))
    requested = evolution_settings.get("active_parameter_count")
    if requested is None:
        return jnp.arange(vstate.n_parameters)
    count = min(int(requested), int(vstate.n_parameters))
    selected = np.random.choice(vstate.n_parameters, size=count, replace=False)
    return jnp.asarray(np.sort(selected))


def _advance_iteration_adaptive(
    *,
    active_indices,
    blurred_sampling_settings: dict[str, Any],
    bp_operator,
    evolution_settings: dict[str, Any],
    hamiltonian,
    logs: dict[str, Path],
    modules: dict[str, Any],
    phase_iteration: int,
    previous_variance: float | None,
    rcond: float,
    started_at: float,
    stage: StageSpec,
    target_interval: float,
    total_completed_iterations: int,
    vstate,
):
    remaining = float(target_interval)
    current_step = remaining
    min_step = abs(float(target_interval)) * float(evolution_settings["min_step_fraction"])
    max_retries = int(evolution_settings["max_step_retries"])
    max_resample_attempts = int(evolution_settings["variance_resample_attempts"])
    accepted_substeps = 0
    retry_count = 0
    retry_events = []
    last_payload = None
    substep_index = 0

    while remaining > max(min_step * 0.5, 1e-15):
        current_step = min(current_step, remaining)
        old_parameters = vstate.parameters
        try:
            payload = _heun_substep(
                active_indices=active_indices,
                blurred_sampling_settings=blurred_sampling_settings,
                bp_operator=bp_operator,
                hamiltonian=hamiltonian,
                max_resample_attempts=max_resample_attempts,
                modules=modules,
                previous_variance=previous_variance,
                rcond=rcond,
                step_size=current_step,
                vstate=vstate,
            )
        except NonFiniteStepError as exc:
            vstate.parameters = old_parameters
            sampler_seed = _reinitialize_sampler(vstate, modules=modules)
            retry_count += 1
            current_step *= 0.5
            retry_event = {
                "event": "substep_retried",
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "stage": stage.name,
                "phase_iteration": phase_iteration,
                "attempt": retry_count,
                "next_step_size": current_step,
                "sampler_seed": sampler_seed,
                "error": str(exc),
                "details": getattr(exc, "details", {}),
            }
            retry_events.append(retry_event)
            _append_jsonl(
                logs["training_log"],
                retry_event,
            )
            if retry_count > max_retries or current_step < min_step:
                raise NonFiniteStepError(
                    f"Step failed after {retry_count} retries; last error: {exc}",
                    details={
                        "source": "adaptive_step_failed",
                        "retry_count": retry_count,
                        "last_error": str(exc),
                        "last_details": getattr(exc, "details", {}),
                    },
                ) from exc
            continue

        substep_index += 1
        accepted_substeps += 1
        remaining = max(0.0, remaining - current_step)
        previous_variance = payload["variance"]
        last_payload = payload
        _append_jsonl(
            logs["training_log"],
            {
                "event": "substep_completed",
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "time_epoch": time.time(),
                "elapsed_seconds": time.time() - started_at,
                "stage": stage.name,
                "phase_iteration": int(phase_iteration),
                "substep": int(substep_index),
                "step_size": float(current_step),
                "remaining_interval": float(remaining),
                "active_parameter_count": int(len(active_indices)),
                "energy": payload["energy"],
                "bp": payload["bp"],
                "step1": payload["step1"],
                "step2": payload["step2"],
                "acceptance": _json_value(getattr(vstate.sampler_state, "acceptance", None)),
            },
        )

    if last_payload is None:
        raise NonFiniteStepError(
            "No accepted substep was produced.",
            details={
                "source": "adaptive_no_accepted_substep",
                "retry_count": retry_count,
                "retry_events": retry_events,
            },
        )

    return {
        "event": "iteration_completed",
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "time_epoch": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "stage": stage.name,
        "phase_iteration": int(phase_iteration),
        "phase_target_iterations": int(stage.n_iter),
        "completed_total_iterations": int(total_completed_iterations),
        "dt": float(stage.dt),
        "target_step_size": float(target_interval),
        "accepted_substeps": int(accepted_substeps),
        "retry_count": int(retry_count),
        "retry_events": retry_events,
        "active_parameter_count": int(len(active_indices)),
        "energy": last_payload["energy"],
        "bp": last_payload["bp"],
        "update_reference_energy_pre": last_payload["update_reference_energy_pre"],
        "update_reference_energy_midpoint": last_payload["update_reference_energy_midpoint"],
        "step1": last_payload["step1"],
        "step2": last_payload["step2"],
        "acceptance": _json_value(getattr(vstate.sampler_state, "acceptance", None)),
    }


def _canonical_sr_direction(
    vstate,
    hamiltonian,
    active_indices,
    *,
    blurred_sampling_settings: dict[str, Any],
    modules: dict[str, Any],
    previous_variance: float,
    rcond: float,
    max_resample_attempts: int,
):
    jnp = modules["jnp"]
    nk = modules["nk"]
    mpi_statistics = modules["mpi_statistics"]
    convert_tree_to_dense_format = modules["convert_tree_to_dense_format"]
    nkjax = modules["nkjax"]
    g = modules["g"]
    cnn = modules["cnn"]

    energy_stats = None
    delta_energy = None
    blurred_data = None
    resample_count = 0
    for _ in range(max(int(max_resample_attempts), 1)):
        if _is_truthy(blurred_sampling_settings.get("enabled")):
            blurred_data = _blurred_operator_estimate(
                vstate,
                hamiltonian,
                blurred_sampling_settings=blurred_sampling_settings,
                modules=modules,
            )
            energy_stats = blurred_data["stats"]
            delta_energy = blurred_data["local_values"] - energy_stats.mean
        else:
            local_energies = vstate.local_estimators(hamiltonian).reshape(-1)
            energy_stats = mpi_statistics(local_energies)
            delta_energy = local_energies - energy_stats.mean
        if float(np.real(np.asarray(energy_stats.Variance))) > 5.0 * float(previous_variance):
            vstate._sampler_state = None
            vstate.reset()
            resample_count += 1
        else:
            break

    if _is_truthy(blurred_sampling_settings.get("enabled")):
        O, rhs, _, _ = _blurred_sr_matrix_and_rhs(
            active_indices=active_indices,
            data=blurred_data,
            energy_stats=energy_stats,
            modules=modules,
            rcond=rcond,
            vstate=vstate,
        )
    else:
        qgt = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(mode="holomorphic"))
        O = qgt.O[:, active_indices] / g.N
        rhs = delta_energy / (np.sqrt(vstate.n_samples) * g.N)
    update_index = cnn._sr_update_param_space(O, rhs, float(rcond))

    update_index_array = np.asarray(update_index)
    update_counts = _array_nonfinite_counts(update_index_array)
    has_nonfinite = bool(update_counts["nonfinite_count"])
    update = jnp.zeros((vstate.n_parameters,), dtype=complex)
    update = update.at[active_indices].set(update_index)
    _, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
    update_tree = nkjax.tree_cast(reassemble(update), vstate.parameters)

    info = {
        "active_parameter_count": int(len(active_indices)),
        "force_l2_norm": None,
        "update_l2_norm": float(np.linalg.norm(update_index_array)),
        "update_max_abs": float(np.max(np.abs(update_index_array))) if update_index_array.size else 0.0,
        "variance_resamples": int(resample_count),
        "has_nan": bool(update_counts["nan_count"]),
        "has_nonfinite": has_nonfinite,
        "update_size": update_counts["size"],
        "update_nan_count": update_counts["nan_count"],
        "update_inf_count": update_counts["inf_count"],
        "update_nonfinite_count": update_counts["nonfinite_count"],
    }
    if blurred_data is not None:
        info["blurred_sampling"] = {
            "enabled": True,
            "q": float(blurred_sampling_settings["q"]),
            "kernel": str(blurred_sampling_settings["kernel"]),
            "ess": _json_value(blurred_data["ess"]),
        }
    return update_tree, energy_stats, has_nonfinite, info


def _advance_iteration_canonical(
    *,
    active_indices,
    blurred_sampling_settings: dict[str, Any],
    bp_operator,
    evolution_settings: dict[str, Any],
    hamiltonian,
    logs: dict[str, Path],
    modules: dict[str, Any],
    phase_iteration: int,
    previous_variance: float,
    rcond: float,
    started_at: float,
    stage: StageSpec,
    target_interval: float,
    total_completed_iterations: int,
    vstate,
):
    del logs
    jax = modules["jax"]
    lr = -float(target_interval)
    old_parameters = vstate.parameters
    max_resample_attempts = int(evolution_settings["variance_resample_attempts"])

    k1, pre_energy, has_nan, step1_info = _canonical_sr_direction(
        vstate,
        hamiltonian,
        active_indices,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
        previous_variance=previous_variance,
        rcond=rcond,
        max_resample_attempts=max_resample_attempts,
    )
    if has_nan:
        vstate.parameters = jax.tree_util.tree_map(lambda x: x, old_parameters)
        vstate.reset()
        return {
            "event": "iteration_skipped",
            "accepted": False,
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "time_epoch": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "stage": stage.name,
            "phase_iteration": int(phase_iteration),
            "phase_target_iterations": int(stage.n_iter),
            "completed_total_iterations": int(total_completed_iterations),
            "dt": float(stage.dt),
            "target_step_size": float(target_interval),
            "skip_reason": "nan_update_k1" if step1_info.get("update_nan_count", 0) else "nonfinite_update_k1",
            "energy": _stats_payload(pre_energy),
            "step1": step1_info,
        }

    vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr * y, old_parameters, k1)
    k2, midpoint_energy, has_nan, step2_info = _canonical_sr_direction(
        vstate,
        hamiltonian,
        active_indices,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
        previous_variance=previous_variance,
        rcond=rcond,
        max_resample_attempts=max_resample_attempts,
    )
    if has_nan:
        vstate.parameters = jax.tree_util.tree_map(lambda x: x, old_parameters)
        vstate.reset()
        return {
            "event": "iteration_skipped",
            "accepted": False,
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "time_epoch": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "stage": stage.name,
            "phase_iteration": int(phase_iteration),
            "phase_target_iterations": int(stage.n_iter),
            "completed_total_iterations": int(total_completed_iterations),
            "dt": float(stage.dt),
            "target_step_size": float(target_interval),
            "skip_reason": "nan_update_k2" if step2_info.get("update_nan_count", 0) else "nonfinite_update_k2",
            "energy": _stats_payload(midpoint_energy),
            "update_reference_energy_pre": _stats_payload(pre_energy),
            "step1": step1_info,
            "step2": step2_info,
        }

    vstate.parameters = jax.tree_util.tree_map(
        lambda x, y1, y2: x + 0.5 * lr * (y1 + y2),
        old_parameters,
        k1,
        k2,
    )
    bp_stats, bp_data = _expect_operator(
        vstate,
        bp_operator,
        blurred_sampling_settings=blurred_sampling_settings,
        modules=modules,
    )
    bp_payload = _stats_payload(bp_stats)
    if bp_data is not None:
        bp_payload["blurred_sampling"] = {
            "enabled": True,
            "q": float(blurred_sampling_settings["q"]),
            "ess": _json_value(bp_data["ess"]),
        }
    return {
        "event": "iteration_completed",
        "accepted": True,
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "time_epoch": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "stage": stage.name,
        "phase_iteration": int(phase_iteration),
        "phase_target_iterations": int(stage.n_iter),
        "completed_total_iterations": int(total_completed_iterations),
        "dt": float(stage.dt),
        "target_step_size": float(target_interval),
        "accepted_substeps": 1,
        "retry_count": 0,
        "active_parameter_count": int(len(active_indices)),
        "energy": _stats_payload(midpoint_energy),
        "bp": bp_payload,
        "update_reference_energy_pre": _stats_payload(pre_energy),
        "update_reference_energy_midpoint": _stats_payload(midpoint_energy),
        "step1": step1_info,
        "step2": step2_info,
        "acceptance": _json_value(getattr(vstate.sampler_state, "acceptance", None)),
    }


def _advance_iteration(**kwargs):
    evolution_settings = kwargs["evolution_settings"]
    if _is_truthy(evolution_settings.get("adaptive_substeps")):
        return _advance_iteration_adaptive(**kwargs)
    return _advance_iteration_canonical(**kwargs)


def _iteration_print_line(record: dict[str, Any]) -> str:
    if not record.get("accepted", True):
        return (
            f"{record['stage']} iter {record['phase_iteration']}/{record['phase_target_iterations']}: "
            f"skipped {record.get('skip_reason', 'iteration')}"
        )
    energy = record["energy"].get("repr", record["energy"])
    bp = record["bp"].get("repr", record["bp"])
    return (
        f"{record['stage']} iter {record['phase_iteration']}/{record['phase_target_iterations']}: "
        f"E={energy} Bp={bp} acc={record['acceptance']} "
        f"substeps={record['accepted_substeps']} retries={record['retry_count']} "
        f"|dw|={record['step2']['update_l2_norm']:.3e}"
    )


def _run_measurements(
    *,
    L: int,
    logs: dict[str, Path],
    measurement_settings: dict[str, Any],
    params_file: Path,
    started_at: float,
) -> dict[str, Any] | None:
    if not measurement_settings.get("enabled", True):
        return None
    if str(measurement_settings.get("platform", "")).strip():
        platform = str(measurement_settings["platform"])
        os.environ.setdefault("MEASURE_JAX_PLATFORM", platform)
        os.environ.setdefault("JAX_PLATFORMS", platform)
        if "," not in platform:
            os.environ.setdefault("JAX_PLATFORM_NAME", platform)

    from d4_measurements import measure_checkpoints

    args = SimpleNamespace(
        L=int(L),
        repo=PROJECT_ROOT,
        out_dir=Path(measurement_settings["out_dir"]),
        small_point_group=measurement_settings["small_point_group"],
        color_labels=measurement_settings["color_labels"],
        n_samples=int(measurement_settings["n_samples"]),
        n_chains=int(measurement_settings["n_chains"]),
        n_discard_per_chain=int(measurement_settings["n_discard_per_chain"]),
        chunk_size=int(measurement_settings["chunk_size"]),
        eval_batch=int(measurement_settings["eval_batch"]),
        string_set=measurement_settings["string_set"],
        max_e_pairs_per_color=measurement_settings["max_e_pairs_per_color"],
        max_m_pairs_per_color=measurement_settings["max_m_pairs_per_color"],
        save_arrays=bool(measurement_settings["save_arrays"]),
        save_samples=bool(measurement_settings["save_samples"]),
    )
    _append_diagnostic(
        logs["diagnostics_log"],
        "measurements_start",
        started_at=started_at,
        params_file=params_file,
        out_dir=args.out_dir,
    )
    result = measure_checkpoints.measure_checkpoint(params_file.resolve(), PROJECT_ROOT, args)
    _write_json(logs["measurement_summary"], result)
    _append_diagnostic(
        logs["diagnostics_log"],
        "measurements_complete",
        started_at=started_at,
        json_path=result.get("json_path"),
        summary_path=logs["measurement_summary"],
    )
    return result


def _write_final_summary(
    *,
    config_payload: dict[str, Any],
    lattice_payload: dict[str, Any],
    logs: dict[str, Path],
    measurement_result: dict[str, Any] | None,
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    _write_json(
        output_path,
        {
            "status": "completed",
            "health_status": metadata.get("health_status", "ok"),
            "warning_count": len(metadata.get("warnings", [])),
            "warnings": metadata.get("warnings", []),
            "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "config": config_payload,
            "lattice": lattice_payload,
            "stages": metadata["stages"],
            "params_file": logs["params_file"],
            "resume_file": logs["resume_file"],
            "training_log": logs["training_log"],
            "diagnostics_log": logs["diagnostics_log"],
            "stdout_log": logs["stdout_log"],
            "measurements": measurement_result,
            "measurement_summary": logs["measurement_summary"] if measurement_result is not None else None,
        },
    )


def run(
    *,
    L: int,
    output_filename: str,
    hx: float = 0.2,
    hy: float = 0.0,
    hz: float | None = 0.2,
    hx_red: float | None = None,
    hy_red: float | None = None,
    hz_red: float | None = None,
    hx_green: float | None = None,
    hy_green: float | None = None,
    hz_green: float | None = None,
    hx_blue: float | None = None,
    hy_blue: float | None = None,
    hz_blue: float | None = None,
    Jz: float = -1.0,
    Jx: float = -1.0,
    staged_training: bool = True,
    pretrain_n_iter: int = 150,
    pretrain_dt: float = 2.5,
    pretrain_n_features: int | None = None,
    pretrain_n_layers: int = 1,
    n_iter: int = 600,
    dt: float = 10.0,
    n_features: int = 6,
    n_layers: int = 3,
    rotation: bool = True,
    use_small_point_group: bool | None = None,
    use_color_labels: bool | None = None,
    sampler_kwargs: dict[str, Any] | None = None,
    vstate_kwargs: dict[str, Any] | None = None,
    runtime_kwargs: dict[str, Any] | None = None,
    checkpoint_kwargs: dict[str, Any] | None = None,
    diagnostics_kwargs: dict[str, Any] | None = None,
    evolution_kwargs: dict[str, Any] | None = None,
    blurred_sampling_kwargs: dict[str, Any] | None = None,
    measurement_kwargs: dict[str, Any] | None = None,
    initial_params_path: str | None = _DEFAULT_INITIAL_PARAMS_PATH,
    seed: int | None = None,
    rcond: float = 1e-12,
    show_progress: bool = True,
) -> None:
    """Run a resumable Non-Abelian topology training job.

    The color fields are additive on top of the global field.  For example,
    ``hx=0.2, hx_red=0.1`` applies an X field of ``0.3`` on red sites and
    ``0.2`` on green/blue sites.
    """

    started_at = time.time()
    output_path = _output_path(output_filename)
    logs = _log_paths(output_path)
    checkpoint_settings = _checkpoint_settings(checkpoint_kwargs)
    diagnostics_settings = _diagnostics_settings(diagnostics_kwargs)
    evolution_settings = _evolution_settings(evolution_kwargs)
    blurred_sampling_settings = _blurred_sampling_settings(blurred_sampling_kwargs)
    sampler_settings = _sampler_settings(sampler_kwargs)
    vstate_settings = _vstate_settings(vstate_kwargs)
    measurement_settings = _measurement_settings(measurement_kwargs, logs)
    field_spec = _field_spec_from_kwargs(
        hx=hx,
        hy=hy,
        hz=hz,
        hx_red=hx_red,
        hy_red=hy_red,
        hz_red=hz_red,
        hx_green=hx_green,
        hy_green=hy_green,
        hz_green=hz_green,
        hx_blue=hx_blue,
        hy_blue=hy_blue,
        hz_blue=hz_blue,
    )
    stages = _stage_specs(
        staged_training=staged_training,
        pretrain_n_iter=pretrain_n_iter,
        pretrain_dt=pretrain_dt,
        pretrain_n_features=pretrain_n_features,
        pretrain_n_layers=pretrain_n_layers,
        n_iter=n_iter,
        dt=dt,
        n_features=n_features,
        n_layers=n_layers,
    )
    resolved_runtime = _resolve_runtime(runtime_kwargs)

    use_small_pg = field_spec.has_color_fields if use_small_point_group is None else bool(use_small_point_group)
    use_colors = use_small_pg if use_color_labels is None else bool(use_color_labels)

    config_payload = {
        "L": int(L),
        "Jx": float(Jx),
        "Jz": float(Jz),
        "fields": asdict(field_spec),
        "staged_training": bool(staged_training),
        "stages": [asdict(stage) for stage in stages],
        "rotation": bool(rotation),
        "use_small_point_group": bool(use_small_pg),
        "use_color_labels": bool(use_colors),
        "sampler_kwargs": sampler_settings,
        "vstate_kwargs": vstate_settings,
        "runtime_kwargs": runtime_kwargs or {},
        "checkpoint_kwargs": checkpoint_settings,
        "diagnostics_kwargs": diagnostics_settings,
        "evolution_kwargs": evolution_settings,
        "blurred_sampling_kwargs": blurred_sampling_settings,
        "measurement_kwargs": _json_value(measurement_settings),
        "initial_params_path": initial_params_path,
        "seed": None if seed is None else int(seed),
        "rcond": float(rcond),
    }
    config_hash = _config_hash(config_payload)

    if output_path.exists() and not logs["resume_file"].exists():
        print(f"{output_path} already exists and no resume metadata was found. Assuming it is complete.")
        return

    _activate_runtime(resolved_runtime)
    modules = _import_modules()
    _ensure_random_state_dispatch(modules)
    if seed is not None:
        np.random.seed(int(seed))

    g = modules["g"]
    nk = modules["nk"]
    jnp = modules["jnp"]
    cnn = modules["cnn"]
    g.L = int(L)
    g.update_globals()
    if field_spec.has_color_fields and g.site_list_r is None:
        raise ValueError("Color-resolved fields require L divisible by 3.")
    if resolved_runtime["sharding_enabled"]:
        _prepare_globals_for_sharding(g)
        _patch_sharding_functions(cnn=cnn, g=g, jax=modules["jax"], jnp=jnp)

    hi = nk.hilbert.Spin(s=1 / 2, N=g.N, inverted_ordering=False)
    sampler = _build_sampler(hi=hi, modules=modules, sampler_settings=sampler_settings)
    hamiltonian = _build_hamiltonian(field_spec=field_spec, hi=hi, Jx=Jx, Jz=Jz, modules=modules)
    bp_operator = _bp_operator(hi=hi, modules=modules)
    lattice_payload = {"L": int(L), "N": int(g.N), "N_plaquette": int(g.N_plaquette)}

    resume_payload = _read_json(logs["resume_file"])
    if resume_payload is not None and resume_payload.get("config_hash") != config_hash:
        raise ValueError(f"Existing resume metadata for {output_path} does not match this run configuration.")

    metadata = resume_payload or _fresh_metadata(
        config_hash=config_hash,
        config_payload=config_payload,
        logs=logs,
        output_path=output_path,
        stages=stages,
        seed=seed,
    )
    if metadata.get("status") == "completed":
        print(f"{output_path} is already complete.")
        return
    if metadata.get("numpy_random_state") is not None:
        np.random.set_state(_numpy_state_from_json(metadata["numpy_random_state"]))

    stage_by_name = {stage.name: stage for stage in stages}
    current_stage_name = metadata.get("current_stage") or next(
        (stage.name for stage in stages if metadata["stages"][stage.name]["status"] != "completed"),
        stages[-1].name,
    )
    current_stage = stage_by_name[current_stage_name]
    vstate = _build_vstate(
        hi=hi,
        modules=modules,
        rotation=rotation,
        sampler=sampler,
        seed=seed,
        stage=current_stage,
        use_color_labels=use_colors,
        use_small_point_group=use_small_pg,
        vstate_settings=vstate_settings,
    )
    if logs["params_file"].exists():
        vstate.parameters = _adapt_params(_load_params(logs["params_file"]), vstate.parameters, modules=modules)
    else:
        loaded_params = _load_params_if_available(initial_params_path)
        if loaded_params is not None:
            vstate.parameters = _adapt_params(loaded_params, vstate.parameters, modules=modules)

    signal_request, restore_signals = _install_signal_handlers()
    pending_interrupt_signum: int | None = None

    with _tee_output(logs["stdout_log"]):
        try:
            _append_diagnostic(
                logs["diagnostics_log"],
                "runtime_setup",
                started_at=started_at,
                jax_platform_name=resolved_runtime.get("jax_platform_name"),
                requested_gpu_count=resolved_runtime.get("requested_gpu_count"),
                requested_gpu_source=resolved_runtime.get("requested_gpu_source"),
                sharding_enabled=resolved_runtime.get("sharding_enabled"),
                sharding_source=resolved_runtime.get("sharding_source"),
                jax_devices=[str(device) for device in modules["jax"].devices()],
            )
            print(
                "Runtime setup: "
                f"JAX platform={resolved_runtime.get('jax_platform_name') or 'default'}, "
                f"gpus={resolved_runtime.get('requested_gpu_count')} "
                f"from {resolved_runtime.get('requested_gpu_source') or 'unknown'}, "
                f"sharding={'enabled' if resolved_runtime.get('sharding_enabled') else 'disabled'}."
            )
            print(f"L={g.L}, N={g.N}, fields={asdict(field_spec)}")

            metadata = _save_checkpoint(
                checkpoint_settings=checkpoint_settings,
                config_hash=config_hash,
                config_payload=config_payload,
                logs=logs,
                metadata=metadata,
                modules=modules,
                output_path=output_path,
                stage_name=current_stage.name,
                status="running",
                vstate=vstate,
            )

            for stage_index, stage in enumerate(stages):
                stage_meta = metadata["stages"][stage.name]
                if stage_meta.get("status") == "completed":
                    continue

                if stage.name != current_stage.name:
                    vstate = _build_vstate(
                        hi=hi,
                        modules=modules,
                        rotation=rotation,
                        sampler=sampler,
                        seed=None if seed is None else seed + stage_index,
                        stage=stage,
                        use_color_labels=use_colors,
                        use_small_point_group=use_small_pg,
                        vstate_settings=vstate_settings,
                    )
                    vstate.parameters = _adapt_params(
                        _load_params(logs["params_file"]),
                        vstate.parameters,
                        modules=modules,
                        missing_scale=_TRANSFER_INIT_SCALE,
                    )
                    current_stage = stage
                    metadata["current_stage"] = stage.name
                    metadata = _save_checkpoint(
                        checkpoint_settings=checkpoint_settings,
                        config_hash=config_hash,
                        config_payload=config_payload,
                        logs=logs,
                        metadata=metadata,
                        modules=modules,
                        output_path=output_path,
                        stage_name=stage.name,
                        status="running",
                        vstate=vstate,
                    )

                completed = int(stage_meta.get("completed_iterations", 0))
                target = int(stage.n_iter)
                interval = float(stage.dt) / max(target, 1)
                print(
                    f"{stage.name}: {target - completed} iterations remaining, "
                    f"dt={stage.dt}, step={interval}, features={stage.n_features}, layers={stage.n_layers}"
                )
                _append_diagnostic(
                    logs["diagnostics_log"],
                    "stage_start",
                    started_at=started_at,
                    stage=stage.name,
                    completed_iterations=completed,
                    target_iterations=target,
                    n_features=stage.n_features,
                    n_layers=stage.n_layers,
                )

                active_indices = None
                last_checkpoint_time = time.time()
                last_checkpoint_iteration = completed
                previous_variance = (
                    None
                    if _is_truthy(evolution_settings.get("adaptive_substeps"))
                    else float(evolution_settings["initial_variance"])
                )
                refresh_interval = int(evolution_settings["active_refresh_interval"])
                print_every = max(int(diagnostics_settings["print_every"]), 1)
                for iteration in range(completed, target):
                    if active_indices is None or iteration % refresh_interval == 0:
                        active_indices = _choose_active_indices(
                            vstate,
                            evolution_settings=evolution_settings,
                            modules=modules,
                        )

                    record = _advance_iteration(
                        active_indices=active_indices,
                        blurred_sampling_settings=blurred_sampling_settings,
                        bp_operator=bp_operator,
                        evolution_settings=evolution_settings,
                        hamiltonian=hamiltonian,
                        logs=logs,
                        modules=modules,
                        phase_iteration=iteration + 1,
                        previous_variance=previous_variance,
                        rcond=float(rcond),
                        started_at=started_at,
                        stage=stage,
                        target_interval=interval,
                        total_completed_iterations=int(metadata.get("completed_total_iterations", 0)) + 1,
                        vstate=vstate,
                    )
                    if record.get("accepted", True):
                        previous_variance = float(np.real(np.asarray(record["energy"].get("Variance", np.nan))))
                    stage_meta["completed_iterations"] = iteration + 1
                    _update_stage_health_counters(stage_meta, record)
                    stage_meta["status"] = _phase_status(iteration + 1, target)
                    if record.get("accepted", True):
                        stage_meta["gamma"] = float(stage_meta.get("gamma", 0.0)) + interval
                    stage_meta["last_record"] = record
                    metadata["completed_total_iterations"] = int(metadata.get("completed_total_iterations", 0)) + 1
                    metadata["current_stage"] = stage.name
                    _append_jsonl(logs["training_log"], record)
                    _emit_iteration_nonfinite_warnings(
                        logs=logs,
                        metadata=metadata,
                        record=record,
                        stage_meta=stage_meta,
                        started_at=started_at,
                    )

                    if show_progress and (
                        record["phase_iteration"] == 1
                        or record["phase_iteration"] == target
                        or record["phase_iteration"] % print_every == 0
                    ):
                        print(_iteration_print_line(record))

                    interrupted = signal_request["signal"] is not None
                    if _should_checkpoint(
                        completed_iterations=stage_meta["completed_iterations"],
                        force=interrupted or stage_meta["completed_iterations"] == target,
                        last_checkpoint_iteration=last_checkpoint_iteration,
                        last_checkpoint_time=last_checkpoint_time,
                        settings=checkpoint_settings,
                    ):
                        metadata = _save_checkpoint(
                            checkpoint_settings=checkpoint_settings,
                            config_hash=config_hash,
                            config_payload=config_payload,
                            logs=logs,
                            metadata=metadata,
                            modules=modules,
                            output_path=output_path,
                            stage_name=stage.name,
                            status="interrupted" if interrupted else "running",
                            vstate=vstate,
                        )
                        last_checkpoint_time = time.time()
                        last_checkpoint_iteration = stage_meta["completed_iterations"]
                        _append_diagnostic(
                            logs["diagnostics_log"],
                            "checkpoint_saved",
                            started_at=started_at,
                            stage=stage.name,
                            completed_iterations=stage_meta["completed_iterations"],
                            completed_total_iterations=metadata["completed_total_iterations"],
                            reason="signal" if interrupted else "periodic",
                            params_file=logs["params_file"],
                        )

                    if interrupted:
                        signum = int(signal_request["signal"])
                        print("Checkpoint saved after interrupt request; exiting for Slurm requeue/resume.")
                        raise _InterruptedForRequeue(signum)

                stage_meta["status"] = "completed"
                metadata["current_stage"] = stage.name
                metadata = _save_checkpoint(
                    checkpoint_settings=checkpoint_settings,
                    config_hash=config_hash,
                    config_payload=config_payload,
                    logs=logs,
                    metadata=metadata,
                    modules=modules,
                    output_path=output_path,
                    stage_name=stage.name,
                    status="running",
                    vstate=vstate,
                )
                _append_diagnostic(
                    logs["diagnostics_log"],
                    "stage_complete",
                    started_at=started_at,
                    stage=stage.name,
                    completed_iterations=stage_meta["completed_iterations"],
                    target_iterations=target,
                    energy=(stage_meta.get("last_record") or {}).get("energy"),
                    bp=(stage_meta.get("last_record") or {}).get("bp"),
                )
                _emit_stage_completion_warnings(
                    logs=logs,
                    metadata=metadata,
                    stage_meta=stage_meta,
                    stage_name=stage.name,
                    started_at=started_at,
                )

            metadata["status"] = "training_completed"
            metadata["current_stage"] = None
            metadata = _save_checkpoint(
                checkpoint_settings=checkpoint_settings,
                config_hash=config_hash,
                config_payload=config_payload,
                logs=logs,
                metadata=metadata,
                modules=modules,
                output_path=output_path,
                stage_name=stages[-1].name,
                status="training_completed",
                vstate=vstate,
            )
            measurement_result = _run_measurements(
                L=int(L),
                logs=logs,
                measurement_settings=measurement_settings,
                params_file=logs["params_file"],
                started_at=started_at,
            )
            metadata["status"] = "completed"
            metadata = _save_checkpoint(
                checkpoint_settings=checkpoint_settings,
                config_hash=config_hash,
                config_payload=config_payload,
                logs=logs,
                metadata=metadata,
                modules=modules,
                output_path=output_path,
                stage_name=stages[-1].name,
                status="completed",
                vstate=vstate,
            )
            _write_final_summary(
                config_payload=config_payload,
                lattice_payload=lattice_payload,
                logs=logs,
                measurement_result=measurement_result,
                metadata=metadata,
                output_path=output_path,
            )
            _append_diagnostic(
                logs["diagnostics_log"],
                "run_complete",
                started_at=started_at,
                output_filename=output_path,
                completed_total_iterations=metadata["completed_total_iterations"],
            )
        except _InterruptedForRequeue as exc:
            pending_interrupt_signum = exc.signum
        except Exception as exc:
            if isinstance(exc, NonFiniteStepError):
                _emit_workflow_warning(
                    code="nonfinite_step_failed",
                    logs=logs,
                    metadata=metadata,
                    severity="error",
                    started_at=started_at,
                    source="adaptive_failure" if _is_truthy(evolution_settings.get("adaptive_substeps")) else "canonical_failure",
                    current_stage=metadata.get("current_stage"),
                    completed_total_iterations=metadata.get("completed_total_iterations"),
                    error=str(exc),
                    details=getattr(exc, "details", {}),
                )
            _append_diagnostic(
                logs["diagnostics_log"],
                "run_failed",
                started_at=started_at,
                current_stage=metadata.get("current_stage"),
                completed_total_iterations=metadata.get("completed_total_iterations"),
                error=str(exc),
            )
            raise
        finally:
            restore_signals()

    if pending_interrupt_signum is not None:
        signal.signal(pending_interrupt_signum, signal.SIG_DFL)
        os.kill(os.getpid(), pending_interrupt_signum)
        raise SystemExit(128 + pending_interrupt_signum)


run_non_abelian_topology = run
