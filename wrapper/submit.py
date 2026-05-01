#!/usr/bin/env python3
"""Editable Slurm submit entry point for ``wrapper.workflow.run``.

Run this from the scratch directory that should own the output files:

    python /path/to/Non-abelian-topology/wrapper/submit.py

Most day-to-day changes should be made in ``FIELD_POINTS`` and ``BASE_TASK``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_ROOT = Path(__file__).resolve().parent
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

import cluster_jobs  # noqa: E402


def _tag(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{float(value):+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def _output_name(params: dict) -> str:
    pieces = [f"L{params['L']}", f"hx{_tag(params.get('hx'))}", f"hz{_tag(params.get('hz'))}"]
    for key in (
        "hy",
        "hx_red",
        "hy_red",
        "hz_red",
        "hx_green",
        "hy_green",
        "hz_green",
        "hx_blue",
        "hy_blue",
        "hz_blue",
    ):
        value = params.get(key)
        if value is not None and float(value) != 0.0:
            pieces.append(f"{key}{_tag(value)}")
    return "_".join(pieces) + ".json"


# Edit this list for sweeps.  Color fields are additive on top of the global
# fields, so hx=0.0 and hx_red=0.1 means red sites see hx=0.1.
FIELD_POINTS = [
    {"hx": 0.0, "hz": 0.0},
    *({"hx": 0.0, "hz": 0.0, "hx_red": round(0.05 * index, 2)} for index in range(1, 21)),
    *({"hx": 0.0, "hz": 0.0, "hz_red": round(0.05 * index, 2)} for index in range(1, 21)),
]


BASE_TASK = {
    "L": 6,
    "Jx": -1.0,
    "Jz": -1.0,
    "staged_training": True,
    "pretrain_n_iter": 150,
    "pretrain_dt": 2.5,
    "pretrain_n_features": None,
    "pretrain_n_layers": 1,
    "n_iter": 600,
    "dt": 10.0,
    "n_features": 6,
    "n_layers": 3,
    "sampler_kwargs": {
        "n_chains": 2**8,
    },
    "vstate_kwargs": {
        "n_samples": 2**13,
        "chunk_size": 2**14,
        "n_discard_per_chain": 10,
    },
    "evolution_kwargs": {
        "active_parameter_count": 4000,
        "active_refresh_interval": 10,
        # False matches the original GS.py update rule.  Set True to retry
        # non-finite updates by resampling and halving the attempted step.
        "adaptive_substeps": False,
        "max_step_retries": 8,
        "min_step_fraction": 1e-4,
        "variance_resample_attempts": 5,
    },
    "blurred_sampling_kwargs": {
        # False preserves the original GS.py Metropolis-based evolution.
        "enabled": False,
        "q": 0.2,
        "kernel": "general",
        "chunk_size": None,
    },
    "checkpoint_kwargs": {
        "every_iterations": 1,
        "interval_seconds": 30 * 60,
    },
    "diagnostics_kwargs": {
        "print_every": 1,
    },
    "measurement_kwargs": {
        "enabled": True,
        "n_samples": 2048,
        "n_chains": 32,
        "n_discard_per_chain": 10,
        "chunk_size": 512,
        "eval_batch": 512,
        "string_set": "paper",
        "save_arrays": False,
    },
    "runtime_kwargs": {
        "use_gpu": True,
        # Sharding is enabled automatically when multiple GPUs are visible.
    },
    "show_progress": True,
}


SLURM_REQUIREMENTS = {
    "mem": "32G",
    "time": "72:00:00",
    "nodes": 1,
    "ntasks": 1,
    "cpus-per-task": 8,
    "gres": "gpu:1",
    "partition": "yao_gpu,gpu,gpu_h200,gpu_requeue",
    "mail-user": "vincent_liu@berkeley.edu",
    # Uncomment for H200-only runs.
    # "constraint": "h200",
}


def build_task_parameters() -> list[dict]:
    tasks = []
    for point in FIELD_POINTS:
        params = deepcopy(BASE_TASK)
        params.update(point)
        params["output_filename"] = _output_name(params)
        tasks.append(params)
    return tasks


def build_config(jobname: str) -> dict:
    return {
        "jobname": jobname,
        "task": {
            "type": "PythonFunctionCall",
            "module": "wrapper.workflow",
            "function": "run",
        },
        "task_parameters": build_task_parameters(),
        "requirements_slurm": SLURM_REQUIREMENTS,
        "options": {
            "cores_per_task": "$SLURM_CPUS_PER_TASK",
            "cluster_jobs_module": str(WRAPPER_ROOT / "cluster_jobs.py"),
            "python_executable": sys.executable,
            "environment_setup": f'export PYTHONPATH="{PROJECT_ROOT}:$PYTHONPATH"',
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobname", default="nonabelian")
    parser.add_argument("--dry-run", action="store_true", help="Write config/script but do not sbatch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = cluster_jobs.SlurmJob(**build_config(args.jobname))
    if args.dry_run:
        script = job.prepare_submit()
        print(f"prepared {script} with {job.N_tasks} task(s)")
        return
    job.submit()


if __name__ == "__main__":
    main()
