# Non-Abelian Training Wrapper

`workflow.run(...)` is a callable version of the original `GS.py` training path with resumable checkpoints, JSONL diagnostics, Slurm-friendly output files, color-resolved fields, optional post-training D4 measurements, and automatic NetKet sharding when multiple GPUs are visible.

By default, state-changing settings match the clean `GS.py`: staged training is on, pretrain is `150` steps at `dt=2.5` with one layer, main training is `600` steps at `dt=10`, `n_features=6`, `n_layers=3`, `Jx=Jz=-1`, `hx=hz=0.2`, `seed=None`, `initial_params_path="init_params_L6"`, and the original SR/Heun update is used. Wrapper-only conveniences such as progress printing, checkpoints, measurement output, and auto-sharding can be left enabled without changing the default calculation.

Useful switches:

- `task_parameters` override any default passed to `workflow.run`.
- `runtime_kwargs`: set `use_gpu`; omit `use_sharding` to auto-shard when multiple GPUs are visible.
- `measurement_kwargs.enabled`: controls post-training D4 checkpoint measurements.
- `evolution_kwargs.adaptive_substeps`: optional robust update mode that retries non-finite steps with resampling and smaller substeps.
- `blurred_sampling_kwargs.enabled`: optional replacement for Metropolis samples in wrapper energy/SR/Bp estimators. It is off by default; `q` is the connected-move probability.

The wrapper is standalone inside `PROJECT_ROOT`: it imports project modules from `Non-abelian-topology/` and uses the local `wrapper/cluster_jobs.py` submit helper. It does not import sibling repos.
