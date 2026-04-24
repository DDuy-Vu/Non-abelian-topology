# Kagome spin liquid
import os
from pyexpat import features
from re import L
import sys
from traceback import print_list
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "platform"
os.environ["nk.config.netket_experimental_disable_ode_jit"] = "True"
os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "True"

from copy import copy
import json
import time
import math
import netket as nk
# import netket_fidelity as nkf
import netket.experimental as nkx
import numpy as np
import optax
import random
import shutil
import functools

from netket.operator.spin import sigmax, sigmaz, sigmay, identity
import jax.numpy as jnp
import scipy
import jax.tree_util
from matplotlib import pyplot as plt
import pickle 

import flax
from flax import struct
import flax.linen as nn
import netket.nn as nknn
from flax import traverse_util
from flax.core import freeze
import qutip as qtp
from tqdm import tqdm
import time

from typing import Any, Optional, Tuple
from netket.utils.types import PyTree, PRNGKeyT
from netket.sampler import MetropolisRule
from netket.stats import statistics as mpi_statistics
from functools import partial
import cnn
import global_vars as g
import gc


def _patch_numpy_asarray_copy_kw():
    try:
        np.asarray(0, copy=False)
    except (TypeError, ValueError):
        original_asarray = np.asarray

        def compat_asarray(a, dtype=None, order=None, *, copy=None, like=None):
            if like is not None:
                return np.array(a, dtype=dtype, order=order, like=like)
            return original_asarray(a, dtype=dtype, order=order)

        np.asarray = compat_asarray


_patch_numpy_asarray_copy_kw()


def _patch_jax_named_shape_pickle():
    try:
        from jax._src import core as jax_core
        original_update = jax_core.ShapedArray.update

        def compat_update(self, *args, **kwargs):
            kwargs.pop("named_shape", None)
            return original_update(self, *args, **kwargs)

        jax_core.ShapedArray.update = compat_update
    except Exception:
        pass


_patch_jax_named_shape_pickle()


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some global variables.')
    parser.add_argument('--L', type=int, required=True, help='Value for L')
    parser.add_argument('--field-axis', type=str, choices=['x', 'z', 'x_red', 'z_red'], help='Field axis to sweep')
    parser.add_argument('--field-values', type=float, nargs='+', help='Explicit field values for the sweep axis')
    parser.add_argument('--field-start', type=float, help='Starting field value for a linear sweep')
    parser.add_argument('--field-stop', type=float, help='Ending field value for a linear sweep')
    parser.add_argument('--field-num', type=int, help='Number of field values for a linear sweep')
    parser.add_argument('--hx', type=float, default=0.2, help='Base X-field strength')
    parser.add_argument('--hz', type=float, default=0.2, help='Base Z-field strength')
    parser.add_argument('--hx-red', type=float, default=None, help='Additional X-field strength on red sites only')
    parser.add_argument('--hz-red', type=float, default=None, help='Additional Z-field strength on red sites only')
    parser.add_argument('--init-params', type=str, default='init_params_L6', help='Parameter file reloaded before each field value')
    parser.add_argument('--n-chains', type=int, default=2**8, help='Number of MCMC chains')
    parser.add_argument('--n-samples', type=int, default=2**13, help='Number of MC samples per iteration')
    parser.add_argument('--chunk-size', type=int, default=2**14, help='Chunk size for model evaluation')
    parser.add_argument('--n-features', type=int, default=12, help='Number of CNN feature channels')
    parser.add_argument('--blur-sample-chunk', type=int, default=None, help='Chunk size over blurred source samples')
    parser.add_argument('--blur-conn-chunk', type=int, default=None, help='Chunk size over connected configurations in blurred sampling')
    parser.add_argument('--n-discard-per-chain', type=int, default=10, help='Discarded samples per chain')
    parser.add_argument('--pretrain-nstep', type=int, default=200, help='Number of mean-field pretraining steps')
    parser.add_argument('--pretrain-dt', type=float, default=2.0, help='Total mean-field pretraining step')
    parser.add_argument('--nstep', type=int, default=300, help='Number of optimization steps')
    parser.add_argument('--dt', type=float, default=5.0, help='Total imaginary-time step')
    parser.add_argument('--show-progress', action='store_true', help='Show tqdm optimization progress')

    args = parser.parse_args()
    if args.field_axis is None:
        has_explicit_sweep = args.field_values is not None or any(
            value is not None for value in (args.field_start, args.field_stop, args.field_num)
        )
        if has_explicit_sweep:
            parser.error('--field-axis is required when sweep values are provided.')
    g.L = args.L
    g.update_globals()
    return args


def format_field_value(value):
    return f"{value:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")


def build_output_paths(hx, hz, hx_red = None, hz_red = None):
    suffix = f"hx_{format_field_value(hx)}_hz_{format_field_value(hz)}"
    if hx_red is not None:
        suffix += f"_hx_red_{format_field_value(hx_red)}"
    if hz_red is not None:
        suffix += f"_hz_red_{format_field_value(hz_red)}"
    return os.path.join("log_files", f"log_data_L6_{suffix}"), f"params_L6_{suffix}"


def get_sweep_values(args):
    if args.field_values is not None:
        return list(args.field_values)

    range_args = [args.field_start, args.field_stop, args.field_num]
    if any(value is not None for value in range_args):
        if not all(value is not None for value in range_args):
            raise ValueError("--field-start, --field-stop, and --field-num must be provided together.")
        return np.linspace(args.field_start, args.field_stop, args.field_num).tolist()

    if args.field_axis is not None:
        if args.field_axis == 'x':
            return [args.hx]
        if args.field_axis == 'z':
            return [args.hz]
        if args.field_axis == 'x_red':
            return [0.0 if args.hx_red is None else args.hx_red]
        return [0.0 if args.hz_red is None else args.hz_red]

    return [None]


def use_small_point_group(args):
    return args.hx_red is not None or args.hz_red is not None or args.field_axis in ('x_red', 'z_red')


def lift_conv_params(old_value, target_value):
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


def adapt_loaded_params(loaded_params, template_params):
    loaded_flat = traverse_util.flatten_dict(flax.core.unfreeze(loaded_params))
    template_flat = traverse_util.flatten_dict(flax.core.unfreeze(template_params))
    adapted_flat = {}

    for key, target_value in template_flat.items():
        if key not in loaded_flat:
            raise KeyError(f"Missing parameter {key} in loaded parameter tree")

        old_value = loaded_flat[key]
        if hasattr(old_value, "shape") and hasattr(target_value, "shape"):
            adapted_flat[key] = lift_conv_params(old_value, target_value)
        else:
            adapted_flat[key] = old_value

    return traverse_util.unflatten_dict(adapted_flat)


def build_vstate(sampler, use_small_pg, use_color_labels, chunk_size, n_samples, n_discard_per_chain, n_features, mean_field_only=False):
    model = (
        cnn.MeanField_symmetric(
            rotation=True,
            use_small_point_group=use_small_pg,
            use_color_labels=use_color_labels,
        )
        if mean_field_only
        else cnn.CNN_symmetric(
            rotation=True,
            use_small_point_group=use_small_pg,
            use_color_labels=use_color_labels,
            n_features=n_features,
        )
    )
    return nk.vqs.MCState(
        sampler,
        model=model,
        chunk_size=chunk_size,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )


def transplant_mean_fields(mean_field_params, full_params):
    mean_field_params = flax.core.unfreeze(mean_field_params)
    full_params = flax.core.unfreeze(full_params)

    for key in ("mean_field0", "mean_field1"):
        if key in mean_field_params:
            full_params[key] = mean_field_params[key]

    return full_params


def run_netket_vmc(vstate, hamiltonian, n_iter, optimizer, log_path, show_progress, variance_stop_threshold=None):
    sr = nk.optimizer.SR(
        qgt=nk.optimizer.qgt.QGTJacobianDense,
        diag_shift=1.0e-3,
        mode="complex",
    )
    driver = nk.driver.VMC(
        hamiltonian,
        optimizer,
        variational_state=vstate,
        preconditioner=sr,
    )
    callback = lambda step, log_data, driver: True
    if variance_stop_threshold is not None:
        def callback(step, log_data, driver):
            return float(jnp.real(driver.energy.Variance)) >= variance_stop_threshold

    driver.run(
        n_iter=n_iter,
        out=log_path,
        show_progress=show_progress,
        callback=callback,
    )

def main():
    
    args = parse_arguments()
    use_small_pg = use_small_point_group(args)
    use_color_labels = use_small_pg
    n_chains, n_samples, chunk_size = args.n_chains, args.n_samples, args.chunk_size
    
    if use_small_pg and g.site_list_r is None:
        raise ValueError("Red-site fields require a 3-colorable lattice, i.e. L divisible by 3.")
    hi = nk.hilbert.Spin(s=1/2, N=g.N, inverted_ordering = False)
    print(args, g.N)

    @nk.hilbert.random.random_state.dispatch
    def random_state(hilb : nk.hilbert.Spin, key, size : int, *, dtype):
        out = jnp.ones((size, g.N), dtype = dtype)
        rs = jax.random.randint(key, shape=(size, g.transform_matrix.shape[1]), minval= 0, maxval= 2)
        @jax.vmap
        def flip(sigma, index):
            b = g.transform_matrix @ index % 2
            s = sigma * (-1) ** b
            return s.astype(dtype)
        
        outp = flip(out, rs)
        return outp
    

    rules =  cnn.Gauge_trans(0.5)
    sampler = nk.sampler.MetropolisSampler(hi, rules, sweep_size = 3 * g.N // 4, n_chains=n_chains, reset_chains=True, dtype=jnp.int64)
    os.makedirs("log_files", exist_ok=True)
    ha1 = cnn.Ruby_Hamiltonian(hi, -1, -1, use_cz_ring= True)
    print(f"MC settings: n_chains={n_chains}, n_samples={n_samples}, chunk_size={chunk_size}")

    field_values = get_sweep_values(args)
    for field_value in field_values:
        hx, hz = args.hx, args.hz
        hx_red = args.hx_red
        hz_red = args.hz_red
        if args.field_axis == 'x':
            hx = field_value
        elif args.field_axis == 'z':
            hz = field_value
        elif args.field_axis == 'x_red':
            hx_red = field_value
        elif args.field_axis == 'z_red':
            hz_red = field_value

        log_path, params_path = build_output_paths(hx, hz, hx_red, hz_red)
        if os.path.exists(log_path):
            print(f"Skipping hx={hx}, hz={hz}, hx_red={hx_red}, hz_red={hz_red}: found existing log {log_path}")
            continue

        ha = ha1 - hx * sum([sigmax(hi, i) for i in range(g.N)]) 
        ha += -hz * sum([sigmaz(hi, i) for i in range(g.N)])
        if hx_red is not None:
            ha += -hx_red * sum([sigmax(hi, int(i)) for i in np.array(g.site_list_r)])
        if hz_red is not None:
            ha += -hz_red * sum([sigmaz(hi, int(i)) for i in np.array(g.site_list_r)])

        print(f"Running hx={hx}, hz={hz}, hx_red={hx_red}, hz_red={hz_red}, small_pg={use_small_pg}")
        mean_field_vstate = build_vstate(
            sampler,
            use_small_pg,
            use_color_labels,
            chunk_size,
            n_samples,
            args.n_discard_per_chain,
            args.n_features,
            mean_field_only=True,
        )
        print(f"Mean-field parameters: {mean_field_vstate.n_parameters}")
        print(mean_field_vstate.expect(ha))

        # cnn.evolve(
        #     mean_field_vstate,
        #     ha,
        #     args.pretrain_nstep,
        #     args.pretrain_dt,
        #     None,
        #     show_progress=args.show_progress,
        #     log_path=log_path + "_pretrain",
        # )
        run_netket_vmc(
            mean_field_vstate,
            ha,
            args.pretrain_nstep,
            nk.optimizer.Sgd(learning_rate=abs(args.pretrain_dt / max(args.pretrain_nstep, 1))),
            None,
            args.show_progress,
            variance_stop_threshold=40.0,
        )

        vstate0 = build_vstate(
            sampler,
            use_small_pg,
            use_color_labels,
            chunk_size,
            n_samples,
            args.n_discard_per_chain,
            args.n_features,
            mean_field_only=False,
        )
        print(f"Full-model parameters: {vstate0.n_parameters}")
        vstate0.parameters = jax.tree_util.tree_map(
            lambda x: x,
            transplant_mean_fields(mean_field_vstate.parameters, vstate0.parameters),
        )

        del mean_field_vstate
        gc.collect()

        print(vstate0.expect(ha))
        cnn.evolve(
            vstate0,
            ha,
            args.nstep,
            args.dt,
            None,
            show_progress=args.show_progress,
            log_path=log_path,
        )
        # second_stage_base_lr = abs(args.dt / max(args.nstep, 1))
        # second_stage_schedule = optax.piecewise_constant_schedule(
        #     init_value=0.1 * second_stage_base_lr,
        #     boundaries_and_scales={max(args.nstep // 2, 1): 10.0},
        # )
        # second_stage_optimizer = optax.adamw(
        #     learning_rate=second_stage_schedule,
        #     weight_decay=1.0e-4,
        # )
        # run_netket_vmc(
        #     vstate0,
        #     ha,
        #     args.nstep,
        #     second_stage_optimizer,
        #     log_path,
        #     args.show_progress,
        # )
        pickle.dump(vstate0.parameters, open(params_path, "wb"))
        print(f"Saved optimized parameters to {params_path}")

        
if __name__ == "__main__":
    main()
