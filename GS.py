import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "platform"
os.environ["nk.config.netket_experimental_disable_ode_jit"] = "True"
os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "True"

import argparse
import pickle

import flax
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from flax import traverse_util

from netket.operator.spin import sigmax, sigmaz

import cnn
import global_vars as g

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
    parser.add_argument('--n-features', type=int, default=6, help='Number of CNN feature channels')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of CNN convolution layers in the full model')
    parser.add_argument('--pretrain-n-features', type=int, default=None, help='Number of CNN feature channels in the first training stage (default: half of --n-features)')
    parser.add_argument('--n-discard-per-chain', type=int, default=10, help='Discarded samples per chain')
    parser.add_argument('--pretrain-nstep', type=int, default=150, help='Number of first-stage CNN training steps')
    parser.add_argument('--pretrain-dt', type=float, default=2.5, help='Total imaginary-time step used in the first CNN stage')
    parser.add_argument('--nstep', type=int, default=600, help='Number of optimization steps')
    parser.add_argument('--dt', type=float, default=10, help='Total imaginary-time step')
    parser.add_argument('--show-progress', action='store_true', help='Show tqdm optimization progress')

    args = parser.parse_args()
    if args.n_features < 1:
        parser.error('--n-features must be at least 1.')
    if args.n_layers < 1:
        parser.error('--n-layers must be at least 1.')
    if args.pretrain_n_features is not None and args.pretrain_n_features < 1:
        parser.error('--pretrain-n-features must be at least 1.')
    if args.pretrain_nstep < 0:
        parser.error('--pretrain-nstep must be non-negative.')
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


def build_output_paths(field_axis, hx, hz, hx_red=None, hz_red=None):
    suffix = f"hx_{format_field_value(hx)}_hz_{format_field_value(hz)}"
    if hx_red is not None:
        suffix += f"_hx_red_{format_field_value(hx_red)}"
    if hz_red is not None:
        suffix += f"_hz_red_{format_field_value(hz_red)}"
    axis_suffix = "single" if field_axis is None else field_axis
    params_dir = f"params_sweep_{axis_suffix}"
    return os.path.join("log_files", f"log_data_L6_{suffix}"), os.path.join(params_dir, f"params_L6_{suffix}")


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


TRANSFER_INIT_SCALE = 0.05


def _param_base_like(target_value, scale=1.0):
    return scale * jnp.array(target_value, dtype=target_value.dtype)


def lift_conv_params(old_value, target_value, scale=1.0):
    if old_value.shape == target_value.shape:
        return old_value.astype(target_value.dtype)

    def copy_overlap(source, target):
        slices = tuple(slice(0, min(source.shape[i], target.shape[i])) for i in range(source.ndim))
        return target.at[slices].set(source[slices])

    adapted = _param_base_like(target_value, scale=scale)

    if len(target_value.shape) == len(old_value.shape):
        return copy_overlap(old_value.astype(target_value.dtype), adapted)

    if len(target_value.shape) == len(old_value.shape) + 1 and target_value.shape[0] == g.n_unit_cell_colors:
        expanded = jnp.stack([old_value] * g.n_unit_cell_colors, axis=0).astype(target_value.dtype)
        return copy_overlap(expanded, adapted)

    raise ValueError(f"Cannot adapt parameter shape {old_value.shape} to {target_value.shape}")


def adapt_loaded_params(loaded_params, template_params, missing_scale=1.0):
    loaded_flat = traverse_util.flatten_dict(flax.core.unfreeze(loaded_params))
    template_flat = traverse_util.flatten_dict(flax.core.unfreeze(template_params))
    adapted_flat = {}

    for key, target_value in template_flat.items():
        if key not in loaded_flat:
            adapted_flat[key] = _param_base_like(target_value, scale=missing_scale)
            continue

        old_value = loaded_flat[key]
        if hasattr(old_value, "shape") and hasattr(target_value, "shape"):
            adapted_flat[key] = lift_conv_params(old_value, target_value, scale=missing_scale)
        else:
            adapted_flat[key] = old_value

    return traverse_util.unflatten_dict(adapted_flat)


def build_vstate(
    sampler,
    use_small_pg,
    use_color_labels,
    chunk_size,
    n_samples,
    n_discard_per_chain,
    n_features,
    n_layers,
):
    model = cnn.CNN_symmetric(
        rotation=True,
        use_small_point_group=use_small_pg,
        use_color_labels=use_color_labels,
        n_features=n_features,
        n_layers=n_layers,
    )
    return nk.vqs.MCState(
        sampler,
        model=model,
        chunk_size=chunk_size,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )


def load_params_if_available(path):
    if path is None or not os.path.exists(path):
        return None

    with open(path, "rb") as handle:
        return pickle.load(handle)


def initialize_vstate_params(vstate, loaded_params):
    if loaded_params is None:
        return
    vstate.parameters = adapt_loaded_params(loaded_params, vstate.parameters)


def get_pretrain_n_features(args):
    if args.pretrain_n_features is not None:
        return args.pretrain_n_features
    return max(1, args.n_features // 2)

def main():
    args = parse_arguments()
    use_small_pg = use_small_point_group(args)
    use_color_labels = use_small_pg
    n_chains, n_samples, chunk_size = args.n_chains, args.n_samples, args.chunk_size
    pretrain_n_features = get_pretrain_n_features(args)
    pretrain_n_layers = 1
    
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
    As = cnn.Ruby_Hamiltonian(hi, 0, 1/g.N_plaquette, use_cz_ring= True)
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

        log_path, params_path = build_output_paths(args.field_axis, hx, hz, hx_red, hz_red)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(params_path), exist_ok=True)

        ha = ha1 - hx * sum([sigmax(hi, i) for i in range(g.N)]) 
        ha += -hz * sum([sigmaz(hi, i) for i in range(g.N)])
        if hx_red is not None:
            ha += -hx_red * sum([sigmax(hi, int(i)) for i in np.array(g.site_list_r)])
        if hz_red is not None:
            ha += -hz_red * sum([sigmaz(hi, int(i)) for i in np.array(g.site_list_r)])

        print(f"Running hx={hx}, hz={hz}, hx_red={hx_red}, hz_red={hz_red}, small_pg={use_small_pg}")
        loaded_params = load_params_if_available(args.init_params)
        if loaded_params is not None:
            print(f"Loaded initial parameters from {args.init_params}")

        pretrain_vstate = build_vstate(
            sampler,
            use_small_pg,
            use_color_labels,
            chunk_size,
            n_samples,
            args.n_discard_per_chain,
            pretrain_n_features,
            pretrain_n_layers,
        )
        initialize_vstate_params(pretrain_vstate, loaded_params)
        print(
            f"Stage 1 CNN parameters: {pretrain_vstate.n_parameters} "
            f"(features={pretrain_n_features}, layers={pretrain_n_layers})"
        )
        print(pretrain_vstate.expect(ha))

        if args.pretrain_nstep > 0:
            cnn.evolve(
                pretrain_vstate,
                ha,
                args.pretrain_nstep,
                args.pretrain_dt,
                None,
                show_progress=args.show_progress,
                log_path=f"{log_path}_stage1",
                op=As,
            )

        vstate0 = build_vstate(
            sampler,
            use_small_pg,
            use_color_labels,
            chunk_size,
            n_samples,
            args.n_discard_per_chain,
            args.n_features,
            args.n_layers,
        )
        vstate0.parameters = adapt_loaded_params(
            pretrain_vstate.parameters,
            vstate0.parameters,
            missing_scale=TRANSFER_INIT_SCALE,
        )
        print(
            f"Stage 2 CNN parameters: {vstate0.n_parameters} "
            f"(features={args.n_features}, layers={args.n_layers})"
        )
        print(vstate0.expect(ha))
        cnn.evolve(
            vstate0,
            ha,
            args.nstep,
            args.dt,
            None,
            show_progress=args.show_progress,
            log_path=log_path,
            op = As,
        )
        with open(params_path, "wb") as handle:
            pickle.dump(vstate0.parameters, handle)
        print(f"Saved optimized parameters to {params_path}")

        
if __name__ == "__main__":
    main()
