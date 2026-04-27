import functools
import json

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from scipy import sparse

import flax.linen as nn

import global_vars as g
from netket.operator.spin import identity, sigmaz
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.jax import tree_cast
from netket.stats import statistics as mpi_statistics
from netket.utils import mpi
from tqdm import tqdm


@functools.lru_cache(maxsize=1)
def _ruby_jx_local_matrix():
    # NetKet's spin basis is ordered as [1, -1].
    states = 1 - 2 * np.array(
        [[(index >> shift) & 1 for shift in range(5, -1, -1)] for index in range(2**6)],
        dtype=np.int8,
    )
    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], dtype=np.int8)
    phase = np.ones(states.shape[0], dtype=np.float64)
    for i, j in edges:
        phase *= (1 + states[:, i] + states[:, j] - states[:, i] * states[:, j]) / 2

    cz_ring = sparse.diags(phase, format="csr")
    sx = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    x_ring = sx
    for _ in range(5):
        x_ring = sparse.kron(x_ring, sx, format="csr")

    return sparse.kron(cz_ring, x_ring, format="csr")


@functools.lru_cache(maxsize=1)
def _ruby_x_ring_local_matrix():
    sx = sparse.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))
    x_ring = sx
    for _ in range(5):
        x_ring = sparse.kron(x_ring, sx, format="csr")

    return x_ring

@nk.utils.struct.dataclass
class Gauge_trans(nk.sampler.rules.MetropolisRule):
    """
    Updates/flips plaquettes
    """
    plaquette_rate: float

    def transition(self, sampler, machine, parameters, state, key, sigmas):
        # Deduce the number of MCMC chains from input shape
        n_chains = sigmas.shape[0]
        key1, key2, key3 = jax.random.split(key, 3)

        # Pick random cluster index on every chain
        ind_cluster = jax.random.randint(key1, shape=(n_chains,), minval=0, maxval=g.N_plaquette)
        ind_single  = jax.random.randint(key2, shape=(n_chains,), minval=0, maxval=g.N)
        ind_which   = jax.random.uniform(key3, shape=(n_chains,))

        @jax.vmap
        def flip(sigma, ind_cluster, ind_single, ind_which):
            a = sigma.at[ind_single].set(-sigma[ind_single])
            u = g.X_list[ind_cluster]
            b = sigma.at[u].set(-sigma[u])
            s = jnp.where(ind_which < self.plaquette_rate, b, a)

            return s.astype(jnp.int64)

        sigmap = flip(sigmas, ind_cluster, ind_single, ind_which)

        return sigmap, None
    
@jax.jit
def activation2(x):
    return x/2 + x**2/4

@jax.jit
def activation4(x):
    return x/2 + x**2/4 - x**4/48


def Ruby_Hamiltonian(hi, Jz, Jx, use_cz_ring=True):

    ha = 0 * identity(hi)

    if Jz != 0:
        ha += Jz * sum([(sigmaz(hi, u[0]) @ sigmaz(hi, u[2]) @ sigmaz(hi, u[4])) for u in g.plaquette_list.tolist()])
        ha += Jz * sum([(sigmaz(hi, u[1]) @ sigmaz(hi, u[3]) @ sigmaz(hi, u[5])) for u in g.plaquette_list.tolist()])
    
    if Jx != 0:
        for i in range(g.N_plaquette):
            if use_cz_ring:
                local_matrix = _ruby_jx_local_matrix()
                support = np.concatenate((np.array(g.plaquette_list[i]), np.array(g.X_list[i]))).tolist()
            else:
                local_matrix = _ruby_x_ring_local_matrix()
                support = np.array(g.X_list[i]).tolist()
            s = nk.operator.LocalOperator(hi, local_matrix, support, dtype=float).to_jax_operator()
            ha += Jx * s

    return ha

class conv2(nn.Module):

    out_features: int
    ker_size: int = 2
    use_color_labels: bool = False
    dtype: type =  complex

    @nn.compact
    def __call__(self,x):
        
        size = x.shape
        N_kernel = self.ker_size**2
        mask = g.kernel2 if self.ker_size == 2 else g.kernel3
        offset_mask = mask[:, :, 0].astype(jnp.int32)
        if self.use_color_labels:
            color_mask = mask[:, :, 1].astype(jnp.int32)
            target_colors = color_mask[:, 0]
            W = jnp.pad(
                self.param(
                    'W',
                    nn.initializers.normal(stddev=0.5 / jnp.sqrt(N_kernel * size[-1])),
                    (g.n_unit_cell_colors, N_kernel, size[-1], self.out_features),
                    self.dtype,
                ),
                pad_width=((0, 0), (0, 1), (0, 0), (0, 0)),
                constant_values=0j,
            )
            kernel = W[color_mask, offset_mask]
        else:
            W = jnp.pad(
                self.param(
                    'W',
                    nn.initializers.normal(stddev=0.5 / jnp.sqrt(N_kernel * size[-1])),
                    (N_kernel, size[-1], self.out_features),
                    self.dtype,
                ),
                pad_width=((0, 1), (0, 0), (0, 0)),
                constant_values=0j,
            )
            kernel = W[offset_mask]
        y = jax.lax.dot_general(x, kernel, (( (1, 2), (0, 2)), ((), ())))

        if self.use_color_labels:
            b = self.param(
                'b',
                nn.initializers.zeros_init(),
                (g.n_unit_cell_colors, self.out_features),
                self.dtype,
            )
            y += b[target_colors][jnp.newaxis, :, :]
        else:
            b = self.param(
                'b',
                nn.initializers.zeros_init(),
                (self.out_features,),
                self.dtype,
            )
            y += b[jnp.newaxis, jnp.newaxis, :]
        
        return y

    
class deep_CNN(nn.Module):
    n_features: int
    n_layers: int = 2
    use_color_labels: bool = False

    @nn.compact
    def __call__(self, x):
        if self.n_layers < 1:
            raise ValueError("deep_CNN requires at least one layer.")

        y = conv2(
            out_features=self.n_features,
            ker_size=3,
            use_color_labels=self.use_color_labels,
            name="input_conv",
        )(x)

        for layer in range(1, self.n_layers):
            residual = y
            update = conv2(
                out_features=self.n_features,
                ker_size=3,
                use_color_labels=self.use_color_labels,
                name=f"residual_conv_{layer}",
            )(activation2(y))
            y = residual + update

        return y

def state_reposition(s):

    s2 = (1 + s) // 2
    
    x_shift = jnp.round(jnp.angle(jnp.sum(g.kx[None, :] * s2, axis = -1)) * g.L/(2*np.pi), 5)
    y_shift = jnp.round(jnp.angle(jnp.sum(g.ky[None, :] * s2, axis = -1)) * g.L/(2*np.pi), 5)

    x_shift = jnp.where(x_shift <= 0, g.L - jnp.ceil(-x_shift),  -jnp.ceil(-x_shift)).astype(int) % g.L
    y_shift = jnp.where(y_shift <= 0, g.L - jnp.ceil(-y_shift),  -jnp.ceil(-y_shift)).astype(int) % g.L

    dis = (y_shift * g.L + x_shift)
    rows = np.arange(s.shape[0])[:, None]

    return s[rows, g.translation_site[dis]], ((g.L - y_shift) % g.L) * g.L + (g.L - x_shift) % g.L

def phase2(x0):

    #return x0.reshape((x0.shape[0], x0.shape[1]//3, 3))

    x, shift = state_reposition(x0)
    u = (((1 - x) // 2) @ g.inverse_matrix.T) % 2
    res = ((1 - x) // 2 + u @ g.transform_matrix.T) % 2

    ### Minimize residual
    a = jnp.sum(res[:, :, jnp.newaxis] * g.transform_matrix[jnp.newaxis, :, :], axis = 1)
    b = jnp.where(a > 3, 1, 0)
    u = (u + b)%2
    res = ((1 - x) // 2 + u @ g.transform_matrix.T) % 2

    u = u[np.arange(x.shape[0])[:, None], g.translation_cell[shift]]
    res = res[np.arange(x.shape[0])[:, None], g.translation_site[shift]]

    out = jnp.concatenate((u[:, :, None], res.reshape((res.shape[0], res.shape[1]//3, 3))), axis = -1)
    return 2 * out - 1

class CNN_symmetric(nn.Module):

    rotation: bool = True
    use_small_point_group: bool = False
    use_color_labels: bool = False
    n_features: int = 12
    n_layers: int = 2

    @nn.compact
    
    def __call__(self, x):
        
        ##### Add symmetric copies  #############
        # Swaping sublattice ##################### 
        rotation_group = g.small_point_group if self.use_small_point_group else g.point_group
        if self.rotation:
            x = x[:, rotation_group].reshape((-1, g.N))

        u0 = phase2(x)
        y1 = deep_CNN(
            n_features=self.n_features,
            n_layers=self.n_layers,
            use_color_labels=self.use_color_labels,
        )(u0)

        u1 = jnp.stack((jnp.prod(x[:, g.left_triangles], axis = -1), jnp.prod(x[:, g.right_triangles], axis = -1)), axis = -1)
        y2 = deep_CNN(
            n_features=self.n_features,
            n_layers=self.n_layers,
            use_color_labels=self.use_color_labels,
        )(u1)

        out = jnp.sum(activation4(y1 + y2), axis = [1, 2]) / jnp.sqrt(3)
         
        if self.rotation:
            out = out.reshape(-1, rotation_group.shape[0])
            out = jnp.log(jnp.mean(jnp.exp(out), axis = -1))

        return out


def _regularized_eigensolve(matrix, vector):
    ev, V = jnp.linalg.eigh(matrix)
    rho = V.conj().T @ vector
    ev_inv = rho / ev
    filter = np.where(np.abs(ev / ev[-1]) > 1e-5, 1e1, np.where(np.abs(ev / ev[-1]) > 1e-8, 0.5, 0.01))
    ev_inv = np.where(np.abs(ev_inv) >= filter, filter * ev_inv / np.abs(ev_inv), ev_inv)
    return V @ ev_inv


def _sr_update_param_space(O, rhs, rcond):
    Sd = mpi.mpi_sum_jax(O.conj().T @ O)[0] + rcond * jnp.eye(O.shape[1], dtype=O.dtype)
    F = nk.stats.sum(O.conj() * rhs[:, None], axis=0)
    return _regularized_eigensolve(Sd, F)


def evolve(
    vstate,
    h0,
    nstep,
    dt,
    index_range,
    show_progress=True,
    log_path="log_data_L6_perturbed",
    op = None,
): 

    def single_update(vstate):  
        max_resample_attempts = 5

        for attempt in range(max_resample_attempts):
            E_loc = vstate.local_estimators(h0).reshape(-1)
            E = mpi_statistics(E_loc)
            ΔE_loc = (E_loc - E.mean)

            if E.Variance > 5 * data["E_Var"][-1]:

                vstate._sampler_state = None
                vstate.reset()
            else:
                break

        S = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(mode = "holomorphic"))
        O = S.O[:, index_range] / (g.N)
        rhs = ΔE_loc / (np.sqrt(vstate.n_samples) * g.N)
        update_index = _sr_update_param_space(O, rhs, rcond)
        has_nan = bool(jnp.any(jnp.isnan(update_index)))

        update = jnp.zeros((vstate.n_parameters), dtype = complex)
        update = update.at[index_range].set(update_index)

        y, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
        dw = tree_cast (reassemble(update), vstate.parameters)

        return dw, E, has_nan

    lr = -dt/nstep
    loop = tqdm(range(nstep)) if show_progress else range(nstep)
    n = 0
    rcond = 1e-12
    sparse_update_size = min(4000, vstate.n_parameters)
    refresh_interval = 10
    dynamic_sparse_update = index_range is None
    if dynamic_sparse_update:
       index_range = None
    else:
       index_range = jnp.asarray(index_range)
        
    data = {"gamma_n": [], "E": [], "E_Var": [1e3], "obs": []} 
    gamma_n = 0
    for _ in enumerate(loop):
        if dynamic_sparse_update and (n % refresh_interval == 0 or index_range is None):
           selected = np.random.choice(vstate.n_parameters, size=sparse_update_size, replace=False)
           index_range = jnp.asarray(np.sort(selected))

        old_pars = vstate.parameters
        k1, E, has_nan = single_update(vstate)
        if has_nan:
            vstate.parameters = jax.tree_util.tree_map(lambda x: x, old_pars)
            vstate.reset()
            n += 1
            if show_progress:
                loop.set_description("Skipped iteration due to NaN update")
            continue
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr * y , old_pars, k1)
        
        k2, E, has_nan = single_update(vstate)
        if has_nan:
            vstate.parameters = jax.tree_util.tree_map(lambda x: x, old_pars)
            vstate.reset()
            n += 1
            if show_progress:
                loop.set_description("Skipped iteration due to NaN update")
            continue
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5* lr * (y1+y2) , old_pars, k1, k2)
        
        n += 1
        gamma_n += -lr

        data["E"].append(jnp.real(E.Mean).item())
        data["E_Var"].append(jnp.real(E.Variance).item())
        if op is not None:
            data["obs"].append(jnp.real(vstate.expect(op).Mean).item())
        data["gamma_n"].append(gamma_n)


        if show_progress:
            loop.set_description(str(E) + " " + str(vstate.sampler_state.acceptance))
    
    if log_path is not None:
        with open(log_path, "w") as handle:
            json.dump(data, handle)
    return E
