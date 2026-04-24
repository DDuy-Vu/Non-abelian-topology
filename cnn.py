import math
import netket as nk
import numpy as np
import random
import shutil
import functools

from netket.operator.spin import sigmax, sigmaz, sigmay, identity
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from scipy import sparse
import jax.tree_util

import flax
from flax import struct, nnx
import flax.linen as nn
import netket.nn as nknn
from flax import traverse_util
from flax.core import freeze

import global_vars as g
from netket import jax as nkjax
from netket.utils import HashableArray, mpi
from netket.stats import Stats, statistics as mpi_statistics
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.optimizer.qgt.qgt_jacobian import QGTJacobian_DefaultConstructor
from netket.jax import tree_cast
import netket.experimental as nkx
import gc
from tqdm import tqdm
from functools import partial
from scipy.stats import gmean
import json

def CZ(hi, site1, site2):
    # Controlled-Z on spin eigenvalues z = +/-1:
    # CZ = (I + Z_i + Z_j - Z_i Z_j) / 2
    return (identity(hi) + sigmaz(hi, site1) + sigmaz(hi, site2) - (sigmaz(hi, site1) @ sigmaz(hi, site2))) / 2


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

def infidel_and_grad(vstate, target):

    afun = vstate._apply_fun
    sigma = vstate.hilbert.all_states()
    model_state = vstate.model_state
    offset = vstate.log_value(g.sstart)

    def expect_fun(params):
        log_value = afun({"params": params, **model_state}, sigma) - offset
        log_value = jnp.where(jnp.real(log_value) < -100, -100, log_value)
        state = jnp.exp(log_value)
        state = state / jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))
        return 1 - jnp.abs(state.T.conj() @ target) ** 2

    F, F_vjp_fun = nkjax.vjp(expect_fun, vstate.parameters, conjugate=True)

    F_grad = F_vjp_fun(jnp.ones_like(F))[0]
    F_grad = jax.tree.map(lambda x: mpi.mpi_mean_jax(x)[0], F_grad)

    return F, F_grad


pi = 3.14159268

class conv(nn.Module):
    out_features: int
    W_scale: float = 1.0
    b_scale: float = 1.0
    kersize: int = 2
    use_color_labels: bool = False
    dtype: type =  complex

    @nn.compact
    def __call__(self,x):
        
        size = x.shape
        N_kernel = self.kersize**2
        mask = g.kernel2 if self.kersize == 2 else g.kernel3
        offset_mask = mask[:, :, 0].astype(jnp.int32)
        if self.use_color_labels:
            color_mask = mask[:, :, 1].astype(jnp.int32)
            target_colors = color_mask[:, 0]
            W = self.param(
                'W',
                nn.initializers.normal(stddev=self.W_scale * 0.5 / jnp.sqrt(N_kernel * (size[2] + self.out_features))),
                (g.n_unit_cell_colors, N_kernel, size[2], self.out_features),
                self.dtype,
            )
            W = jnp.pad(W, pad_width=((0, 0), (0, 1), (0, 0), (0, 0)), constant_values=0j)
            kernel = W[color_mask, offset_mask]
        else:
            W = self.param(
                'W',
                nn.initializers.normal(stddev=self.W_scale * 0.5 / jnp.sqrt(N_kernel * (size[2] + self.out_features))),
                (N_kernel, size[2], self.out_features),
                self.dtype,
            )
            W = jnp.pad(W, pad_width=((0, 1), (0, 0), (0, 0)), constant_values=0j)
            kernel = W[offset_mask]
        y = jax.lax.dot_general(x, kernel, (((1, 2), (0, 2)), ((), ())))
        
        if self.b_scale > 1e-5:
            if self.use_color_labels:
                b = self.param(
                    'b',
                    nn.initializers.normal(stddev=self.b_scale * 0.5 / jnp.sqrt(N_kernel * (size[2] + self.out_features))),
                    (g.n_unit_cell_colors, self.out_features),
                    self.dtype,
                )
                y += b[target_colors][jnp.newaxis, :, :]
            else:
                b = self.param(
                    'b',
                    nn.initializers.normal(stddev=self.b_scale * 0.5 / jnp.sqrt(N_kernel * (size[2] + self.out_features))),
                    (self.out_features,),
                    self.dtype,
                )
                y += b[jnp.newaxis, jnp.newaxis, :]
        
        return y
    
    
class conv2(nn.Module):

    out_features: int
    W_scale: float = 1.0
    b_scale: float = 1.0
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
                    nn.initializers.normal(stddev=self.W_scale * 0.5 / jnp.sqrt(N_kernel * size[-1])),
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
                    nn.initializers.normal(stddev=self.W_scale * 0.5 / jnp.sqrt(N_kernel * size[-1])),
                    (N_kernel, size[-1], self.out_features),
                    self.dtype,
                ),
                pad_width=((0, 1), (0, 0), (0, 0)),
                constant_values=0j,
            )
            kernel = W[offset_mask]
        y = jax.lax.dot_general(x, kernel, (( (1, 2), (0, 2)), ((), ())))

        if self.b_scale > 1e-5:
            if self.use_color_labels:
                b = self.param(
                    'b',
                    nn.initializers.normal(stddev=self.b_scale * 0.5 / jnp.sqrt(N_kernel * self.out_features)),
                    (g.n_unit_cell_colors, self.out_features),
                    self.dtype,
                )
                y += b[target_colors][jnp.newaxis, :, :]
            else:
                b = self.param(
                    'b',
                    nn.initializers.normal(stddev=self.b_scale * 0.5 / jnp.sqrt(N_kernel * self.out_features)),
                    (self.out_features,),
                    self.dtype,
                )
                y += b[jnp.newaxis, jnp.newaxis, :]
        
        return y

    
class deep_CNN(nn.Module):
    n_features: int
    use_color_labels: bool = False
    @nn.compact
    
    def __call__(self, x):

        ### Convolutional NN
        y1b = conv2(out_features = self.n_features, ker_size = 3, use_color_labels = self.use_color_labels)(x)
        y1b = conv2(out_features = self.n_features, ker_size = 3, use_color_labels = self.use_color_labels)(activation2(y1b))
        y1b = conv2(out_features = self.n_features, ker_size = 3, use_color_labels = self.use_color_labels)(activation2(y1b))

        return y1b
    
@jax.vmap
def product_znot(u):
    a =  (1 + u[0]+u[1] - u[0]*u[1]) * (1 + u[1]+u[2] - u[1]*u[2]) * (1 + u[2]+u[3] - u[2]*u[3]) / 2**3
    a *= (1 + u[3]+u[4] - u[3]*u[4]) * (1 + u[4]+u[5] - u[4]*u[5]) * (1 + u[5]+u[0] - u[5]*u[0]) / 2**3

    return (1 - a) // 2

def phase(x):

    u = (((1 - x) // 2) @ g.inverse_matrix.T) % 2

    # a = jnp.where(jnp.sum(u[:, g.X_list_r], axis = -1) <= g.N_plaquette / 3, 0, 1)
    # u = u.at[:,g.X_list_r].set(a[:, jnp.newaxis] + (-1)**a[:, jnp.newaxis] * u[:, g.X_list_r])

    # a = jnp.where(jnp.sum(u[:, g.X_list_b], axis = -1) <= g.N_plaquette / 3, 0, 1)
    # u = u.at[:,g.X_list_b].set(a[:, jnp.newaxis] + (-1)**a[:, jnp.newaxis] * u[:, g.X_list_b])

    # a = jnp.where(jnp.sum(u[:, g.X_list_g], axis = -1) <= g.N_plaquette / 3, 0, 1)
    # u = u.at[:,g.X_list_g].set(a[:, jnp.newaxis] + (-1)**a[:, jnp.newaxis] * u[:, g.X_list_g])

    # C = (-1)**jnp.sum( (u[:, g.adjacent_matrix])[:, :, :, jnp.newaxis] * g.path_matrix[jnp.newaxis, :, :, :], axis = 2 )
    # y = product_znot(C.reshape((-1, 6))).reshape((-1, g.N_plaquette))
    # out = jnp.sum( y * u, axis = 1)

    return u

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

    return jnp.concatenate((u[:, :, None], res.reshape((res.shape[0], res.shape[1]//3, 3))), axis = -1)

class mean_field(nn.Module):
    use_color_labels: bool = False

    @nn.compact
    def __call__(self, x):
        if self.use_color_labels and g.n_unit_cell_colors > 1:
            alpha = self.param(
                'alpha',
                nn.initializers.normal(stddev=0.01),
                (g.n_unit_cell_colors, x.shape[-1]),
                complex,
            )
            color_labels = g.plaquette_color_labels.astype(jnp.int32)
            return jnp.sum(alpha[color_labels][None, :, :] * x, axis=[1, 2])

        alpha = self.param(
            'alpha',
            nn.initializers.normal(stddev=0.01),
            (x.shape[-1],),
            complex,
        )
        return jnp.sum(alpha[None, None, :] * x, axis=[1, 2])


class MeanField_symmetric(nn.Module):

    rotation: bool = True
    use_small_point_group: bool = False
    use_color_labels: bool = False

    @nn.compact
    def __call__(self, x):

        rotation_group = g.small_point_group if self.use_small_point_group else g.point_group
        if self.rotation:
            x = x[:, rotation_group].reshape((-1, g.N))

        u0 = phase2(x)
        out = mean_field(name="mean_field0", use_color_labels=False)(u0)

        u1 = jnp.stack((jnp.prod(x[:, g.left_triangles], axis = -1), jnp.prod(x[:, g.right_triangles], axis = -1)), axis = -1)
        out += mean_field(name="mean_field1", use_color_labels=False)(u1)

        if self.rotation:
            out = out.reshape(-1, rotation_group.shape[0])
            out = jnp.log(jnp.mean(jnp.exp(out), axis = -1))

        return out

class CNN_symmetric(nn.Module):

    rotation: bool = True
    use_small_point_group: bool = False
    use_color_labels: bool = False
    n_features: int = 12

    @nn.compact
    
    def __call__(self, x):
        
        ##### Add symmetric copies  #############
        # Swaping sublattice ##################### 
        rotation_group = g.small_point_group if self.use_small_point_group else g.point_group
        if self.rotation:
            x = x[:, rotation_group].reshape((-1, g.N))

        u0 = phase2(x)
        out = mean_field(name="mean_field0", use_color_labels=False)(u0)
        y1 =  deep_CNN(n_features=self.n_features, use_color_labels=self.use_color_labels)(u0)

        u1 = jnp.stack((jnp.prod(x[:, g.left_triangles], axis = -1), jnp.prod(x[:, g.right_triangles], axis = -1)), axis = -1)
        out += mean_field(name="mean_field1", use_color_labels=False)(u1)
        y2 =  deep_CNN(n_features=self.n_features, use_color_labels=self.use_color_labels)(u1)

        out += jnp.sum(activation4(y1 + y2), axis = [1, 2]) / jnp.sqrt(3)
         
        if self.rotation:
            out = out.reshape(-1, rotation_group.shape[0])
            out = jnp.log(jnp.mean(jnp.exp(out), axis = -1))

        return out


def blurred_sample(
    x,
    key,
    variables,
    q,
    apply_fn,
    op,
    chunk_size,
    sample_chunk_size=None,
    conn_chunk_size=None,
    diagonal_mels=True,
):
    max_conn_size = int(op.max_conn_size)
    if sample_chunk_size is None:
        sample_chunk_size = 1 if chunk_size is None else max(1, int(chunk_size) // max_conn_size)
    else:
        sample_chunk_size = max(1, int(sample_chunk_size))

    if conn_chunk_size is None:
        conn_chunk_size = max_conn_size if chunk_size is None else max(1, int(chunk_size))
    else:
        conn_chunk_size = max(1, int(conn_chunk_size))

    batch_size = x.shape[0]
    rng = jax.random.uniform(key, shape=(batch_size, 2))

    def apply_model_chunked(sigma):
        if sigma.ndim == 1:
            return apply_fn(variables, sigma)
        sigma_shape = sigma.shape
        sigma_flat = sigma.reshape((-1, sigma.shape[-1]))
        if sigma_flat.shape[0] <= conn_chunk_size:
            values = apply_fn(variables, sigma_flat)
        else:
            values = nkjax.apply_chunked(
                lambda sigma_chunk: apply_fn(variables, sigma_chunk),
                in_axes=0,
                chunk_size=conn_chunk_size,
                axis_0_is_sharded=False,
            )(sigma_flat)
        return values.reshape(sigma_shape[:-1])

    if batch_size <= sample_chunk_size:
        sample_chunk_size = batch_size

    blurred_chunks = []
    weight_chunks = []
    eloc_chunks = []
    for start in range(0, batch_size, sample_chunk_size):
        stop = min(start + sample_chunk_size, batch_size)
        sigma_chunk = x[start:stop]
        rng_chunk = rng[start:stop]
        u_stay = rng_chunk[:, 0]
        u_conn = rng_chunk[:, 1]

        sigma_conn, _ = op.get_conn_padded(sigma_chunk)
        n_conn = sigma_conn.shape[-2] - 1 if diagonal_mels else sigma_conn.shape[-2]
        idx = jnp.floor(u_conn * n_conn).astype(jnp.int32)
        proposal_index = idx + 1 if diagonal_mels else idx
        proposal = sigma_conn[jnp.arange(sigma_chunk.shape[0]), proposal_index]
        sigma_p = jnp.where(u_stay[:, None] > q, sigma_chunk, proposal)

        sigma_p_conn, mels = op.get_conn_padded(sigma_p)
        logpsi_sigma = apply_model_chunked(sigma_p)
        logpsi_conn = apply_model_chunked(sigma_p_conn)

        logp_sigma = 2.0 * jnp.real(logpsi_sigma)
        logp_conn = 2.0 * jnp.real(logpsi_conn)
        log_term_stay = jnp.log1p(-q) + logp_sigma
        if diagonal_mels:
            log_term_moves = jnp.log(q) - jnp.log(n_conn) + jsp.special.logsumexp(logp_conn[:, 1:], axis=-1)
        else:
            log_term_moves = jnp.log(q) - jnp.log(n_conn) + jsp.special.logsumexp(logp_conn, axis=-1)

        log_w_bridge = jsp.special.logsumexp(jnp.stack((log_term_stay, log_term_moves), axis=-1), axis=-1)
        weight_chunk = jnp.exp(logp_sigma - log_w_bridge)
        eloc_chunk = jnp.sum(mels * jnp.exp(logpsi_conn - logpsi_sigma[:, None]), axis=-1)

        blurred_chunks.append(sigma_p)
        weight_chunks.append(weight_chunk)
        eloc_chunks.append(eloc_chunk)

    return (
        jnp.concatenate(blurred_chunks, axis=0),
        jnp.concatenate(weight_chunks, axis=0),
        jnp.concatenate(eloc_chunks, axis=0),
    )


def _qgt_dense_components(S, index_range):
    O = S.O
    if S.mode == "complex":
        O = O.reshape(-1, 2, O.shape[-1])
        O = O[:, 0, :] + 1j * O[:, 1, :]

    O = O[:, index_range]
    Sd = S.to_dense()[jnp.ix_(index_range, index_range)]

    return O, Sd


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
): 

    def single_update(vstate):  

        E_loc = vstate.local_estimators(h0).reshape(-1)
        E = mpi_statistics(E_loc)
        ΔE_loc = (E_loc - E.mean)

        init_indices = np.random.choice(jnp.arange(vstate.n_samples)[jnp.abs(ΔE_loc) <= max(jnp.sqrt(E.Variance), 1.0)], vstate.sampler.n_chains, replace = False)
        vstate.sampler_state = vstate.sampler_state.replace(σ = vstate.samples.reshape(-1, g.N)[init_indices, :])

        S = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(mode = "holomorphic"))
        O = S.O[:, index_range] / (g.N)
        rhs = ΔE_loc / (np.sqrt(vstate.n_samples) * g.N)
        update_index = _sr_update_param_space(O, rhs, rcond)

        update = jnp.zeros((vstate.n_parameters), dtype = complex)
        update = update.at[index_range].set(update_index)

        y, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
        dw = tree_cast (reassemble(update), vstate.parameters)

        return dw, E

    lr = -dt/nstep
    loop = tqdm(range(nstep)) if show_progress else range(nstep)
    n = 0
    rcond = 1e-8
    index_range = jnp.arange(vstate.n_parameters) if index_range is None else jnp.asarray(index_range)

    data = {"gamma_n": [], "E": [], "E_Var": []} 

    for _ in enumerate(loop):

        a = 0.2 if n <= 0.5 * nstep else 1

        old_pars = vstate.parameters
        k1, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + a*lr*y , old_pars, k1)
        
        k2, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5*(a*lr)*(y1+y2) , old_pars, k1, k2)
        
        data["E"].append(jnp.real(E.Mean).item())
        data["E_Var"].append(jnp.real(E.Variance).item())
        data["gamma_n"].append(n*lr)

        n += 1

        if show_progress:
            loop.set_description(str(E) + " " + str(vstate.sampler_state.acceptance))
    
    json.dump(data, open(log_path, "w"))
    return E


def evolve_blur(
    vstate,
    h0,
    nstep,
    dt,
    index_range,
    blur_q=0.5,
    blur_sample_chunk_size=None,
    blur_conn_chunk_size=None,
    diagonal_mels=True,
    show_progress=True,
    log_path="log_data_L6_blur",
):

    def single_update(vstate):
        samples = vstate.samples
        sample_shape = samples.shape[:-1]
        flat_samples = samples.reshape((-1, samples.shape[-1]))
        variables = {"params": vstate.parameters, **vstate.model_state}
        chunk_size = getattr(vstate, "chunk_size", None)
        key = jax.random.PRNGKey(random.randint(0, np.iinfo(np.int32).max))

        samples_q, importance_weights, E_loc = blurred_sample(
            flat_samples,
            key,
            variables,
            blur_q,
            vstate._apply_fun,
            h0,
            chunk_size,
            sample_chunk_size=blur_sample_chunk_size,
            conn_chunk_size=blur_conn_chunk_size,
            diagonal_mels=diagonal_mels,
        )

        samples_q = samples_q.reshape(samples.shape)
        importance_weights = importance_weights.reshape(sample_shape)
        importance_weights = importance_weights / jnp.mean(importance_weights)
        E_loc = E_loc.reshape(-1)
        E = mpi_statistics(importance_weights.reshape(-1) * E_loc)

        S = QGTJacobian_DefaultConstructor(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples_q,
            pdf=importance_weights / importance_weights.size,
            dense=True,
            mode="complex",
            chunk_size=chunk_size,
        )

        O, Sd = _qgt_dense_components(S, index_range)
        O = O * jnp.sqrt(importance_weights.reshape(-1, 1) / importance_weights.size)
        ΔE_loc = E_loc - E.mean
        OEdata = O.conj() * ΔE_loc[:, None]
        F = nk.stats.sum(OEdata, axis=0)

        ev, V = jnp.linalg.eigh(Sd + rcond * jnp.eye(len(index_range)))
        rho = V.conj().T @ F
        ev_inv = rho / ev
        filter = np.where(np.abs(ev / ev[-1]) > 1e-5, 1e1, np.where(np.abs(ev / ev[-1]) > 1e-8, 0.5, 0.01))
        ev_inv = np.where((np.abs(ev_inv) >= filter), filter * ev_inv / np.abs(ev_inv), ev_inv)

        update = jnp.zeros((vstate.n_parameters), dtype=complex)
        update = update.at[index_range].set(V @ ev_inv)

        y, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
        dw = tree_cast(reassemble(update if jnp.iscomplexobj(y) else update.real), vstate.parameters)

        weights_flat = importance_weights.reshape(-1)
        ess = (jnp.mean(weights_flat) ** 2) / (jnp.mean(weights_flat**2) + jnp.finfo(weights_flat.dtype).eps)

        return dw, E, ess

    lr = -dt / nstep
    loop = tqdm(range(nstep)) if show_progress else range(nstep)
    n = 0
    rcond = 1e-6
    index_range = jnp.arange(vstate.n_parameters) if index_range is None else jnp.asarray(index_range)

    data = {"gamma_n": [], "E": [], "E_Var": [], "ess_blur": []}

    for _ in enumerate(loop):
        old_pars = vstate.parameters
        k1, E, ess = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr * y, old_pars, k1)

        k2, E, ess = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5 * lr * (y1 + y2), old_pars, k1, k2)

        data["E"].append(jnp.real(E.Mean).item())
        data["E_Var"].append(jnp.real(E.Variance).item())
        data["gamma_n"].append(n * lr)
        data["ess_blur"].append(jnp.real(ess).item())

        n += 1

        if show_progress:
            loop.set_description(str(E) + f" ess={float(ess):.3f} " + str(vstate.sampler_state.acceptance))

    json.dump(data, open(log_path, "w"))
    return E
