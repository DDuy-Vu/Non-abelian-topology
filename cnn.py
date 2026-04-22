import math
import netket as nk
import numpy as np
import random
import shutil
import functools

from netket.operator.spin import sigmax, sigmaz, sigmay, identity
import jax
import jax.numpy as jnp
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


def Ruby_Hamiltonian(hi, Jz, Jx):

    ha = 0 * identity(hi)

    if Jz != 0:
        ha += Jz * sum([(sigmaz(hi, u[0]) @ sigmaz(hi, u[2]) @ sigmaz(hi, u[4])) for u in g.plaquette_list.tolist()])
        ha += Jz * sum([(sigmaz(hi, u[1]) @ sigmaz(hi, u[3]) @ sigmaz(hi, u[5])) for u in g.plaquette_list.tolist()])
    
    if Jx != 0:
        local_matrix = _ruby_jx_local_matrix()
        for i in range(g.N_plaquette):
            support = np.concatenate((np.array(g.plaquette_list[i]), np.array(g.X_list[i]))).tolist()
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


class CNN_symmetric(nn.Module):

    rotation: bool = True
    use_small_point_group: bool = False
    use_color_labels: bool = False
    freeze_mag: bool = False
    freeze_phase: bool = False

    @nn.compact
    
    def __call__(self, x):
        
        ##### Add symmetric copies  #############
        # Swaping sublattice ##################### 
        rotation_group = g.small_point_group if self.use_small_point_group else g.point_group
        if self.rotation:
            x = x[:, rotation_group].reshape((-1, g.N))

        size = x.shape

        u =   phase2(x) #x.reshape((x.shape[0], x.shape[1]//3, 3)) # 
        y1 =  deep_CNN(n_features = 8, use_color_labels = self.use_color_labels)(u)

        u = jnp.stack((jnp.prod(x[:, g.left_triangles], axis = -1), jnp.prod(x[:, g.right_triangles], axis = -1)), axis = -1)
        y2 =  deep_CNN(n_features = 8, use_color_labels = self.use_color_labels)(u)

        out = activation4(y1 + y2)
        out = jnp.sum(out, axis = [1, 2]) / jnp.sqrt(3)
         
        if self.rotation:
            out = out.reshape(-1, rotation_group.shape[0])
            out = jnp.log(jnp.mean(jnp.exp(out), axis = -1))

        return out
    


def evolve(vstate, h0, nstep, dt, index_range, show_progress = True, log_path = "log_data_L6_perturbed"): 

    def single_update(vstate):  

        E_loc = vstate.local_estimators(h0).reshape(-1)
        E = mpi_statistics(E_loc)
        ΔE_loc = (E_loc - E.mean)

        init_indices = np.random.choice(jnp.arange(vstate.n_samples)[jnp.abs(ΔE_loc) <= max(jnp.sqrt(E.Variance), 1.0)], vstate.sampler.n_chains, replace = False)
        vstate.sampler_state = vstate.sampler_state.replace(σ = vstate.samples.reshape(-1, g.N)[init_indices, :])

        S = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(mode = "complex"))
        O = S.O[:, :, index_range] / (g.N)
        O = O[:, 0, :] + 1j * O[:, 1, :]
        Sd = mpi.mpi_sum_jax(O.conj().T @ O)[0] + rcond*jnp.eye(len(index_range))
        
        OEdata = O.conj()/np.sqrt(vstate.n_samples) * ΔE_loc[:, None] / (g.N)
        F = nk.stats.sum(OEdata, axis=0)

        #### Regularizing based on rotated F / eigs of S
        ev, V = jnp.linalg.eigh(Sd)
        rho = V.conj().T @ F

        #### Regularizing based on rotated F / eigs of S
    
        ev_inv = rho/ev
        filter = np.where(np.abs(ev/ev[-1]) > 1e-5, 1e1, np.where(np.abs(ev/ev[-1]) > 1e-8, 0.5, 0.01 ))
        ev_inv = np.where((np.abs(ev_inv) >= filter), filter * ev_inv/np.abs(ev_inv), ev_inv)
        update = jnp.zeros((vstate.n_parameters), dtype = complex)
        update = update.at[index_range].set(V @ ev_inv)

        y, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
        dw = tree_cast (reassemble(update), vstate.parameters)

        return dw, E

    lr = -dt/nstep
    loop = tqdm(range(nstep)) if show_progress else range(nstep)
    n = 0
    rcond = 1e-6
    index_range = jnp.arange(vstate.n_parameters)

    data = {"gamma_n": [], "E": [], "E_Var": []} 

    for _ in enumerate(loop):

        old_pars = vstate.parameters
        k1, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr*y , old_pars, k1)
        
        k2, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5*lr*(y1+y2) , old_pars, k1, k2)
        
        data["E"].append(jnp.real(E.Mean).item())
        data["E_Var"].append(jnp.real(E.Variance).item())
        data["gamma_n"].append(n*lr)

        n += 1

        if show_progress:
            loop.set_description(str(E) + " " + str(vstate.sampler_state.acceptance))
    
    json.dump(data, open(log_path, "w"))
    return E
