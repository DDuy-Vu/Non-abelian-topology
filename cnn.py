import math
import netket as nk
import numpy as np
import random
import shutil
import functools

from netket.operator.spin import sigmax, sigmaz, sigmay, identity
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
    D = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    D = sparse.coo_matrix(D)
    return  nk.operator.LocalOperator(hi, D, [site1, site2], dtype=float).to_jax_operator()

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
        ha += Jz * sum([sigmaz(hi, u[0]) * sigmaz(hi, u[2]) * sigmaz(hi, u[4]) for u in g.plaquette_list.tolist()])
        ha += Jz * sum([sigmaz(hi, u[1]) * sigmaz(hi, u[3]) * sigmaz(hi, u[5]) for u in g.plaquette_list.tolist()])
    
    if Jx != 0:
        for i in range(g.N_plaquette):
            u = g.plaquette_list[i].tolist()
            s = CZ(hi, u[0], u[1]) * CZ(hi, u[1], u[2]) * CZ(hi, u[2], u[3]) * CZ(hi, u[3], u[4]) * CZ(hi, u[4], u[5]) * CZ(hi, u[5], u[0])

            u = g.X_list[i].tolist()
            s *= sigmax(hi, u[0]) * sigmax(hi, u[1]) * sigmax(hi, u[2]) * sigmax(hi, u[3]) * sigmax(hi, u[4]) * sigmax(hi, u[5])
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
    dtype: type =  complex

    @nn.compact
    def __call__(self,x):
        
        size = x.shape
        W =  self.param('W',nn.initializers.normal(stddev = self.W_scale*0.5/jnp.sqrt(self.kersize**2*(size[2]+self.out_features))),
                (self.kersize**2, size[2], self.out_features,), self.dtype)
        if self.kersize == 2:         
            y = jnp.tensordot(x[:, g.kernel2,:], W, axes=([2,3],[0,1]))
        else:
            y = jnp.tensordot(x[:, g.kernel3,:], W, axes=([2,3],[0,1]))
        
        if self.b_scale > 1e-5:
            b =  self.param('b',nn.initializers.normal(stddev = self.b_scale*0.5/jnp.sqrt(self.kersize**2*(size[2]+self.out_features))),
                    (self.out_features,), self.dtype)        
            y += jnp.tensordot(jnp.ones((size[0],size[1])), b, axes=0)
        
        return y
    
    
class conv2(nn.Module):

    out_features: int
    W_scale: float = 1.0
    b_scale: float = 1.0
    ker_size: int = 2
    dtype: type =  complex

    @nn.compact
    def __call__(self,x):
        
        size = x.shape
        N_kernel = self.ker_size**2

        W =  jnp.pad(self.param('W',nn.initializers.normal(stddev = self.W_scale*0.5/jnp.sqrt(N_kernel * size[-1])), (N_kernel, size[-1], self.out_features), self.dtype), 
                        pad_width = ((0, 1), (0, 0), (0, 0)), constant_values = 0j)
        kernel = jnp.take(W, g.kernel2, axis = 0) if self.ker_size == 2 else jnp.take(W, g.kernel3, axis = 0)
        y = jax.lax.dot_general(x, kernel, (( (1, 2), (0, 2)), ((), ())))

        if self.b_scale > 1e-5:
            b =  self.param('b',nn.initializers.normal(stddev = self.b_scale*0.5/jnp.sqrt(N_kernel* self.out_features)), (self.out_features,), self.dtype)        
            y += b[jnp.newaxis, jnp.newaxis, :]
        
        return y

    
class deep_CNN(nn.Module):
    n_features: int
    @nn.compact
    
    def __call__(self, x):

        ### Convolutional NN
        y1b = conv2(out_features = self.n_features, ker_size = 3)(x)
        y1b = conv2(out_features = self.n_features, ker_size = 3)(activation2(y1b))
        # y1b = conv2(out_features = self.n_features, ker_size = 3)(activation2(y1b))

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
    freeze_mag: bool = False
    freeze_phase: bool = False

    @nn.compact
    
    def __call__(self, x):
        
        ##### Add symmetric copies  #############
        # Swaping sublattice ##################### 
        if self.rotation:
            x = x[:, g.point_group].reshape((-1, g.N))

        size = x.shape

        u =   phase2(x) #x.reshape((x.shape[0], x.shape[1]//3, 3)) # 
        y1 =  deep_CNN(n_features = 6)(u)

        u = jnp.stack((jnp.prod(x[:, g.left_triangles], axis = -1), jnp.prod(x[:, g.right_triangles], axis = -1)), axis = -1)
        y2 =  deep_CNN(n_features = 6)(u)

        out = activation4(y1 + y2)
        out = jnp.sum(out, axis = [1, 2]) / jnp.sqrt(3)
         
        if self.rotation:
            out = out.reshape(-1, len(g.point_group))
            out = jnp.log(jnp.mean(jnp.exp(out), axis = -1))

        return out
    


def evolve(vstate, h0, nstep, dt, index_range, op, show_progress = True): 

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

    data = {"gamma_n": [], "E": [], "Bp": []} 

    for _ in enumerate(loop):

        # if n % 150 == 0:
        #     rcond = rcond / 3
            # print("change random parameters")
            # index_range = np.random.choice(vstate.n_parameters, size = 800, replace = False)

        old_pars = vstate.parameters
        k1, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr*y , old_pars, k1)
        
        k2, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5*lr*(y1+y2) , old_pars, k1, k2)
        
        data["E"].append(jnp.real(E.Mean).item())
        data["Bp"].append(jnp.real(vstate.expect(op).Mean).item())
        data["gamma_n"].append(n*lr)

        n += 1

        if show_progress:
            loop.set_description(str(E) + " " + str(vstate.sampler_state.acceptance))
    
    json.dump(data, open("log_data_L6_perturbed","w"))
    return E


def train_fidelity(vstate, overlap, nstep, dt, show_progress = True): 

    def single_update(vstate):  

        E, F = vstate.expect_and_grad(overlap)
        F, reassemble = convert_tree_to_dense_format(F, "holomorphic")

        S = vstate.quantum_geometric_tensor(nk.optimizer.qgt.QGTJacobianDense(mode = "holomorphic"))
        O = S.O[:,  index_range]
        Sd = mpi.mpi_sum_jax(O.conj().T @ O)[0] + rcond*jnp.eye(len(index_range))

        #### Regularizing based on rotated F / eigs of S
        ev, V = jnp.linalg.eigh(Sd)
        rho = V.conj().T @ F

        #### Regularizing based on rotated F / eigs of S
    
        ev_inv = rho/ev
        filter = np.where(np.abs(ev/ev[-1]) > 1e-5, 1e1, np.where(np.abs(ev/ev[-1]) > 1e-8, 0.5, 0.01 ))
        ev_inv = np.where((np.abs(ev_inv) >= filter), filter * ev_inv/np.abs(ev_inv), ev_inv)
        update = jnp.zeros((vstate.n_parameters), dtype = complex)
        update = update.at[index_range].set(V @ ev_inv)

        dw = tree_cast (reassemble(update), vstate.parameters)

        return dw, E

    lr = -dt/nstep
    loop = tqdm(range(nstep)) if show_progress else range(nstep)
    n = 0
    rcond = 1e-6
    index_range = jnp.arange(vstate.n_parameters)
    


    for _ in enumerate(loop):

        # if n % 150 == 0:
        #     rcond = rcond / 3
            # print("change random parameters")
            # index_range = np.random.choice(vstate.n_parameters, size = 800, replace = False)

        old_pars = vstate.parameters
        k1, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr*y , old_pars, k1)
        
        k2, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5*lr*(y1+y2) , old_pars, k1, k2)

        n += 1

        if show_progress:
            loop.set_description(str(E) + " " + str(vstate.sampler_state.acceptance))

    return E