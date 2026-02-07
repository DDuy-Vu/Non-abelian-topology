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


import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some global variables.')
    parser.add_argument('--L', type=int, required=True, help='Value for L')\

    args = parser.parse_args()
    g.L = args.L
    g.update_globals()
    return args

@partial(jax.vmap, in_axes=(0, None, None))
def CZ(s, i1, i2):
    return (1 + s[i1]+s[i2] - s[i1]*s[i2]) / 2

def half_braiding1(s, u0):
    ## CZ gates
    x_positions =g.X_list[u0][0:3]
    plaquette_data = s[:, g.plaquette_list[u0]]
    cz  = CZ(plaquette_data, 1, 0) * CZ(plaquette_data, 3, 2) * CZ(plaquette_data, 3, 0)
    
    return s.at[:, x_positions].set(-s[:, x_positions]), cz

def half_braiding2(s, u0):
    ## CZ gates
    x_positions =g.X_list[u0][3:6]
    plaquette_data = s[:, g.plaquette_list[u0]]
    cz  = CZ(plaquette_data, 5, 4) * CZ(plaquette_data, 5, 2) * CZ(plaquette_data, 5, 0)
    
    return s.at[:, x_positions].set(-s[:, x_positions]), cz

def plaquette_flip(s, u0):
    
    x_positions =g.X_list[u0]
    plaquette_data = s[:, g.plaquette_list[u0]]
    cz  = CZ(plaquette_data, 0, 1) * CZ(plaquette_data, 1, 2) * CZ(plaquette_data, 2, 3)
    cz *= CZ(plaquette_data, 3, 4) * CZ(plaquette_data, 4, 5) * CZ(plaquette_data, 5, 0)
    
    return s.at[:, x_positions].set(-s[:, x_positions]), cz


def main():
    
    args = parse_arguments()
    print(args)
    hi = nk.hilbert.Spin(s=1/2, N=g.N, inverted_ordering = False)

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
    

    ha = cnn.Ruby_Hamiltonian(hi, -1, -1)

    rules =  cnn.Gauge_trans(0.5)
    sampler = nk.sampler.MetropolisSampler(hi, rules, sweep_size = 3 * g.N // 4, n_chains=2**8, reset_chains=True, dtype=jnp.int64)
    vstate0 = nk.vqs.MCState(sampler, model = cnn.CNN_symmetric(mean_field = 0), chunk_size = 2**14, n_samples=2**13, n_discard_per_chain=10)
    params = pickle.load(open("init_params_L6_nopath_perturbed","rb"))
    vstate0.parameters = jax.tree_util.tree_map(lambda x: x, params)
    
    
    print(vstate0.n_parameters)
    print(vstate0.expect(ha))


    ####  Expectation value after brading
    s = vstate0.samples.reshape((-1, g.N))
    log_psi = vstate0.log_value(s)

    s_p, cz1 = half_braiding1(s, 14)
    
    s_p, cz2 = half_braiding1(s_p, 15)
    cz1 *= cz2
    s_p, cz2 = half_braiding2(s_p, 15)
    cz1 *= cz2

    s_p, cz2 = half_braiding2(s_p, 14)
    cz1 *= cz2
    
    data = {"coor":[], "Bp_braid":[], "Bp_original":[]}

    for i in range(g.N_plaquette):
        s_p2, cz3 = plaquette_flip(s_p, i)
        
        s_p2, cz = half_braiding2(s_p2, 14)
        cz3 *= cz

        s_p2, cz = half_braiding2(s_p2, 15)
        cz3 *= cz
        s_p2, cz = half_braiding1(s_p2, 15)
        cz3 *= cz

        s_p2, cz = half_braiding1(s_p2, 14)
        cz3 *= cz

        O_loc = jnp.mean(jnp.exp(vstate0.log_value(s_p2) - log_psi) * cz1 * cz3)
        
        s_p2, cz = plaquette_flip(s, i)
        O_loc_original = jnp.mean(jnp.exp(vstate0.log_value(s_p2) - log_psi) * cz)

        print(g.in2coor(i, g.L), O_loc, O_loc_original)
        data["coor"].append(g.in2coor(i, g.L))
        data["Bp_braid"].append(jnp.real(O_loc).item())
        data["Bp_original"].append(jnp.real(O_loc_original).item())

   
    json.dump(data, open("braiding_data_nopath","w"))
    # pickle.dump(vstate0.parameters, open("init_params_L6", "wb"))
        
if __name__ == "__main__":
    main()