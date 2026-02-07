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

def main():
    
    args = parse_arguments()
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
    

    ha = cnn.Ruby_Hamiltonian(hi, -1, -1)
    ha += -0.2 * sum([sigmax(hi, i) for i in range(g.N)]) 
    ha += -0.2 * sum([sigmaz(hi, i) for i in range(g.N)])

    rules =  cnn.Gauge_trans(0.5)
    sampler = nk.sampler.MetropolisSampler(hi, rules, sweep_size = 3 * g.N // 4, n_chains=2**8, reset_chains=True, dtype=jnp.int64)
    # eig_val, eig_vec = ha.to_qobj().groundstate(sparse = True)
    # print(eig_val)

    s = hi.random_state(size = 10, key = jax.random.PRNGKey(0))

    Bp = cnn.Ruby_Hamiltonian(hi, 0, 1/(g.N_plaquette)) 

    vstate0 = nk.vqs.MCState(sampler, model = cnn.CNN_symmetric(rotation = True), chunk_size = 2**14, n_samples=2**11, n_discard_per_chain=10)

    # params = pickle.load(open("init_params_L6","rb"))
    # vstate0.parameters = jax.tree_util.tree_map(lambda x: x, params)
    print(vstate0.n_parameters)
    print(vstate0.expect(ha))
    cnn.evolve(vstate0, ha, 300, 5.0, [0, vstate0.n_parameters], op = Bp, show_progress = True)
    params = flax.core.copy(vstate0.parameters)

    # pickle.dump(vstate0.parameters, open("init_params_L6_perturbed", "wb"))
        
if __name__ == "__main__":
    main()