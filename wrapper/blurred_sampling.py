"""Minimal blurred-sampling helper vendored for the wrapper.

This is adapted from ``nqs_blurred_sampling/src/tdvp_utils.py`` so the wrapper
does not import anything outside ``PROJECT_ROOT``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import netket.jax as nkjax


@jax.jit
def ess_from_weights(weights):
    s1_sq = jnp.mean(weights, axis=0) ** 2
    s2 = jnp.mean(weights**2, axis=0)
    return (s1_sq / (s2 + jnp.finfo(weights.dtype).eps)).squeeze()


def _variables(params, model_state):
    if model_state is None:
        return {"params": params}
    return {"params": params, **model_state}


def get_nconn(sample, op):
    connected, mels = op.get_conn_padded(sample)
    offdiag = (mels != 0) & ~jnp.all(connected == sample, axis=-1)
    return jnp.sum(offdiag).astype(mels.real.dtype)


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def blurred_sample_general(x, key, params, model_state, q: float, apply_fn, op, chunk_size):
    """One-step connected-state proposal with importance weights.

    ``x`` is a flat batch of configurations.  The returned weights reweight the
    proposal distribution back to the variational ``|psi|^2`` distribution.
    """

    batch_size = x.shape[0]
    keys = jax.random.split(key, batch_size)

    def get_blurred_sample_and_eloc(items):
        sample, rng = items
        key_stay, key_conn = jax.random.split(rng, 2)
        sample_shape = sample.shape
        sample = sample.reshape(-1)

        connected, mels_orig = op.get_conn_padded(sample)
        n_conn = connected.shape[-2]
        nonzero_mask = (mels_orig != 0) & ~jnp.all(connected == sample, axis=-1)
        probs = nonzero_mask / jnp.sum(nonzero_mask)
        idx = jax.random.choice(key_conn, n_conn, p=probs)
        proposed = connected[idx]

        u1 = jax.random.uniform(key_stay)
        sample_q = jnp.where(u1 > q, sample, proposed)
        connected_q, mels = op.get_conn_padded(sample_q)
        n_conn_all = jax.vmap(partial(get_nconn, op=op))(connected_q)

        variables = _variables(params, model_state)
        logpsi_stay = apply_fn(variables, sample_q)
        logpsi_all = apply_fn(variables, connected_q)

        logp_stay = 2.0 * logpsi_stay.real
        logp_all = 2.0 * logpsi_all.real
        offdiag_mask_q = (mels != 0) & ~jnp.all(connected_q == sample_q, axis=-1)
        logp_all_masked = jnp.where(offdiag_mask_q, logp_all, -jnp.inf)

        log_term_main = jnp.log1p(-q) + logp_stay
        log_term_moves = jsp.special.logsumexp(logp_all_masked, b=q / n_conn_all)
        log_weight_denom = jsp.special.logsumexp(jnp.stack([log_term_main, log_term_moves]))
        weight = jnp.exp(logp_stay - log_weight_denom)

        eloc = jnp.sum(
            mels * jnp.exp(logpsi_all - jnp.expand_dims(logpsi_stay, -1)),
            axis=-1,
        )
        return sample_q.reshape(sample_shape), weight, jnp.atleast_1d(eloc)

    vmapped = jax.vmap(get_blurred_sample_and_eloc, in_axes=0)
    if chunk_size is None:
        return vmapped((x, keys))
    return nkjax.apply_chunked(
        vmapped,
        in_axes=0,
        chunk_size=chunk_size,
        axis_0_is_sharded=False,
    )((x, keys))
