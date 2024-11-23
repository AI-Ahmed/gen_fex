import collections.abc
from functools import partial
from typing import Tuple, Optional, Sequence, Union

import jax
from jax import jit
import jax.numpy as jnp
from jax.numpy.linalg import inv as jinv

import chex
import distrax

import numpy as np
from ._ppcax import PPCA

PRNGKey = chex.PRNGKey
Array = Union[chex.Array, chex.ArrayNumpy]
IntLike = Union[int, np.int16, np.int32, np.int64]
FloatLike = Union[float, np.float16, np.float32, np.float64]


def convert_seed_and_sample_shape(
    seed: Union[IntLike, PRNGKey],
    sample_shape: Union[IntLike, Sequence[IntLike]]
) -> Tuple[PRNGKey, Tuple[int, ...]]:
  """
  Shared functionality to ensure that seeds and shapes are the right type.
  Ref: https://github.com/google-deepmind/distrax/blob/ee17707c419766252386da3337f24751a6d12905/distrax/_src/distributions/distribution.py#L312
  """

  if not isinstance(sample_shape, collections.abc.Sequence):
    sample_shape = (sample_shape,)
  sample_shape = tuple(map(int, sample_shape))

  if isinstance(seed, (int, np.signedinteger)):
    rng = jax.random.PRNGKey(seed)
  else:  # key is of type PRNGKey
    rng = seed

  return rng, sample_shape  # type: ignore[bad-return-type]


class PKPCA(PPCA):
    """Probabilistic KPCA based on Wishart Process"""
    def kernel_function(self, P: Array, kernel: str = 'rbf', gamma: Optional[float] = None) -> Array:
        """
        Kernel function for KPCA. Supports RBF and linear kernels.

        Parameters
        ----------
        P : Array
            Input data matrix.
        kernel : str, optional
            Type of kernel function. Default is 'rbf'.
        gamma : float, optional
            Parameter for the RBF kernel. Default is 1.0.

        Returns
        -------
        Array
            Kernel matrix.
        """
        if gamma is None:
            gamma = 1. / P.shape[0]

        if kernel == 'linear':
            return P @ P.T
        elif kernel == 'rbf':
            dist = jnp.sum((P[:, np.newaxis] - P[np.newaxis, :]) ** 2, axis=-1)
            return jnp.exp(-gamma * dist)
        else:
            raise ValueError("Unsupported kernel function.")

    @partial(jit, static_argnums=(0, 2))
    def sample_wishart_covariance(self, seed: PRNGKey,
                                  df: IntLike, K_fn: Array) -> Array:
        """
        Sample a covariance matrix from a Wishart-like distribution using distrax,
        and apply centering with matrix H if required.

        Parameters
        ----------
        seed : PRNGKey
            JAX random key.
        df : IntLike
            Degrees of freedom for the Wishart distribution.
        K_fn : Array
            Kernal matrix function for the Wishart distribution.

        Returns
        -------
        Array
            Sampled and centered covariance matrix.
        """
        n, d = K_fn.shape
        _, subkey = jax.random.split(seed)

        samples = jax.random.multivariate_normal(subkey, jnp.zeros(n), jnp.cov(K_fn), (df,))

        # Construct the kernel matrix
        K = samples.T @ samples

        H = (jnp.eye(n) - jnp.ones((n, n)) / n) / n ** 0.5 # Centering matrix H
        HKH = H.T @ K @ H

        # print(k_fn.shape, samples.shape, K.shape, H.shape, HKH.shape)
        return HKH

    def _fit_em(self, max_iter: Array, verbose: IntLike):
        def m_step(state, i):
            (W, sigma, HKH, mu, I_q, N, verb) = state

            inv_M = jinv(W.T @ W + sigma * I_q)  # q x q
            xi = jinv(N * sigma * I_q + inv_M @ W.T @ HKH @ W)  # q x q

            W = HKH @ W @ xi  # N x q
            sigma = jnp.float32(1/(N ** 2) * jnp.trace(HKH - HKH @ W @ inv_M @ W.T))
            lg_sigma = jnp.log(sigma)

            ell = self._ell(W=W, mu=mu, sigma=sigma, lg_sigma=lg_sigma)
            jax.lax.cond(verb == 1,
                        lambda: jax.debug.print("Iter: {}, Updated ell: {}, sigma: {}", i+1, ell, sigma),
                        lambda: None)

            return (W, sigma, HKH, mu, I_q, N, verb), ell


        I_q = jnp.eye(self.q)
        self.mu = jnp.mean(self.P, axis=1)[:, np.newaxis]  # N x 1
        # p_cent = self.P - self.mu

        # Sample covariance matrix using Wishart process
        rng, _ = convert_seed_and_sample_shape(self.seed, (1, 1))
        rng, _ = jax.random.split(rng)

        K_fn = self.kernel_function(self.P, kernel='rbf', gamma=1 / (2 * jnp.float32(jnp.abs(self.sigma))))  # Kernelized scale matrix
        HKH = self.sample_wishart_covariance(rng, df=self.q, K_fn=K_fn) # N x N

        (self.W, self.sigma,
        *params), ell = jax.lax.scan(m_step, (self.W, self.sigma, HKH,
                                              self.mu, I_q, self.N, verbose),
                                     max_iter)
        return ell
