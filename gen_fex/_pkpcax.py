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

        k_fn = K_fn + 1e-6 * jnp.eye(n) # Add small noise to ensure numerical stability (in case of near-singularity)
        L_tri = jnp.linalg.cholesky(k_fn)

        # Use Sigma for multivariate normal sampling
        dist = distrax.MultivariateNormalTri(loc=jnp.zeros(n), scale_tri=jnp.cov(L_tri))
        samples = dist.sample(seed=subkey, sample_shape=(df,))

        # Construct the sampled kernel matrix
        K = samples.T @ samples

        H = (jnp.eye(n) - jnp.ones((n, n)) / n) / n ** 0.5 # Centering matrix H
        HKH = H.T @ K @ H

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

    def sample(self, 
               n_samples: int = 1, 
               seed: Optional[Union[IntLike, PRNGKey]] = None, 
               add_noise: bool = True) -> Array:
        """
        Generate synthetic samples using the PKPCA model with Wishart-derived covariance.

        Samples are drawn from a multivariate normal distribution parameterized by:
        - Covariance matrix sampled from Wishart process using the RBF kernel
        - Learned mean vector from training data
        - Optional observation noise scaled by model's sigma parameter

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic samples to generate. Default is 1.
        seed : Union[IntLike, PRNGKey], optional
            PRNG key or integer seed. Uses model seed if None.
        add_noise : bool, optional
            Whether to add observation noise. Default is True.

        Returns
        -------
        Array
            Generated samples of shape (n_samples, D)

        Notes
        -----
        1. Uses kernel matrix from training data to maintain temporal structure
        2. Wishart sampling captures covariance uncertainty for risk-aware generation
        3. Noise variance matches model's heteroskedastic uncertainty estimate
        """
        # Seed handling
        if seed is None:
            seed = self.seed
        rng = jax.random.PRNGKey(seed) if isinstance(seed, (int, np.integer)) else seed
        rng_wishart, rng_noise = jax.random.split(rng)

        # 1. Compute RBF kernel matrix from training data
        K = self.kernel_function(self.P, kernel='rbf', 
                                gamma=1/(2*jnp.abs(self.sigma)))

        # 2. Sample covariance from Wishart distribution
        HKH = self.sample_wishart_covariance(rng_wishart, self.q, K)

        # 3. Sample from multivariate normal with Wishart covariance
        samples = jax.random.multivariate_normal(
            rng_noise,
            mean=jnp.squeeze(self.mu),  # Match covariance dimensions
            cov=HKH + 1e-6*jnp.eye(HKH.shape[0]),  # Ensure PD
            shape=(n_samples,)
        )

        # 4. Add heteroskedastic noise if enabled
        if add_noise:
            noise_scale = self.sigma * jnp.ones(samples.shape)
            samples += jax.random.normal(rng_noise, samples.shape) * noise_scale

        return samples