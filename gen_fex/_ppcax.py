import collections.abc
from typing import Tuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.numpy.linalg import inv as jinv

import chex
import distrax
from distrax._src.utils import jittable

import numpy as np
from sklearn.base import BaseEstimator

PRNGKey = chex.PRNGKey
Array = Union[chex.Array, chex.ArrayNumpy, jnp.ndarray]
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


class PPCA(jittable.Jittable, BaseEstimator):
    """
    Probabilistic Principal Component Analysis (PPCA).

    This implementation is influenced by the work of @cangermuller: https://github.com/cangermueller/ppca.
    PPCA is a variant of Principal Component Analysis that accounts for uncertainty in the data by using
    a probabilistic model, allowing for latent variable inference and model-based missing data handling.

    Attributes
    ----------
    P : Array
        Input dataset.
    sigma : FloatLike
        Noise variance parameter.
    W : Array
        Principal components matrix.
    mu : Array
        Mean vector of the input data.
    q : IntLike
        Number of latent dimensions.
    prior_sigma : FloatLike
        Prior variance for the Gaussian prior.
    seed : Union[IntLike, PRNGKey]
        PRNG key or integer seed.
    """

    @staticmethod
    def sample_W(seed: Union[IntLike, PRNGKey],
                 sample_shape: Union[IntLike, Sequence[IntLike]]) -> Array:
        """
        Sample a matrix for principal components.

        Parameters
        ----------
        seed : Union[IntLike, PRNGKey]
            PRNG key or integer seed.
        sample_shape : Union[IntLike, Sequence[IntLike]]
            Additional leading dimensions for the sample.

        Returns
        -------
        Array
            A sample with the specified shape.
        """
        rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
        return jax.random.uniform(rng, shape=sample_shape)

    def __init__(self,
                 q: IntLike = 2,
                 prior_sigma: FloatLike = 1.0,
                 seed: Union[IntLike, PRNGKey] = 42):
        """
        Initialize the Probabilistic PCA (PPCA) model.

        Parameters
        ----------
        q : IntLike, optional
            Number of latent dimensions. Default is 2.
        prior_sigma : FloatLike, optional
            Prior variance for the Gaussian prior. Default is 1.0.
        seed : Union[IntLike, PRNGKey], optional
            PRNG key or integer seed. Default is 42.
        """
        self._sigma = prior_sigma
        self.q = q
        self.prior_sigma = prior_sigma
        self.seed = seed

    @property
    def P(self) -> Array:
        return self._P

    @property
    def W(self) -> Array:
        return self._W

    @property
    def sigma(self) -> FloatLike:
        return self._sigma

    @P.setter
    def P(self, value):
        self._P = value

    @W.setter
    def W(self, value):
        self._W = value

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    def data_reshape(self, data) -> Tuple[Array, Tuple[IntLike, IntLike]]:
        """
        Reshape the input data into the required form for fitting.

        Parameters
        ----------
        data : Array-like
            Input data matrix.

        Returns
        -------
        Tuple[Array, Tuple[IntLike, IntLike]]
            Reshaped data matrix and sample shape.
        """
        P = data if isinstance(data, jnp.ndarray) else jnp.asarray(data)
        self.N, self.D = P.shape
        sample_shape = (self.N, self.q)
        return P, sample_shape

    def fit(self, data, use_em=True, max_iter: IntLike=20, verbose: IntLike=0):
        """
        Fit the PPCA model to the input data.

        Parameters
        ----------
        data : Array-like
            Input data matrix.
        use_em : bool, optional
            Whether to use the Expectation-Maximization (EM) algorithm. Default is True.
        max_iter : IntLike, optional
            Maximum number of iterations. Default is 20.
        verbose : IntLike, optional
            Verbosity level. Default is 0.

        Returns
        -------
        float or None
            Negative log-likelihood if using EM, None if using Maximum Likelihood (ML) estimation.
        """
        self.P, sample_shape = self.data_reshape(data)

        if use_em:
            self.W: Array = self.sample_W(self.seed, sample_shape)
            n_iter = jnp.arange(max_iter).reshape(1, -1)
            ell = jax.vmap(self._fit_em, in_axes=(0, None))(n_iter, verbose)
            return ell
        else:
            self.__fit_ml()

    def transform(self,
                  P: Optional[Array] = None) -> Array:
        """
        Transform the input data into the latent space.

        Parameters
        ----------
        P : Array, optional
            Input data matrix. If not provided, the model's training data will be used.

        Returns
        -------
        Array
            Transformed data in the latent space.

        Notes
        -----
        This method transforms the input data into the latent space using the current parameters
        of the model. It calculates the latent variables based on the formula:

            z = inv((W.T @ W) + sigma * I_q) @ W.T @ (P - mu) @ (P - mu).T

        Where:
        - z is the matrix of latent variables.
        - W is the matrix of principal components.
        - sigma^2 is the noise variance parameter.
        - I_q is the q-dimensional identity matrix.
        - P is the input data matrix.
        - mu is the mean of the input data.
        """
        if P is None:
            P = self.P
        else:
            P, _ = self.data_reshape(P)

        try:
            M_inv = jinv(self.W.T @ self.W + self.sigma * np.eye(self.q))  # pylint-disable
        except (FloatingPointError, np.linalg.LinAlgError) as e:
            raise FloatingPointError(f"Matrix Inversion failed (singularity error): {e}")

        pmu = (P - self.mu)
        z = M_inv @ self.W.T @ pmu

        return z

    def fit_transform(self,
                      *args,
                      **kwargs
    ) -> Union[Tuple[Union[Array, FloatLike], Array], Array]:
        """
        Fit the model to the data and simultaneously transform the data.

        Returns
        -------
        Union[Tuple[Union[Array, FloatLike], Array], Array]
            Negative log-likelihood if using EM, None if using ML estimation, and the transformed data.
        """
        ell = self.fit(*args, **kwargs)

        if isinstance(ell, Array):
            return ell, self.transform()
        else:
            return self.transform()

    def inverse_transform(self,
                          z: Optional[Array] = None,
                          is_add_noise: bool = False) -> Array:
        """
        Transform the latent data back to the reconstructed data in the original space.

        Parameters
        ----------
        z : Array, optional
            Latent data matrix. If not provided, it will be inferred using the transform method.
        is_add_noise : bool, optional
            Whether to add noise to the reconstructed data. Default is False.

        Returns
        -------
        Array
            Reconstructed data in the original space.
        """
        if z is None:
            z = self.transform()

        recon_data = self.W @ z + self.mu

        if is_add_noise:
            n, d = recon_data.shape[0], recon_data.shape[1]
            d_iter = jnp.arange(d)

            k = jax.random.PRNGKey(self.seed)
            keys = jax.random.split(k, d)

            dist = distrax.Normal(loc=0., scale=self.sigma)
            noises = jnp.stack([dist.sample(seed=keys[i], sample_shape=n)
                                for i in range(len(keys))]).T

            def add_noise(state, i):
                recon_data, noises = state
                recon_data = recon_data.at[:, i].add(noises[:, i])
                _ = 0.
                return (noises, recon_data), _

            (_, recon_data), _ = jax.lax.scan(add_noise, (recon_data, noises), d_iter)
        return recon_data

    def _ell(self,
             W: Array,
             mu: Array,
             sigma: FloatLike,
             lg_sigma: Array,
             ell_norm=True
             ) -> Array:
        """
        Calculate the negative log-likelihood of the PPCA model.

        Parameters
        ----------
        W : Array
            Principal components matrix.
        mu : Array
            Mean vector of the data.
        sigma : FloatLike
            Noise variance.
        lg_sigma : Array
            Log of the noise variance.
        ell_norm : bool, optional
            Whether to normalize the log-likelihood. Default is True.

        Returns
        -------
        float
            Negative log-likelihood.
        """
        # E-step
        M_inv = jinv(W.T @ W + sigma * np.eye(W.shape[1]))
        M_inv_WT = M_inv @ W.T
        ell = jnp.float32(0.)

        def e_step(state, i):
            P, mu, W, sigma, ell = state
            p = P[:, i][:, None]
            pmu = p - mu

            phi = M_inv_WT @ pmu
            Phi = sigma * M_inv + phi @ phi.T
            ell += 0.5 * jnp.trace(Phi)
            ell += jnp.where(sigma > 1e-5,
                        1/(2 * sigma) * jnp.float32((pmu.T @ pmu).reshape(-1,)[0]), 0.0)
            ell -= jnp.where(sigma > 1e-5,
                        1/sigma * jnp.float32((phi.T @ W.T @ pmu).reshape(-1,)[0]), 0.0)

            ell += jnp.where(sigma > 1e-5,
                                1/(2 * sigma) * jnp.trace(W.T @ W @ Phi), 0.0)

            return (P, mu, W, sigma, ell), ell

        n_iter = jnp.arange(self.D)
        (*params, ell), _ = jax.lax.scan(e_step, (self.P, mu, W, sigma, ell), n_iter)

        ell += jax.lax.cond(sigma > 1e-5,
                            lambda x: 0.5 * self.D * self.N * x,
                            lambda x: 0.,
                            lg_sigma)
        ell *= -1.0
        ell /= jax.lax.cond(ell_norm,
                            lambda: jnp.float32(self.D),
                            lambda: 1.)
        return ell

    def _fit_em(self, max_iter: Array, verbose: IntLike):
        """
        Fit the model using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        max_iter : Array
            Maximum number of iterations.
        verbose : IntLike
            Verbosity level.

        Returns
        -------
        Array
            Negative log-likelihood at each iteration.

        Notes
        -----
        This method fits the model to the input data using the EM algorithm.
        It iteratively updates the parameters of the model to maximize the likelihood.
        The EM algorithm involves alternating between:

        - E-step: Estimating the latent variables using the current model parameters.
        - M-step: Updating the model parameters based on the estimated latent variables.
        """
        def m_step(state, i):
            (W, sigma, S, mu, I_q, N, verb) = state

            # M-step
            inv_M = jinv(W.T @ W + sigma * I_q)  # q x q
            xi = jinv(sigma * I_q + inv_M @ W.T @ S @ W)  # q x q

            W = S @ W @ xi  # N x q
            sigma = jnp.float32(1/N * jnp.trace(S - S @ W @ inv_M @ W.T))
            lg_sigma = jnp.log(sigma)

            ell = self._ell(W=W, mu=mu, sigma=sigma, lg_sigma=lg_sigma)
            jax.lax.cond(verb == 1,
                         lambda: jax.debug.print("Iter: {}, Updated ell: {}", i+1, ell),
                         lambda: None)

            return (W, sigma, S, mu, I_q, N, verb), ell

        I_q = jnp.eye(self.q)
        self.mu = jnp.mean(self.P, axis=1)[:, np.newaxis]  # N x 1
        p_cent = self.P - self.mu

        # Sample Covariance Matrix
        if self.N < self.D:
            S = self.N**-1 * p_cent @ p_cent.T  # N x N
        else:
            S = self.D**-1 * p_cent @ p_cent.T  # N x N

        (self.W, self.sigma,
         *params), ell = jax.lax.scan(m_step, (self.W, self.sigma, S,
                                               self.mu, I_q, self.N, verbose),
                                      max_iter)
        return ell

    def __fit_ml(self):
        r"""
        Fit the model using Maximum Likelihood (ML) estimation.
        """
        self.mu = np.mean(self.P, axis=1)[:, np.newaxis]  # N x 1
        u, s, v = np.linalg.svd(self.P - self.mu)

        if self.q > len(s):
            ss = np.zeros(self.q)
            ss[:len(s)] = s
        else:
            ss = s[:self.q]

        ss = np.sqrt(np.maximum(0, ss**2 - self.sigma))
        self.W = u[:, :self.q].dot(np.diag(ss))

        if self.q < self.D:
            self.sigma = 1.0 / (self.D - self.q) * np.sum(s[self.q:]**2)
        else:
            self.sigma = 0.0

    def sample(self, 
               n_samples: int = 1, 
               seed: Optional[Union[IntLike, PRNGKey]] = None, 
               add_noise: bool = True) -> Array:
        """
        Generate synthetic samples from the fitted PPCA model.

        The method leverages the generative process defined by the PPCA model parameters.
        Latent variables are sampled from a standard normal distribution and transformed
        into the observed space using the learned loading matrix and mean. Optionally,
        Gaussian noise is added to reflect the model's estimated variance.

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic samples to generate. Default is 1.
        seed : Union[IntLike, PRNGKey], optional
            PRNG key or integer seed for reproducibility. If None, uses the model's seed.
        add_noise : bool, optional
            Whether to include observation noise in the generated samples. Default is True.

        Returns
        -------
        Array
            Generated data matrix of shape (n_samples, D), where D is the number of features.

        Notes
        -----
        The generative process follows:
            1. Sample z ~ N(0, I_q) where q is the latent dimension.
            2. Compute x_mean = W @ z.T + mu (project to observed space).
            3. Add noise: x = x_mean + ε, where ε ~ N(0, sigma^2 I) if add_noise=True.
        """
        # Handle seed/PRNGKey
        if seed is None:
            seed = self.seed
        rng = jax.random.PRNGKey(seed) if isinstance(seed, (int, np.integer)) else seed
        rng_z, rng_eps = jax.random.split(rng)

        # Sample latent variables z ~ N(0, I_q)
        z = jax.random.normal(rng_z, shape=(n_samples, self.q))

        # Project to observed space: x_mean = W @ z.T + mu
        x_mean = jnp.dot(z, self.W.T) + self.mu.T  # Shape (n_samples, D)

        # Add observation noise if specified
        if add_noise:
            noise = jax.random.normal(rng_eps, x_mean.shape) * self.sigma
            samples = x_mean + noise
        else:
            samples = x_mean

        return samples