"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class BayesianLinearRegressionParameters(NamedTuple):
    a: float
    b: float
    mu: jnp.ndarray
    cov: jnp.ndarray


class BayesianLinearRegression:
    """
    Bayesian linear regression with a posterior parametrised as
    pi(beta, sigma^2) = Normal(beta|sigma^2) * InvGamma(sigma^2).
    """
    def __init__(self, feature_dim: int, a0: float, b0: float, lambda_prior: float, intercept: bool):
        self.feature_dim = feature_dim
        self.a0 = a0
        self.b0 = b0
        self.lambda_prior = lambda_prior
        self.intercept = intercept

    def init(self, rng, x):
        # Setup inverse gamma parameters
        a = self.a0
        b = self.b0

        # Setup Gaussian parameters
        mu = jnp.zeros((self.feature_dim + self.intercept, ))
        cov = (1.0 / self.lambda_prior) * jnp.eye(self.feature_dim + self.intercept)

        return BayesianLinearRegressionParameters(a=a, b=b, mu=mu, cov=cov)

    def apply(self, params, rng, x):
        """
        Sample a single set of parameters from the posterior and return prediction.
        """
        a, b, mu, cov = params.a, params.b, params.mu, params.cov
        rng_gamma, rng_gaussian = jax.random.split(rng, 2)

        # Sample sigma^2 and beta from posterior
        sigma2_s = b / jax.random.gamma(rng_gamma, a)
        beta_s = jax.random.multivariate_normal(rng_gaussian, mu, sigma2_s * cov)

        # NOTE: This could fail if covariance is not positive definite
        # assert all(jnp.linalg.eigvalsh(self.cov))
        # pos_definite = jnp.all(jnp.linalg.eigvalsh(self.cov))

        # Sample from unit Gaussian as backup
        # beta_s_fallback = jax.random.multivariate_normal(
        #     rng_gaussian, jnp.zeros((self.feature_dim + self.intercept)), jnp.eye(self.feature_dim + self.intercept)
        # )
        # beta_s = pos_definite * beta_s + (1.0 - pos_definite) * beta_s_fallback

        if self.intercept:
            return jnp.dot(beta_s[:-1], x.T) + beta_s[-1]
        else:
            return jnp.dot(beta_s, x.T)

    def fit(self, params, x, y):
        """
        Fit params to data x,y using non-sequential update (i.e. everything is recomputed)
        """
        mask = jnp.all(x != 0, axis=1)
        num_data = jnp.sum(mask)  # Instead of x.shape[0] to for allow masking

        if self.intercept:
            x = jnp.column_stack((x, mask))

        precision = jnp.dot(x.T, x) + self.lambda_prior * jnp.eye(self.feature_dim + self.intercept)
        cov = jnp.linalg.inv(precision)
        mu = jnp.dot(cov, jnp.dot(x.T, y))

        # Inverse Gamma posterior update
        a = self.a0 + num_data / 2.0
        b = self.b0 + 0.5 * (jnp.dot(y.T, y) - jnp.dot(mu.T, jnp.dot(precision, mu)))

        return BayesianLinearRegressionParameters(a=a, b=b, mu=mu, cov=cov)


class RidgeRegressionParams(NamedTuple):
    weight: jnp.ndarray


class RidgeRegression:
    def __init__(self, feature_dim: int, l2_reg: float, intercept: bool) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.l2_reg = l2_reg
        self.intercept = intercept

    def __call__(self, x):
        self.apply(x)

    def init(self, rng, x):
        weight = jnp.zeros(shape=(self.feature_dim + self.intercept, ), dtype=x.dtype)
        return RidgeRegressionParams(weight=weight)

    def apply(self, params, x):
        weight = params.weight

        if self.intercept:
            return jnp.dot(weight[:-1], x.T) + weight[-1]
        else:
            return jnp.dot(weight, x.T)

    def fit(self, params, x, y):
        """
        Minimize ridge loss 0.5 * (jnp.mean((jnp.dot(x, w) - y) ** 2) + l2_reg * jnp.sum(w ** 2))
        using the conjugate gradient method.
        """
        if self.intercept:
            mask = jnp.all(x != 0, axis=1)
            x = jnp.column_stack((x, mask))

        def matvec(u):
            return jnp.dot(x.T, jnp.dot(x, u)) + self.l2_reg * u

        weight = jax.scipy.sparse.linalg.cg(
            matvec,
            jnp.dot(x.T, y),
            x0=params.weight,
            maxiter=10
        )[0]

        return RidgeRegressionParams(weight=weight)

    def score(self, params, x, y):
        """
        Compute the R2 coefficient of determination.
        """
        y_pred = self.apply(params, x)
        return 1.0 - (jnp.sum((y - y_pred) ** 2) / jnp.sum((y - jnp.mean(y)) ** 2))
