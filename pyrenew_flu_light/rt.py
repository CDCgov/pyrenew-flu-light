import logging

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as t
from jax.typing import ArrayLike
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.metaclass import (
    DistributionalRV,
    RandomVariable,
    TransformedRandomVariable,
)
from pyrenew.process import SimpleRandomWalkProcess


class CFAEPIM_Rt(RandomVariable):  # numpydoc ignore=GL08
    def __init__(
        self,
        intercept_RW_prior: numpyro.distributions,
        max_rt: float,
        gamma_RW_prior_scale: float,
        week_indices: ArrayLike,
    ):  # numpydoc ignore=GL08
        """
        Initialize the CFAEPIM_Rt class.

        Parameters
        ----------
        intercept_RW_prior : numpyro.distributions.Distribution
            Prior distribution for the random walk intercept.
        max_rt : float
            Maximum value of the reproduction number. Used as
            the scale in the `ScaledLogitTransform()`.
        gamma_RW_prior_scale : float
            Scale parameter for the HalfNormal distribution
            used for random walk standard deviation.
        week_indices : ArrayLike
            Array of week indices used for broadcasting
            the Rt values.
        """
        logging.info("Initializing CFAEPIM_Rt")
        self.intercept_RW_prior = intercept_RW_prior
        self.max_rt = max_rt
        self.gamma_RW_prior_scale = gamma_RW_prior_scale
        self.week_indices = week_indices

    @staticmethod
    def validate(
        intercept_RW_prior: any,
        max_rt: any,
        gamma_RW_prior_scale: any,
        week_indices: any,
    ) -> None:  # numpydoc ignore=GL08
        """
        Validate the parameters of the CFAEPIM_Rt class.

        Raises
        ------
        ValueError
            If any of the parameters are not valid.
        """
        logging.info("Validating CFAEPIM_Rt parameters")
        if not isinstance(intercept_RW_prior, dist.Distribution):
            raise ValueError(
                f"intercept_RW_prior must be a numpyro distribution; was type {type(intercept_RW_prior)}"
            )
        if not isinstance(max_rt, (float, int)) or max_rt <= 0:
            raise ValueError(
                f"max_rt must be a positive number; was type {type(max_rt)}"
            )
        if (
            not isinstance(gamma_RW_prior_scale, (float, int))
            or gamma_RW_prior_scale <= 0
        ):
            raise ValueError(
                f"gamma_RW_prior_scale must be a positive number; was type {type(gamma_RW_prior_scale)}"
            )
        if not isinstance(week_indices, (np.ndarray, jnp.ndarray)):
            raise ValueError(
                f"week_indices must be an array-like structure; was type {type(week_indices)}"
            )

    def sample(self, n_steps: int, **kwargs) -> tuple:  # numpydoc ignore=GL08
        """
        Sample the Rt values using a random walk process
        and broadcast them to daily values.

        Parameters
        ----------
        n_steps : int
            Number of time steps to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls.

        Returns
        -------
        ArrayLike
            An array containing the broadcasted Rt values.
        """
        # sample the standard deviation for the random walk process
        sd_wt = numpyro.sample(
            "Wt_rw_sd", dist.HalfNormal(self.gamma_RW_prior_scale)
        )
        # Rt random walk process
        wt_rv = SimpleRandomWalkProcess(
            name="Wt",
            step_rv=DistributionalRV(
                name="rw_step_rv",
                dist=dist.Normal(0, sd_wt),
                reparam=LocScaleReparam(0),
            ),
            init_rv=DistributionalRV(
                name="init_Wt_rv",
                dist=self.intercept_RW_prior,
            ),
        )
        # transform Rt random walk w/ scaled logit
        transformed_rt_samples = TransformedRandomVariable(
            name="transformed_rt_rw",
            base_rv=wt_rv,
            transforms=t.ScaledLogitTransform(x_max=self.max_rt).inv,
        ).sample(n_steps=n_steps, **kwargs)
        # broadcast the Rt samples to daily values
        broadcasted_rt_samples = transformed_rt_samples[0].value[
            self.week_indices
        ]
        logging.debug(f"Broadcasted Rt samples: {broadcasted_rt_samples}")
        return broadcasted_rt_samples
