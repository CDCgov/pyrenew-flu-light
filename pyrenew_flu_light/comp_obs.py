"""
The observation process component in pyrenew-flu-light.
"""

import logging

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pyrenew.transformation as t
from jax.typing import ArrayLike
from pyrenew.metaclass import DistributionalRV, RandomVariable
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.regression import GLMPrediction


class CFAEPIM_Observation(RandomVariable):
    """
    Class representing the observation process
    in the CFAEPIM model. This class handles the generation
    of the alpha (instantaneous ascertaintment rate) process
    and the negative binomial observation process for
    modeling hospitalizations from latent infections.

    Parameters
    ----------
    predictors : ArrayLike
        Array of predictor (covariates) values for the alpha process.
    alpha_prior_dist : numpyro.distributions
        Prior distribution for the intercept in the alpha process.
    coefficient_priors : numpyro.distributions
        Prior distributions for the coefficients in the alpha process.
    nb_concentration_prior : numpyro.distributions
        Prior distribution for the concentration parameter of
        the negative binomial distribution.
    """

    def __init__(
        self,
        predictors,
        alpha_prior_dist,
        coefficient_priors,
        nb_concentration_prior,
    ):  # numpydoc ignore=GL08
        logging.info("Initializing CFAEPIM_Observation")

        CFAEPIM_Observation.validate(
            predictors,
            alpha_prior_dist,
            coefficient_priors,
            nb_concentration_prior,
        )

        self.predictors = predictors
        self.alpha_prior_dist = alpha_prior_dist
        self.coefficient_priors = coefficient_priors
        self.nb_concentration_prior = nb_concentration_prior

        self._init_alpha_t()
        self._init_negative_binomial()

    def _init_alpha_t(self):
        """
        Initialize the alpha process using a generalized
        linear model (GLM) (transformed linear predictor).
        The transform is set to the inverse of the sigmoid
        transformation.
        """
        logging.info("Initializing alpha process")
        self.alpha_process = GLMPrediction(
            name="alpha_t",
            fixed_predictor_values=self.predictors,
            intercept_prior=self.alpha_prior_dist,
            coefficient_priors=self.coefficient_priors,
            transform=t.SigmoidTransform().inv,
        )

    def _init_negative_binomial(self):
        """
        Sets up the negative binomial
        distribution for modeling hospitalizations
        with a prior on the concentration parameter.
        """
        logging.info("Initializing negative binomial process")
        self.nb_observation = NegativeBinomialObservation(
            name="negbinom_rv",
            concentration_rv=DistributionalRV(
                name="nb_concentration",
                dist=self.nb_concentration_prior,
            ),
        )

    @staticmethod
    def validate(
        predictors: any,
        alpha_prior_dist: any,
        coefficient_priors: any,
        nb_concentration_prior: any,
    ) -> None:
        """
        Validate the parameters of the CFAEPIM observation process. Checks that
        the predictors, alpha prior distribution, coefficient priors, and negative
        binomial concentration prior are correctly specified. If any parameter
        is invalid, an appropriate error is raised.
        """
        logging.info("Validating CFAEPIM_Observation parameters")
        if not isinstance(predictors, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"Predictors must be an array-like structure; was type {type(predictors)}"
            )
        if not isinstance(alpha_prior_dist, dist.Distribution):
            raise TypeError(
                f"alpha_prior_dist must be a numpyro distribution; was type {type(alpha_prior_dist)}"
            )
        if not isinstance(coefficient_priors, dist.Distribution):
            raise TypeError(
                f"coefficient_priors must be a numpyro distribution; was type {type(coefficient_priors)}"
            )
        if not isinstance(nb_concentration_prior, dist.Distribution):
            raise TypeError(
                f"nb_concentration_prior must be a numpyro distribution; was type {type(nb_concentration_prior)}"
            )

    def sample(
        self,
        infections: ArrayLike,
        inf_to_hosp_dist: ArrayLike,
        **kwargs,
    ) -> tuple:
        """
        Sample from the observation process. Generates samples
        from the alpha process and calculates the expected number
        of hospitalizations by convolving the infections with
        the infection-to-hospitalization (delay distribution)
        distribution. It then samples from the negative binomial
        distribution to model the observed
        hospitalizations.

        Parameters
        ----------
        infections : ArrayLike
            Array of infection counts over time.
        inf_to_hosp_dist : ArrayLike
            Array representing the distribution of times
            from infection to hospitalization.
        **kwargs : dict, optional
            Additional keyword arguments passed through
            to internal sample calls, should there be any.

        Returns
        -------
        tuple
            A tuple containing the sampled instantaneous
            ascertainment values and the expected
            hospitalizations.
        """
        alpha_samples = self.alpha_process.sample()["prediction"]
        alpha_samples = alpha_samples[: infections.shape[0]]
        expected_hosp = (
            alpha_samples
            * jnp.convolve(infections, inf_to_hosp_dist, mode="full")[
                : infections.shape[0]
            ]
        )
        logging.debug(f"Alpha samples: {alpha_samples}")
        logging.debug(f"Expected hospitalizations: {expected_hosp}")
        return alpha_samples, expected_hosp
