"""
The infections process component in pyrenew-flu-light.
"""

import logging

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from pyrenew.latent import logistic_susceptibility_adjustment
from pyrenew.metaclass import RandomVariable


class CFAEPIM_Infections(RandomVariable):
    """
    Class representing the infection process in
    the CFAEPIM model. This class handles the sampling of
    infection counts over time, considering the
    reproduction number, generation interval, and population size,
    while accounting for susceptibility depletion.

    Parameters
    ----------
    I0 : ArrayLike
        Initial infection counts.
    susceptibility_prior : numpyro.distributions
        Prior distribution for the susceptibility proportion
        (S_{v-1} / P).
    """

    def __init__(
        self,
        I0: ArrayLike,
        susceptibility_prior: numpyro.distributions,
    ):
        logging.info("Initializing CFAEPIM_Infections")
        self.I0 = I0
        self.susceptibility_prior = susceptibility_prior

    @staticmethod
    def validate(I0: any, susceptibility_prior: any) -> None:
        """
        Validate the parameters of the
        infection process. Checks that the initial infections
        (I0) and susceptibility_prior are
        correctly specified. If any parameter is invalid,
        an appropriate error is raised.

        Raises
        ------
        TypeError
            If I0 is not array-like or
            susceptibility_prior is not
            a numpyro distribution.
        """
        logging.info("Validating CFAEPIM_Infections parameters")
        if not isinstance(I0, (np.ndarray, jnp.ndarray)):
            raise TypeError(
                f"Initial infections (I0) must be an array-like structure; was type {type(I0)}"
            )
        if not isinstance(susceptibility_prior, dist.Distribution):
            raise TypeError(
                f"susceptibility_prior must be a numpyro distribution; was type {type(susceptibility_prior)}"
            )

    def sample(
        self, Rt: ArrayLike, gen_int: ArrayLike, P: float, **kwargs
    ) -> tuple:
        """
        Given an array of reproduction numbers,
        a generation interval, and the size of a
        jurisdiction's population,
        calculate infections under the scheme
        of susceptible depletion.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction numbers over time; this is an array of
            Rt values for each time step.
        gen_int : ArrayLike
            Generation interval probability mass function. This is
            an array of probabilities representing the
            distribution of times between successive infections
            in a chain of transmission.
        P : float
            Population size. This is the total population
            size used for susceptibility adjustment.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        tuple
            A tuple containing two arrays: all_I_t, an array of
            latent infections at each time step and all_S_t, an
            array of susceptible individuals at each time step.

        Raises
        ------
        ValueError
            If the length of the initial infections
            vector (I0) is less than the length of
            the generation interval.
        """

        # get initial infections
        I0_samples = self.I0.sample()
        I0 = I0_samples[0].value
        logging.debug(f"I0 samples: {I0}")
        # reverse generation interval (recency)
        gen_int_rev = jnp.flip(gen_int)
        if I0.size < gen_int.size:
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.size}, "
                f"generation interval length: {gen_int.size}."
            )
        recent_I0 = I0[-gen_int_rev.size :]
        # sample the initial susceptible population proportion S_{v-1} / P from prior
        init_S_proportion = numpyro.sample(
            "S_v_minus_1_over_P", self.susceptibility_prior
        )
        logging.debug(f"Initial susceptible proportion: {init_S_proportion}")
        # calculate initial susceptible population S_{v-1}
        init_S = init_S_proportion * P

        def update_infections(carry, Rt):
            S_t, I_recent = carry
            # compute raw infections
            i_raw_t = Rt * jnp.dot(I_recent, gen_int_rev)
            # apply the logistic susceptibility adjustment to a potential new incidence
            i_t = logistic_susceptibility_adjustment(
                I_raw_t=i_raw_t, frac_susceptible=S_t / P, n_population=P
            )
            # update susceptible population
            S_t -= i_t
            # update infections
            I_recent = jnp.concatenate([I_recent[:-1], jnp.array([i_t])])
            return (S_t, I_recent), i_t

        # initial carry state
        init_carry = (init_S, recent_I0)
        # scan to iterate over time steps and update infections
        (all_S_t, _), all_I_t = numpyro.contrib.control_flow.scan(
            update_infections, init_carry, Rt
        )
        logging.debug(f"All infections: {all_I_t}")
        logging.debug(f"All susceptibles: {all_S_t}")
        return all_I_t, all_S_t
