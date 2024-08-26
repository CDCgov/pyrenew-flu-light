# import jax
# import jax.numpy as jnp
# import numpyro
# import numpyro.distributions as dist
# from jax.typing import ArrayLike
# from pyrenew.deterministic import DeterministicPMF
# from pyrenew.latent import (
#     InfectionInitializationProcess,
#     InitializeInfectionsFromVec,
# )
# from pyrenew.metaclass import (
#     DistributionalRV,
#     Model,
#     SampledValue,
# )


# class CFAEPIM_Model_Sample(NamedTuple):  # numpydoc ignore=GL08
#     Rts: SampledValue | None = None
#     latent_infections: SampledValue | None = None
#     susceptibles: SampledValue | None = None
#     ascertainment_rates: SampledValue | None = None
#     expected_hospitalizations: SampledValue | None = None

#     def __repr__(self):
#         return (
#             f"CFAEPIM_Model_Sample(Rts={self.Rts}, "
#             f"latent_infections={self.latent_infections}, "
#             f"susceptibles={self.susceptibles}, "
#             f"ascertainment_rates={self.ascertainment_rates}, "
#             f"expected_hospitalizations={self.expected_hospitalizations}"
#         )


# class CFAEPIM_Model(Model):
#     """
#     CFAEPIM Model class for epidemic inference,
#     ported over from `cfaepim`. This class handles the
#     initialization and sampling of the CFAEPIM model,
#     including the transmission process, infection process,
#     and observation process.

#     Parameters
#     ----------
#     config : dict[str, any]
#         Configuration dictionary containing model parameters.
#     population : int
#         Total population size.
#     week_indices : ArrayLike
#         Array of week indices corresponding to the time steps.
#     first_week_hosp : int
#         Number of hospitalizations in the first week.
#     predictors : list[int]
#         List of predictors (covariates) for the model.
#     data_observed_hosp_admissions : pl.DataFrame
#         DataFrame containing observed hospital admissions data.
#     """

#     def __init__(
#         self,
#         config: dict[str, any],
#         population: int,
#         week_indices: ArrayLike,
#         first_week_hosp: int,
#         predictors: list[int],
#     ):  # numpydoc ignore=GL08
#         self.population = population
#         self.week_indices = week_indices
#         self.first_week_hosp = first_week_hosp
#         self.predictors = predictors

#         self.config = config
#         for key, value in config.items():
#             setattr(self, key, value)

#         # transmission: generation time distribution
#         self.pmf_array = jnp.array(self.generation_time_dist)
#         self.gen_int = DeterministicPMF(name="gen_int", value=self.pmf_array)
#         # update: record in sample ought to be False by default

#         # transmission: prior for RW intercept
#         self.intercept_RW_prior = dist.Normal(
#             self.rt_intercept_prior_mode, self.rt_intercept_prior_scale
#         )

#         # transmission: Rt process
#         self.Rt_process = CFAEPIM_Rt(
#             intercept_RW_prior=self.intercept_RW_prior,
#             max_rt=self.max_rt,
#             gamma_RW_prior_scale=self.weekly_rw_prior_scale,
#             week_indices=self.week_indices,
#         )

#         # infections: get value rate for infection seeding (initialization)
#         self.mean_inf_val = (
#             self.inf_model_prior_infections_per_capita * self.population
#         ) + (self.first_week_hosp / (self.ihr_intercept_prior_mode * 7))

#         # infections: initial infections
#         self.I0 = InfectionInitializationProcess(
#             name="I0_initialization",
#             I_pre_init_rv=DistributionalRV(
#                 name="I0",
#                 dist=dist.Exponential(rate=1 / self.mean_inf_val).expand(
#                     [self.inf_model_seed_days]
#                 ),
#             ),
#             infection_init_method=InitializeInfectionsFromVec(
#                 n_timepoints=self.inf_model_seed_days
#             ),
#             t_unit=1,
#         )

#         # infections: susceptibility depletion prior
#         # update: truncated Normal needed here, done
#         # "under the hood" in Epidemia, use Beta for the
#         # time being.
#         # self.susceptibility_prior = dist.Beta(
#         #     1
#         #     + (
#         #         self.susceptible_fraction_prior_mode
#         #         / self.susceptible_fraction_prior_scale
#         #     ),
#         #     1
#         #     + (1 - self.susceptible_fraction_prior_mode)
#         #     / self.susceptible_fraction_prior_scale,
#         # )
#         # now:
#         self.susceptibility_prior = dist.TruncatedNormal(
#             self.susceptible_fraction_prior_mode,
#             self.susceptible_fraction_prior_scale,
#             low=0.0,
#         )

#         # infections component
#         self.infections = CFAEPIM_Infections(
#             I0=self.I0, susceptibility_prior=self.susceptibility_prior
#         )

#         # observations: negative binomial concentration prior
#         self.nb_concentration_prior = dist.Normal(
#             self.reciprocal_dispersion_prior_mode,
#             self.reciprocal_dispersion_prior_scale,
#         )

#         # observations: instantaneous ascertainment rate prior
#         self.alpha_prior_dist = dist.Normal(
#             self.ihr_intercept_prior_mode, self.ihr_intercept_prior_scale
#         )

#         # observations: prior on covariate coefficients
#         self.coefficient_priors = dist.Normal(
#             loc=jnp.array(
#                 self.day_of_week_effect_prior_modes
#                 + [
#                     self.holiday_eff_prior_mode,
#                     self.post_holiday_eff_prior_mode,
#                     self.non_obs_effect_prior_mode,
#                 ]
#             ),
#             scale=jnp.array(
#                 self.day_of_week_effect_prior_scales
#                 + [
#                     self.holiday_eff_prior_scale,
#                     self.post_holiday_eff_prior_scale,
#                     self.non_obs_effect_prior_scale,
#                 ]
#             ),
#         )

#         # observations component
#         self.obs_process = CFAEPIM_Observation(
#             predictors=self.predictors,
#             alpha_prior_dist=self.alpha_prior_dist,
#             coefficient_priors=self.coefficient_priors,
#             nb_concentration_prior=self.nb_concentration_prior,
#         )

#     @staticmethod
#     def validate(
#         population: any,
#         week_indices: any,
#         first_week_hosp: any,
#         predictors: any,
#     ) -> None:
#         """
#         Validate the parameters of the CFAEPIM model.

#         This method checks that all necessary parameters and priors are correctly specified.
#         If any parameter is invalid, an appropriate error is raised.

#         Raises
#         ------
#         ValueError
#             If any parameter is missing or invalid.
#         """
#         if not isinstance(population, int) or population <= 0:
#             raise ValueError("Population must be a positive integer.")
#         if not isinstance(week_indices, jax.ndarray):
#             raise ValueError("Week indices must be an array-like structure.")
#         if not isinstance(first_week_hosp, int) or first_week_hosp < 0:
#             raise ValueError(
#                 "First week hospitalizations must be a non-negative integer."
#             )
#         if not isinstance(predictors, jnp.ndarray):
#             raise ValueError("Predictors must be a list of integers.")

#     def sample(
#         self,
#         n_steps: int,
#         data_observed_hosp_admissions: ArrayLike = None,
#         **kwargs,
#     ) -> tuple:
#         # shift towards "reduced statefulness", include here week indices &
#         # predictors which might change; for the same model and different
#         # models.
#         """
#         Samples the reproduction numbers, generation interval,
#         infections, and hospitalizations from the CFAEPIM model.

#         Parameters
#         ----------
#         n_steps : int
#             Number of time steps to sample.
#         data_observed_hosp_admissions : ArrayLike, optional
#             Observation hospital admissions.
#             Defaults to None.
#         **kwargs : dict, optional
#             Additional keyword arguments passed through to
#             internal sample calls, should there be any.

#         Returns
#         -------
#         CFAEPIM_Model_Sample
#             A named tuple containing sampled values for reproduction numbers,
#             latent infections, susceptibles, ascertainment rates, expected
#             hospitalizations, and observed hospital admissions.
#         """
#         sampled_Rts = self.Rt_process.sample(n_steps=n_steps)
#         sampled_gen_int = self.gen_int.sample(record=False)
#         all_I_t, all_S_t = self.infections.sample(
#             Rt=sampled_Rts,
#             gen_int=sampled_gen_int[0].value,
#             P=self.population,
#         )
#         sampled_alphas, expected_hosps = self.obs_process.sample(
#             infections=all_I_t,
#             inf_to_hosp_dist=jnp.array(self.inf_to_hosp_dist),
#         )
#         # observed_hosp_admissions = self.obs_process.nb_observation.sample(
#         #     mu=expected_hosps,
#         #     obs=data_observed_hosp_admissions,
#         #     **kwargs,
#         # )
#         numpyro.deterministic("Rts", sampled_Rts)
#         numpyro.deterministic("latent_infections", all_I_t)
#         numpyro.deterministic("susceptibles", all_S_t)
#         numpyro.deterministic("alphas", sampled_alphas)
#         numpyro.deterministic("expected_hospitalizations", expected_hosps)
#         return CFAEPIM_Model_Sample(
#             Rts=sampled_Rts,
#             latent_infections=all_I_t,
#             susceptibles=all_S_t,
#             ascertainment_rates=sampled_alphas,
#             expected_hospitalizations=expected_hosps,
#         )
