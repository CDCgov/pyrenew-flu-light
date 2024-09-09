import argparse
import logging
import os

import jax
import numpyro
import polars as pl

import pyrenew_flu_light


def process_jurisdictions(value: str) -> list[str]:
    """
    Function for customized argparse argument for
    entering jurisdictions, including avoiding
    certain jurisdictions via use of "not".
    """
    try:
        if value.lower() == "all":
            return pyrenew_flu_light.JURISDICTIONS
        elif value.lower().startswith("not:"):
            exclude = value[4:].split(",")
            return [
                state
                for state in pyrenew_flu_light.JURISDICTIONS
                if state not in exclude
            ]
        else:
            return value.split(",")
    except AttributeError:
        raise AttributeError("Invalid input: 'value' must be a string.")
    except NameError:
        raise NameError(
            "Ensure 'pyrenew_flu_light' and 'JURISDICTIONS' are imported correctly."
        )
    except TypeError:
        raise TypeError(
            "Input should be a string. The provided value is not compatible."
        )
    except ImportError:
        raise ImportError("'pyrenew_flu_light' could not be imported.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def run_single_jurisdiction(
    jurisdiction: str,
    dataset: pl.DataFrame,
    config: dict[str, any],
    forecasting: bool = False,
    n_post_observation_days: int = 0,
):
    """
    Runs the ported `cfaepim` model on a single
    jurisdiction. Pre- and post-observation data
    for the Rt burn in and for forecasting,
    respectively, is done before the prior predictive,
    posterior, and posterior predictive samples
    are returned.

    Parameters
    ----------
    jurisdiction : str
        The jurisdiction.
    dataset : pl.DataFrame
        The incidence data of interest.
    config : dict[str, any]
        A configuration file for the model.
    forecasting : bool, optional
        Whether or not forecasts are being made.
        Defaults to True.
    n_post_observation_days : int, optional
        The number of days to look ahead. Defaults
        to 0 if not forecasting.

    Returns
    -------
    tuple
        A tuple of prior predictive, posterior, and
        posterior predictive samples.
    """
    # filter data to be the jurisdiction alone
    filtered_data_jurisdiction = dataset.filter(
        pl.col("location") == jurisdiction
    )

    # add the pre-observation period to the dataset
    filtered_data = pyrenew_flu_light.add_pre_observation_period(
        dataset=filtered_data_jurisdiction,
        n_pre_observation_days=config["n_pre_observation_days"],
    )

    logging.info(f"{jurisdiction}: Dataset w/ pre-observation ready.")

    if forecasting:
        # add the post-observation period if forecasting
        filtered_data = pyrenew_flu_light.add_post_observation_period(
            dataset=filtered_data,
            n_post_observation_days=n_post_observation_days,
        )
        logging.info(f"{jurisdiction}: Dataset w/ post-observation ready.")

    # extract jurisdiction population
    population = (
        filtered_data.select(pl.col("population"))
        .unique()
        .to_numpy()
        .flatten()
    )[0]

    # extract indices for weeks for Rt broadcasting (weekly to daily)
    week_indices = filtered_data.select(pl.col("week")).to_numpy().flatten()

    # extract first week hospitalizations for infections seeding
    first_week_hosp = (
        filtered_data.select(pl.col("first_week_hosp"))
        .unique()
        .to_numpy()
        .flatten()
    )[0]

    # extract covariates (typically weekday, holidays, nonobs period)
    day_of_week_covariate = (
        filtered_data.select(pl.col("day_of_week"))
        .to_dummies()
        .select(pl.exclude("day_of_week_Thu"))
    )
    remaining_covariates = filtered_data.select(
        ["is_holiday", "is_post_holiday", "nonobservation_period"]
    )
    covariates = pl.concat(
        [day_of_week_covariate, remaining_covariates], how="horizontal"
    )
    predictors = covariates.to_numpy()

    # extract observation hospital admissions
    # NOTE: from filtered_data_jurisdiction, not filtered_data, which has null hosp
    observed_hosp_admissions = (
        filtered_data.select(pl.col("hosp")).to_numpy().flatten()
    )

    logging.info(f"{jurisdiction}: Variables extracted from dataset.")

    # instantiate CFAEPIM model (for fitting)
    total_steps = week_indices.size
    steps_excluding_forecast = total_steps - n_post_observation_days
    cfaepim_MSR_fit = pyrenew_flu_light.CFAEPIM_Model(
        config=config,
        population=population,
        week_indices=week_indices[:steps_excluding_forecast],
        first_week_hosp=first_week_hosp,
        predictors=predictors[:steps_excluding_forecast],
    )

    logging.info(f"{jurisdiction}: CFAEPIM model instantiated (fitting)!")

    # run the CFAEPIM model
    cfaepim_MSR_fit.run(
        rng_key=jax.random.key(config["seed"]),
        n_steps=steps_excluding_forecast,
        data_observed_hosp_admissions=observed_hosp_admissions[
            :steps_excluding_forecast
        ],
        num_warmup=config["n_warmup"],
        num_samples=config["n_iter"],
        nuts_args={
            "target_accept_prob": config["adapt_delta"],
            "max_tree_depth": config["max_treedepth"],
            "init_strategy": numpyro.infer.init_to_sample,
            "find_heuristic_step_size": True,
        },
        mcmc_args={
            "num_chains": config["n_chains"],
            "progress_bar": True,
        },  # progress_bar False if use vmap
    )

    logging.info(f"{jurisdiction}: CFAEPIM model (fitting) ran!")

    cfaepim_MSR_fit.print_summary()

    # prior predictive simulation samples
    prior_predictive_sim_samples = cfaepim_MSR_fit.prior_predictive(
        n_steps=steps_excluding_forecast,
        numpyro_predictive_args={"num_samples": config["n_iter"]},
        rng_key=jax.random.key(config["seed"]),
    )

    logging.info(f"{jurisdiction}: Prior predictive simulation complete.")

    # posterior predictive simulation samples
    posterior_predictive_sim_samples = cfaepim_MSR_fit.posterior_predictive(
        n_steps=steps_excluding_forecast,
        numpyro_predictive_args={"num_samples": config["n_iter"]},
        rng_key=jax.random.key(config["seed"]),
        data_observed_hosp_admissions=None,
    )

    # specific draws, might save, but most often don't save spread_draws
    # rds is rough equivalent of using pickle dump()
    # sensible use of "pickle"; don't open untrusted pickles from PR
    # if arviz has fully re-instantiatible objects available, use
    # use of exp folders within output
    # output
    #   exp_01orUUID
    #      results.txt (could be cfaepim v. PFL)
    #      some_plots.pdf
    #   exp_02orUUID
    #       results.txt (could be PFL v. PFL)
    #       some_plots.pdf

    logging.info(f"{jurisdiction}: Posterior predictive simulation complete.")

    # posterior predictive forecasting samples
    if forecasting:
        cfaepim_MSR_for = pyrenew_flu_light.CFAEPIM_Model(
            config=config,
            population=population,
            week_indices=week_indices,
            first_week_hosp=first_week_hosp,
            predictors=predictors,
        )

        # run the CFAEPIM model (forecasting, required to do so
        # single `posterior_predictive` gets sames (need self.mcmc)
        # from passed model);
        # ISSUE: inv()
        # PR: sample() + OOP behavior & statefulness
        cfaepim_MSR_for.mcmc = cfaepim_MSR_fit.mcmc

        posterior_predictive_for_samples = (
            cfaepim_MSR_for.posterior_predictive(
                n_steps=total_steps,
                numpyro_predictive_args={"num_samples": config["n_iter"]},
                rng_key=jax.random.key(config["seed"]),
                data_observed_hosp_admissions=None,
            )
        )

        logging.info(
            f"{jurisdiction}: Posterior predictive forecasts complete."
        )

        return (
            cfaepim_MSR_for,
            observed_hosp_admissions,
            prior_predictive_sim_samples,
            posterior_predictive_sim_samples,
            posterior_predictive_for_samples,
            filtered_data,
        )
    else:
        posterior_predictive_for_samples = None

    return (
        cfaepim_MSR_fit,
        observed_hosp_admissions,
        prior_predictive_sim_samples,
        posterior_predictive_sim_samples,
        posterior_predictive_for_samples,
        filtered_data,
    )


def main(args):

    logging.info("Initiating Pyrenew Flu Light...")
    # determine and set number of CPU cores
    numpyro.set_platform("cpu")
    num_cores = os.cpu_count()
    numpyro.set_host_device_count(num_cores - (num_cores - 3))
    logging.info("Number of cores set.")
    # ensure proper output directory exists, depending on mode
    current_dir = os.path.abspath(os.path.dirname(__file__))
    pyrenew_flu_light.check_output_directories(
        args=args, current_dir=current_dir
    )
    # active mode, likely using NSSP, NOTE: not yet implemented
    if not args.historical_data:
        pass
    # mode for using historical cfaepim datasets
    if args.historical_data:
        # check that historical cfaepim data exists for given reporting date
        data_file_path = pyrenew_flu_light.check_historical_data_files(
            current_dir=current_dir, reporting_date=args.reporting_date
        )
        # load config file from toml
        config = pyrenew_flu_light.load_config_file(
            current_dir=current_dir, reporting_date=args.reporting_date
        )
        logging.info("Configuration (historical) loaded.")
        # load the historical hospitalization data
        influenza_hosp_data = pyrenew_flu_light.load_saved_data(
            data_path=data_file_path, sep="\t"
        )
        logging.info("Historical NHSN influenza incidence data loaded.")
        # iterate over jurisdictions selected, running the model
        for jurisdiction in args.regions:

            # NOTE: subject to change wrt what is returned here
            (model, obs, prior_p_ss, post_p_ss, post_p_fs, dataset) = (
                run_single_jurisdiction(
                    jurisdiction=jurisdiction,
                    dataset=influenza_hosp_data,
                    config=config,
                    forecasting=args.forecast,
                    n_post_observation_days=args.lookahead,
                )
            )

            # create and save data as arviz inference data object
            # run_output = az.from_numpyro(
            #     mcmc=model.mcmc,
            #     prior=prior_p_ss,
            #     posterior_predictive=post_p_fs,
            #     coords={"school": np.arange(eight_school_data["J"])},
            #     dims={"theta": ["school"]},
            # )

            # NOTE: subject to change wrt arviz idata usage
            # post_p_fs_as_draws = pyrenew_flu_light.generate_draws_from_samples(
            #     post_p_fs=post_p_fs,
            #     variable_name="negbinom_rv",
            #     dataset=dataset,
            #     forecast_days=forecast_days,
            #     reporting_date=args.reporting_date,
            # )


if __name__ == "__main__":

    # e.g. python3 tut_epim_port_msr.py
    # --reporting_date 2024-01-20 --regions all --historical --forecast
    # python3 run.py --reporting_date 2024-01-20 --regions NY --historical --forecast

    # use argparse for command line running
    parser = argparse.ArgumentParser(
        description="Forecast, simulate, and analyze the CFAEPIM model."
    )
    parser.add_argument(
        "--regions",
        type=process_jurisdictions,
        required=True,
        help="Specify jurisdictions as a comma-separated list. Use 'all' for all states, or 'not:state1,state2' to exclude specific states, or 'state1,state2' for specific states.",
    )
    parser.add_argument(
        "--reporting_date",
        type=str,
        required=True,
        help="The reporting date.",
    )
    parser.add_argument(
        "--historical_data",
        action="store_true",
        help="Load model weights before training.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Whether to make a forecast.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=28,
        help="The number of days to forecast ahead.",
    )
    args = parser.parse_args()
    main(args)
