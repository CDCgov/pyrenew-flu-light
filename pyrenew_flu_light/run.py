"""
File to run pyrenew-flu-light from the
command line. Example, for running:
python3 run.py --reporting_date 2024-01-20 --regions NY --historical
"""

import argparse
import logging
import os
import sys

import arviz as az
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


def load_data_variables_for_model(
    jurisdiction: str,
    dataset: pl.DataFrame,
    args: dict[str, any],
    config: dict[str, any],
) -> dict[str, any]:
    """
    Returns a dictionary of values for
    use in instantiating the
    pyrenew-flu-light model.
    """
    # filter out jurisdiction data
    filtered_data_jurisdiction = dataset.filter(
        pl.col("location") == jurisdiction
    )
    # add the pre-observation period to the dataset
    filtered_data = pyrenew_flu_light.add_pre_observation_period(
        dataset=filtered_data_jurisdiction,
        n_pre_observation_days=config["n_pre_observation_days"],
    )
    # add post-observation period if forecast
    if args.forecast:
        filtered_data = pyrenew_flu_light.add_post_observation_period(
            dataset=filtered_data,
            n_post_observation_days=args.lookahead,
        )
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
    observed_hosp_admissions = (
        filtered_data.select(pl.col("hosp")).to_numpy().flatten()
    )
    # output dictionary
    out = {
        "population": population,
        "first_week_hosp": first_week_hosp,
        "week_indices": week_indices,
        "predictors": predictors,
        "observations": observed_hosp_admissions,
        "total_steps": week_indices.size,
    }
    return out


def instantiate_model(
    jurisdiction: str,
    dataset: pl.DataFrame,
    args: dict[str, any],
    config: dict[str, any],
):
    """
    Instantiate the pyrenew-flu-light model
    using the config and the variables derived
    from the jurisdiction's dataset.
    """
    # load model variables from jurisdiction dataset
    variables_from_data = load_data_variables_for_model(
        jurisdiction=jurisdiction, dataset=dataset, args=args, config=config
    )
    # define the number of time points, will change if forecasting, since
    # synthetic data some steps ahead has covariates
    steps_excluding_forecast = (
        variables_from_data["total_steps"] - args.lookahead
    )
    non_forecast_week_indices = variables_from_data["week_indices"][
        :steps_excluding_forecast
    ]
    non_forecast_predictors = variables_from_data["predictors"][
        :steps_excluding_forecast
    ]
    # get pyrenew flu light model
    model = pyrenew_flu_light.CFAEPIM_Model(
        config=config,  # NOTE: change to variable by variable param?
        population=variables_from_data["population"],
        week_indices=non_forecast_week_indices,
        first_week_hosp=variables_from_data["first_week_hosp"],
        predictors=non_forecast_predictors,
    )
    return variables_from_data, model


def get_samples_from_ran_model(model, config, steps, data_obs_hosp=None):
    # simulate data from model
    prior_predictive_samples = model.prior_predictive(
        n_steps=steps,
        numpyro_predictive_args={"num_samples": config["n_iter"]},
        rng_key=jax.random.key(config["seed"]),
    )
    # fit the model to the data
    posterior_predictive_samples = model.posterior_predictive(
        n_steps=steps,
        numpyro_predictive_args={"num_samples": config["n_iter"]},
        rng_key=jax.random.key(config["seed"]),
        data_observed_hosp_admissions=data_obs_hosp,
    )
    return prior_predictive_samples, posterior_predictive_samples


def run_pyrenew_flu_light_model(model, config, steps, data_obs_hosp):
    """
    Fit pyrenew-flu-light to data for a single
    jurisdiction.
    """
    # run the model
    model.run(
        rng_key=jax.random.key(config["seed"]),
        n_steps=steps,
        data_observed_hosp_admissions=data_obs_hosp,
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
        },
    )
    model.print_summary()
    return model.mcmc


def run_jurisdiction(
    jurisdiction: str,
    dataset: pl.DataFrame,
    args: dict[str, any],
    config: dict[str, any],
):
    # define the model and retrieve variables from data
    variables_from_data, model = instantiate_model(
        jurisdiction,
        dataset,
        args,
        config,
    )
    # define the observed hospital admissions data to exclude observations
    # in the synthetic data, of which there are n_post_observation_days worth
    steps_excluding_forecast = (
        variables_from_data["total_steps"] - args.lookahead
    )
    data_obs_hosp = variables_from_data["observations"][
        :steps_excluding_forecast
    ]
    # run the model, getting the samples
    model_mcmc = run_pyrenew_flu_light_model(
        model,
        config,
        steps=steps_excluding_forecast,
        data_obs_hosp=data_obs_hosp,
    )
    # different approach for forecasting
    if args.forecast:
        # reinstaniate another model
        for_variables_from_data, forecasting_model = instantiate_model(
            jurisdiction, dataset, args, config
        )
        # modify forecasting model mcmc object
        forecasting_model.mcmc = model_mcmc  # NOTE: this might change
        # get samples from the forecasting model
        for_priorp, for_postp = get_samples_from_ran_model(
            forecasting_model,
            config,
            steps=steps_excluding_forecast,
            data_obs_hosp=None,
        )
        # define arviz object
        idata = az.from_numpyro(
            forecasting_model.mcmc,
            posterior_predictive=for_postp,
            prior=for_priorp,
        )
        return idata
    # get samples for non-forecasting model
    priorp, postp = get_samples_from_ran_model(
        model, config, steps=steps_excluding_forecast, data_obs_hosp=None
    )
    idata = az.from_numpyro(
        model_mcmc,
        posterior_predictive=postp,
        prior=priorp,
    )
    return idata


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
        # make sure an experiment name exists
        assert isinstance(
            args.exp_name, str
        ), "A short experiment name must be provided to use the historical mode."
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
        # create new experiment or check it is for comparison
        experiments_dir, samples_dir = pyrenew_flu_light.check_experiments(
            args=args, current_dir=current_dir
        )
        # save data relevant data and information to experiment folder
        pyrenew_flu_light.save_experiment_information(
            args=args,
            config=config,
            experiments_dir=experiments_dir,
            command_line_args=" ".join(sys.argv),
        )
        # iterate over jurisdictions selected, running the model
        for jurisdiction in args.regions:
            # name for saving jurisdiction
            save_name = f"{jurisdiction}_{args.reporting_date}.csv"
            save_path = os.path.join(samples_dir, save_name)
            # check if the file does not already exist
            if not os.path.exists(save_path):
                # retrieve fit or forecast
                idata = run_jurisdiction(
                    jurisdiction=jurisdiction,
                    dataset=influenza_hosp_data,
                    args=args,
                    config=config,
                )
                # convert to dataframe and then to csv
                idata.to_dataframe().to_csv(save_path, index=False)


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
    parser.add_argument(
        "--exp_name",
        type=str,
        help="The name of a given experiment.",
    )
    args = parser.parse_args()
    main(args)
