"""
Functions for scoring PFL forecasts or comparing
posterior predictive distributions based on
historical observations.
"""

import arviz as az
import numpy as np
import polars as pl


def read_idata_from_netcdf(file_path: str) -> az.InferenceData:
    return az.from_netcdf(file_path)


def write_idata_netcdf_to_csv(idata: az.InferenceData, csv_path: str) -> None:
    df = idata.to_dataframe()
    df.to_csv(csv_path, index=False)


def get_vars_from_idata_object(
    idata: az.InferenceData, var_names: list[str]
) -> az.InferenceData:
    return az.extract(idata, var_names=var_names)


def write_idata_to_netcdf(idata: az.InferenceData, file_path: str) -> None:
    idata.to_netcdf(file_path)


def generate_draws_from_samples(
    post_p_fs: dict[str, list[float]],
    variable_name: str,
    dataset: pl.DataFrame,
    forecast_days: int,
    reporting_date: str,
):
    """
    Receive numpyro samples taken from a posterior
    predictive distribution for observed hospital
    admissions and converts these samples to dated
    draws.

    Parameters
    ----------
    post_p_fs
        Posterior predictive samples for the reference date.
    variable_name
        Which variable in the posterior predictive samples
        to convert to dated draws.
    dataset
        The full dataset used for forecasting, including the
        padded pre- and post-observation periods.
    forecast_days
        The number of days for which hospitalizations
        were forecasted.
    reporting_date
        The date in YYYY-MM-DD format for which the forecasts
        have been made.

    Returns
    -------
    pl.DataFrame
        Dated draws for the variable of interest.
    """
    # get dates for which forecasts were made
    data_and_forecast_dates = (
        dataset.select(pl.col("date")).to_numpy().flatten()
    )
    forecast_dates = data_and_forecast_dates[-forecast_days:]
    # posterior predictive <variable_name> forecasts
    forecasted_samples = post_p_fs[variable_name][:, -forecast_days:]
    # number of draws is the number of samples
    n_draws = post_p_fs.shape[0]
    # dated draws
    dated_draws = {
        ".draw": np.repeat(
            np.arange(0, n_draws), len(forecast_dates)
        ).tolist(),
        "date": forecast_dates * n_draws,
        f"{variable_name}": forecasted_samples.flatten().tolist(),
    }
    df = pl.DataFrame(dated_draws)
    df.write_csv("test.csv")
