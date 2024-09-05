"""
Comparisons made between posterior samples.
"""

import polars as pl
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


def quantilize_forecasts(
    samples_dict,
    state_abbr,
    start_date,
    end_date,
    fitting_data,
    output_path,
    reference_date,
):
    pandas2ri.activate()
    forecasttools = importr("forecasttools")
    # dplyr = importr("dplyr")
    # tidyr = importr("tidyr")
    # cli = importr("cli")

    posterior_samples = pl.DataFrame(samples_dict)
    posterior_samples_pd = posterior_samples.to_pandas()
    r_posterior_samples = pandas2ri.py2rpy(posterior_samples_pd)

    fitting_data_pd = fitting_data.to_pandas()
    r_fitting_data = pandas2ri.py2rpy(fitting_data_pd)

    results_list = ro.ListVector({state_abbr: r_posterior_samples})

    horizons = ro.IntVector([-1, 0, 1, 2, 3])

    forecast_output = forecasttools.forecast_and_output_flusight(
        data=r_fitting_data,
        results=results_list,
        output_path=output_path,
        reference_date=reference_date,
        horizons=horizons,
        seed=62352,
    )

    forecast_output_pd = pandas2ri.rpy2py(forecast_output)
    forecast_output_pl = pl.from_pandas(forecast_output_pd)
    print(forecast_output_pl)
