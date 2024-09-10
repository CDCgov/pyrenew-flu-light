"""
Aggregate relevant classes, functions, and
module-level constants for running
pyrenew-flu-light.
"""

from datetime import datetime as dt

import pytz
from checks import (
    check_experiments,
    check_file_path_valid,
    check_historical_data_files,
    check_output_directories,
)
from comp_inf import CFAEPIM_Infections
from comp_obs import CFAEPIM_Observation
from comp_tran import CFAEPIM_Rt
from model import CFAEPIM_Model
from pad import add_post_observation_period, add_pre_observation_period
from post import generate_draws_from_samples
from pre import load_config_file, load_saved_data, save_experiment_information
from run import (
    get_samples_from_ran_model,
    instantiate_model,
    load_data_variables_for_model,
    run_jurisdiction,
    run_pyrenew_flu_light_model,
)

# the 50 states in the United States
JURISDICTIONS = [
    "AK",
    "AL",
    "AR",
    "AZ",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "IA",
    "ID",
    "IL",
    "IN",
    "KS",
    "KY",
    "LA",
    "MA",
    "MD",
    "ME",
    "MI",
    "MN",
    "MO",
    "MS",
    "MT",
    "NC",
    "ND",
    "NE",
    "NH",
    "NJ",
    "NM",
    "NV",
    "NY",
    "OH",
    "OK",
    "OR",
    "PA",
    "PR",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "US",
    "UT",
    "VA",
    "VI",
    "VT",
    "WA",
    "WI",
    "WV",
    "WY",
]


# holidays as model covariates
HOLIDAYS = ["2023-11-23", "2023-12-25", "2023-12-31", "2024-01-01"]


# current ISO 8601 formatted EST timezone date
est = pytz.timezone("US/Eastern")
CURRENT_DATE = dt.now(est).strftime("%Y-%m-%d")

__all__ = [
    "JURISDICTIONS",
    "CURRENT_DATE",
    "HOLIDAYS",
    "CFAEPIM_Infections",
    "CFAEPIM_Observation",
    "CFAEPIM_Rt",
    "CFAEPIM_Model",
    "add_post_observation_period",
    "add_pre_observation_period",
    "check_file_path_valid",
    "check_historical_data_files",
    "check_output_directories",
    "check_experiments",
    "load_config_file",
    "load_saved_data",
    "save_experiment_information",
    "generate_draws_from_samples",
    "load_data_variables_for_model",
    "instantiate_model",
    "get_samples_from_ran_model",
    "run_pyrenew_flu_light_model",
    "run_jurisdiction",
]
