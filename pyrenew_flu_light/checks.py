"""
Methods to verify pathing and existence
of certain files for use of pyrenew-flu-light.
"""

import os


def check_file_path_valid(file_path: str) -> None:
    """
    Checks if a file path is valid. Used to check
    the entered data and config paths.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path '{file_path}' does not exist.")
    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"The path '{file_path}' is not a file.")
    return None


def check_experiments(args: dict[str, any], current_dir: str) -> None:
    """
    Finds the experiments folder for the model
    comparison output using historical data and
    creates the correct folder.
    """
    top_level_dir = "pyrenew-flu-light"
    # get top-level path regardless of depth of call
    while not os.path.basename(
        current_dir
    ) == top_level_dir and current_dir != os.path.dirname(current_dir):
        current_dir = os.path.dirname(current_dir)
    # look in model comparison, output dir for experiments
    # no need to check these dirs since check_historical_data_files
    # already does so
    model_comparison_dir = os.path.join(current_dir, "model_comparison")
    # define name of current experiment
    experiments_dir = os.path.join(model_comparison_dir, args.exp_name)
    # check if experiments folder exists, create if not
    os.makedirs(experiments_dir, exist_ok=True)
    # create samples folder
    samples_dir = os.path.join(experiments_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    return experiments_dir, samples_dir


def check_output_directories(args: dict[str, any], current_dir: str) -> None:
    """
    Checks for an output folder if in active mode and
    checks for a model_comparison folder and an output
    subfolder if in historical mode.
    """

    top_level_dir = "pyrenew-flu-light"
    # active mode, likely using NSSP
    if not args.historical_data:
        # get top-level path regardless of depth of call
        while not os.path.basename(
            current_dir
        ) == top_level_dir and current_dir != os.path.dirname(current_dir):
            current_dir = os.path.dirname(current_dir)
        active_output_dir = os.path.join(current_dir, "output")
        # make output folder if it does not exist
        os.makedirs(active_output_dir, exist_ok=True)
    # historical mode, using historical data
    if args.historical_data:
        # get top-level path regardless of depth of call
        while not os.path.basename(
            current_dir
        ) == top_level_dir and current_dir != os.path.dirname(current_dir):
            current_dir = os.path.dirname(current_dir)
        hist_output_dir_upper = os.path.join(current_dir, "model_comparison")
        # make model_comparison folder, if it does not exist
        os.makedirs(hist_output_dir_upper, exist_ok=True)
        # make output folder in model_comparison folder, if it does not exist
        hist_output_dir_lower = os.path.join(hist_output_dir_upper, "output")
        os.makedirs(hist_output_dir_lower, exist_ok=True)


def check_historical_data_files(
    reporting_date: str,
    current_dir: str,
):
    """
    For the historical mode, make sure the model
    comparison folder exists with data and config
    subfolders. This is run for each reporting
    date experiment.
    """

    top_level_dir = "pyrenew-flu-light"
    # get top-level path regardless of depth of call
    while not os.path.basename(
        current_dir
    ) == top_level_dir and current_dir != os.path.dirname(current_dir):
        current_dir = os.path.dirname(current_dir)
    # check if model_comparison folder exists at top-level
    model_comparison_dir = os.path.join(current_dir, "model_comparison")
    assert os.path.isdir(
        model_comparison_dir
    ), f"The folder {model_comparison_dir} does not exist when it should."
    # check if data folder exists within model_comparison folder
    data_dir = os.path.join(model_comparison_dir, "data")
    assert os.path.isdir(
        data_dir
    ), f"The folder {data_dir} does not exist when it should."
    # check that required files are in the data folder
    required_data_files = [
        f"{reporting_date}_clean_data.tsv",
        f"{reporting_date}_config.toml",
        f"{reporting_date}-cfarenewal-cfaepimlight.csv",
    ]
    for file in required_data_files:
        data_dir_reporting = os.path.join(data_dir, reporting_date)
        assert os.path.exists(
            data_dir_reporting
        ), f"Required file {file} does not exist in {data_dir_reporting}."
    # return the data directory for the reporting date
    data_file_path = os.path.join(
        data_dir_reporting, f"{reporting_date}_clean_data.tsv"
    )
    return data_file_path
