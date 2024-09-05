"""
Methods to verify pathing and existence
of certain files for use of pyrenew-flu-light.
"""

import os

import toml


def check_file_path_valid(file_path: str) -> None:
    """
    Checks if a file path is valid. Used to check
    the entered data and config paths.

    Parameters
    ----------
    file_path : str
        Path to the file (usually data or config).

    Returns
    -------
    None
        Checks files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path '{file_path}' does not exist.")
    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"The path '{file_path}' is not a file.")
    return None


def load_config(config_path: str) -> dict[str, any]:
    """
    Attempts to load config toml file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file,
        read in via argparse.

    Returns
    -------
    config
        A dictionary of variables with
        associates values for `cfaepim`.
    """
    check_file_path_valid(file_path=config_path)
    try:
        config = toml.load(config_path)
    except toml.TomlDecodeError as e:
        raise ValueError(
            f"Failed to parse the TOML file at '{config_path}'; error: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while reading the TOML file: {e}"
        )
    return config


def ensure_output_directory(args: dict[str, any]):
    output_directory = "./output/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if args.historical_data:
        output_directory += f"Historical_{args.reporting_date}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    if not args.historical_data:
        output_directory += f"{args.reporting_date}/"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    return output_directory


def assert_historical_data_files_exist(
    reporting_date: str,
):
    data_directory = f"../model_comparison/data/{reporting_date}/"
    assert os.path.exists(
        data_directory
    ), f"Data directory {data_directory} does not exist."
    required_files = [
        f"{reporting_date}_clean_data.tsv",
        f"{reporting_date}_config.toml",
        f"{reporting_date}-cfarenewal-cfaepimlight.csv",
    ]
    for file in required_files:
        assert os.path.exists(
            os.path.join(data_directory, file)
        ), f"Required file {file} does not exist in {data_directory}."
    return data_directory
