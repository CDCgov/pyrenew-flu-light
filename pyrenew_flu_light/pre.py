"""
ETL system for pyrenew-flu-light.
"""

import json
import os

import polars as pl
import toml

import pyrenew_flu_light


def display_data(
    data: pl.DataFrame,
    n_row_count: int = 15,
    n_col_count: int = 5,
    first_only: bool = False,
    last_only: bool = False,
) -> None:
    """
    Display the columns and rows of
    a polars dataframe.

    Parameters
    ----------
    data : pl.DataFrame
        A polars dataframe.
    n_row_count : int, optional
        How many rows to print.
        Defaults to 15.
    n_col_count : int, optional
        How many columns to print.
        Defaults to 15.
    first_only : bool, optional
        If True, only display the first `n_row_count` rows. Defaults to False.
    last_only : bool, optional
        If True, only display the last `n_row_count` rows. Defaults to False.

    Returns
    -------
    None
        Displays data.
    """
    rows, cols = data.shape
    assert (
        1 <= n_col_count <= cols
    ), f"Must have reasonable column count; was type {n_col_count}"
    assert (
        1 <= n_row_count <= rows
    ), f"Must have reasonable row count; was type {n_row_count}"
    assert (
        first_only + last_only
    ) != 2, "Can only do one of last or first only."
    if first_only:
        data_to_display = data.head(n_row_count)
    elif last_only:
        data_to_display = data.tail(n_row_count)
    else:
        data_to_display = data.head(n_row_count)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_hide_column_data_types(True)
    with pl.Config(tbl_rows=n_row_count, tbl_cols=n_col_count):
        print(f"Dataset In Use For `cfaepim`:\n{data_to_display}\n")


def save_experiment_information(
    args: dict[str, any],
    config: dict[str, any],
    experiments_dir: str,
    command_line_args: str,
):
    """
    Creates pre-processing (informational) content
    of the experiments folder for a particular
    run. This content includes the configuration
    settings and information on the experiment.
    """

    info_path = os.path.join(experiments_dir, "information.txt")
    with open(info_path, "w") as f:
        # store name of experiment
        f.write(f"NAME: {args.exp_name}\n")
        # store current date
        f.write(f"DATE: {pyrenew_flu_light.CURRENT_DATE}\n")
        # store command line that produced the experiment
        f.write(f"ARGS: {command_line_args}")
        f.close()
    # store the config file as a json
    config_path = os.path.join(
        experiments_dir, f"params_{args.reporting_date}.json"
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        f.close()


def load_config_file(current_dir: str, reporting_date: str) -> dict[str, any]:
    """
    Attempt loading of config toml file.
    """
    top_level_dir = "pyrenew-flu-light"
    # get top-level path regardless of depth of call
    while not os.path.basename(
        current_dir
    ) == top_level_dir and current_dir != os.path.dirname(current_dir):
        current_dir = os.path.dirname(current_dir)
    # check that config directory exists
    config_dir = os.path.join(current_dir, "config")
    assert os.path.isdir(
        config_dir
    ), f"The folder {config_dir} does not exist when it should."
    # attempt to toml load the config file
    config_file = os.path.join(config_dir, f"params_{reporting_date}.toml")
    pyrenew_flu_light.check_file_path_valid(file_path=config_file)
    try:
        config = toml.load(config_file)
    except toml.TomlDecodeError as e:
        raise ValueError(
            f"Failed to parse the TOML file at '{config_file}'; error: {e}."
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while reading the TOML file: {e}."
        )
    return config


def load_saved_data(
    data_path: str,
    sep: str = "\t",
    schema_length: int = 10000,
) -> pl.DataFrame:
    """
    Loads historical (i.e., `.tsv` data generated
    `cfaepim` for a weekly run) data.

    Parameters
    ----------
    data_path : str
        The path to the tsv file to be read.
    sep : str, optional
        The separator between values in the
        data file. Defaults to tab-separated.
    schema_length : int, optional
        An approximation of the expected
        maximum number of rows. Defaults
        to 10000.

    Returns
    -------
    pl.DataFrame
        An unvetted polars dataframe of NHSN
        hospitalization data.
    """
    pyrenew_flu_light.check_file_path_valid(file_path=data_path)
    assert sep in [
        "\t",
        ",",
    ], f"Separator must be tabs or commas; was type {sep}"
    assert (
        7500 <= schema_length <= 25000
    ), f"Schema length must be reasonable; was type {schema_length}"
    data = pl.read_csv(
        data_path, separator=sep, infer_schema_length=schema_length
    )
    data = data.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    return data
