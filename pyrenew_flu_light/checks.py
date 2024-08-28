"""
Methods to verify pathing and existence
of certain files for use of pyrenew-flu-light.
"""

import os

import polar as pl
import toml


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


def load_data(
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
    check_file_path_valid(file_path=data_path)
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
    return data


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


def ensure_output_directory(args: dict[str, any]):  # numpydoc ignore=GL08
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
