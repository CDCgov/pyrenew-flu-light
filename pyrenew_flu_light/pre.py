# -*- coding: utf-8 -*-

"""
ETL system for pyrenew-flu-light.
"""

import polars as pl

from pyrenew_flu_light import check_file_path_valid


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
    data = data.with_columns(pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"))
    return data
