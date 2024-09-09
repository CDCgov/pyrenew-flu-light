"""
Functions to handle tacking on data for the
non-observation period and also data for the
post-observation period (used in forecasting).
"""

from datetime import datetime, timedelta

import polars as pl

import pyrenew_flu_light


def add_post_observation_period(
    dataset: pl.DataFrame, n_post_observation_days: int
) -> pl.DataFrame:  # numpydoc ignore=RT01
    """
    Receives a dataframe that is filtered down to a
    particular jurisdiction, that has pre-observation
    data, and adds new rows to the end of the dataframe
    for the post-observation (forecasting) period.
    """

    # calculate the dates from the latest date in the dataframe
    max_date = dataset["date"].max()
    post_observation_dates = [
        (max_date + timedelta(days=i))
        for i in range(1, n_post_observation_days + 1)
    ]

    # get the days of the week (e.g. Fri) from the calculated dates
    day_of_weeks = (
        pl.Series(post_observation_dates)
        .dt.strftime("%a")
        .alias("day_of_week")
    )
    weekends = day_of_weeks.is_in(["Sat", "Sun"])

    # calculate the epiweeks and epiyears, which might not evenly mod 7
    last_epiweek = dataset["epiweek"][-1]
    epiweek_counts = dataset.filter(pl.col("epiweek") == last_epiweek).shape[0]
    epiweeks = [last_epiweek] * (7 - epiweek_counts) + [
        (last_epiweek + 1 + (i // 7))
        for i in range(n_post_observation_days - (7 - epiweek_counts))
    ]
    last_epiyear = dataset["epiyear"][-1]
    epiyears = [
        last_epiyear if epiweek <= 52 else last_epiyear + 1
        for epiweek in epiweeks
    ]
    epiweeks = [
        epiweek if epiweek <= 52 else epiweek - 52 for epiweek in epiweeks
    ]

    # calculate week values
    last_week = dataset["week"][-1]
    week_counts = dataset.filter(pl.col("week") == last_week).shape[0]
    weeks = [last_week] * (7 - week_counts) + [
        (last_week + 1 + (i // 7))
        for i in range(n_post_observation_days - (7 - week_counts))
    ]
    weeks = [week if week <= 52 else week - 52 for week in weeks]

    # calculate holiday series
    holidays = [
        datetime.strptime(elt, "%Y-%m-%d")
        for elt in pyrenew_flu_light.HOLIDAYS
    ]
    holidays_values = [date in holidays for date in post_observation_dates]
    post_holidays = [holiday + timedelta(days=1) for holiday in holidays]
    post_holiday_values = [
        date in post_holidays for date in post_observation_dates
    ]

    # fill in post-observation data entries, zero hospitalizations
    post_observation_data = pl.DataFrame(
        {
            "location": [dataset["location"][0]] * n_post_observation_days,
            "date": post_observation_dates,
            "hosp": [-9999] * n_post_observation_days,  # possible
            "epiweek": epiweeks,
            "epiyear": epiyears,
            "day_of_week": day_of_weeks,
            "is_weekend": weekends,
            "is_holiday": holidays_values,
            "is_post_holiday": post_holiday_values,
            "recency": [0] * n_post_observation_days,
            "week": weeks,
            "location_code": [dataset["location_code"][0]]
            * n_post_observation_days,
            "population": [dataset["population"][0]] * n_post_observation_days,
            "first_week_hosp": [dataset["first_week_hosp"][0]]
            * n_post_observation_days,
            "nonobservation_period": [False] * n_post_observation_days,
        }
    )

    # stack post_observation_data ONTO dataset
    merged_data = dataset.vstack(post_observation_data)
    return merged_data


def add_pre_observation_period(
    dataset: pl.DataFrame, n_pre_observation_days: int
) -> pl.DataFrame:  # numpydoc ignore=RT01
    """
    Receives a dataframe that is filtered down to a
    particular jurisdiction and adds new rows to the
    beginning of the dataframe for the non-observation
    period.
    """

    # create new nonobs column, set to False by default
    dataset = dataset.with_columns(
        pl.lit(False).alias("nonobservation_period")
    )

    # backcalculate the dates from the earliest date in the dataframe
    min_date = dataset["date"].min()
    pre_observation_dates = [
        (min_date - timedelta(days=i))
        for i in range(1, n_pre_observation_days + 1)
    ]
    pre_observation_dates.reverse()

    # get the days of the week (e.g. Fri) from the backcalculated dates
    day_of_weeks = (
        pl.Series(pre_observation_dates).dt.strftime("%a").alias("day_of_week")
    )
    weekends = day_of_weeks.is_in(["Sat", "Sun"])

    # backculate the epiweeks, which might not evenly mod 7
    first_epiweek = dataset["epiweek"][0]
    counts = dataset.filter(pl.col("epiweek") == first_epiweek).shape[0]
    epiweeks = [first_epiweek] * (7 - counts) + [
        (first_epiweek - 1 - (i // 7))
        for i in range(n_pre_observation_days - (7 - counts))
    ]
    epiweeks.reverse()

    # calculate holiday series
    holidays = [
        datetime.strptime(elt, "%Y-%m-%d")
        for elt in pyrenew_flu_light.HOLIDAYS
    ]
    holidays_values = [date in holidays for date in pre_observation_dates]
    post_holidays = [holiday + timedelta(days=1) for holiday in holidays]
    post_holiday_values = [
        date in post_holidays for date in pre_observation_dates
    ]

    # fill in pre-observation data entries, zero hospitalizations
    pre_observation_data = pl.DataFrame(
        {
            "location": [dataset["location"][0]] * n_pre_observation_days,
            "date": pre_observation_dates,
            "hosp": [0] * n_pre_observation_days,
            "epiweek": epiweeks,
            "epiyear": [dataset["epiyear"][0]] * n_pre_observation_days,
            "day_of_week": day_of_weeks,
            "is_weekend": weekends,
            "is_holiday": holidays_values,
            "is_post_holiday": post_holiday_values,
            "recency": [0] * n_pre_observation_days,
            "week": [dataset["week"][0]] * n_pre_observation_days,
            "location_code": [dataset["location_code"][0]]
            * n_pre_observation_days,
            "population": [dataset["population"][0]] * n_pre_observation_days,
            "first_week_hosp": [dataset["first_week_hosp"][0]]
            * n_pre_observation_days,
            "nonobservation_period": [True] * n_pre_observation_days,
        }
    )

    # stack dataset ONTO pre_observation_data
    merged_data = pre_observation_data.vstack(dataset)
    return merged_data
