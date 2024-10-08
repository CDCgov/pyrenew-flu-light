#' synthetic_forecast_data
#'
#' Given a dataframe of timeseries data used
#' for epidemia model fitting,
#' produce a synthetic dataframe
#' in the same format to use for
#' forward projection. This is required
#' because the rstanarm::posterior_predict
#' function that epidemia uses expects a
#' dataframe in the same format as the one
#' used to fit, with both covariates and
#' (unused) observations
#'
#' @param data data frame used for model fitting
#' @param start_date first date to forecast
#' @param end_date last date to forecast
#' @return tibble of synthetic data
#' @export
synthetic_forecast_data <- function(data,
                                    start_date,
                                    end_date) {
  data <- data |> dplyr::ungroup()
  first_forecast_date <- lubridate::ymd(start_date)
  last_forecast_date <- lubridate::ymd(end_date)
  last_data_date <- as.Date(max(data$date))
  last_data_week <- max(data$week)

  if (last_forecast_date > last_data_date) {
    new_dates <- seq(
      last_data_date + 1,
      last_forecast_date,
      by = "day"
    )

    ## index synthetic weeks to weeks as
    ## in the dataset
    n_days_final_week <- data |>
      dplyr::filter(week == !!last_data_week) |>
      dplyr::pull() |>
      length()

    if (n_days_final_week > 7) {
      cli::cli_abort(paste0(
        "Final week in dataset to ",
        "augment contains more than ",
        "seven entries; check the dataset"
      ))
    } else if (n_days_final_week < 1) {
      cli::cli_abort(paste0(
        "Final week in dataset to ",
        "augment contains no entries; ",
        "check the dataset"
      ))
    }

    last_week_day_count <- (n_days_final_week - 1L)

    new_weeks <- (
      last_data_week + floor(as.numeric(
        new_dates - last_data_date + last_week_day_count,
        "weeks"
      ))
    )

    holidays <- as.Date(c(
      "2023-11-23",
      "2023-12-25",
      "2023-12-31",
      "2024-01-01"
    ))

    new_rows <- tibble::tibble(
      date = !!new_dates,
      week = !!new_weeks,
      location = !!data$location[1],
      hosp = 9999,
      population = !!data$population[1]
    ) |>
      add_date_properties(recency_effect_length = 0)

    result <- dplyr::bind_rows(data, new_rows) |>
      dplyr::distinct(date, .keep_all = TRUE)
  } else {
    result <- data
  }

  result <- result |>
    dplyr::mutate(
      nonobservation_period = 0L,
      hosp = ifelse(date >= first_forecast_date,
        hosp,
        NA ## only forecast needed days
      )
    ) |>
    dplyr::select(-group)
  return(result)
}

#' forecast
#'
#' Given an epidemia model fit (`epimodel` object),
#' the data used to fit it, and the final day of the
#' forecast period, return a set of forecasts (
#' as well as posterior predictive retrocasts
#' for the fitting period, for the given signals.
#'
#' @param fit epidemia model fit as an `epimodel` object.
#' @param start_date first day of the forecast period
#' @param end_date last day of the forecast period
#' @param signals vector of signals to forecast. Default `c('hosp')`.
#' @param seed seed for the posterior predictive pseudorandom
#' number generator, passed to [epidemia::posterior_predict()].
#' Default `NULL`.
#' @return posterior predictive output, in the list format
#' generated by `rstanarm::posterior_predict`
#' @export
forecast <- function(fit,
                     start_date,
                     end_date,
                     signals = c("hosp"),
                     seed = NULL) {
  old_data <- fit$data
  forecast_synth_data <- synthetic_forecast_data(
    old_data,
    start_date,
    end_date
  )

  return(epidemia::posterior_predict(fit,
    newdata = forecast_synth_data,
    types = signals,
    seed = seed
  ))
}


#' pivot_forecast_to_long
#'
#' Take in the output of `rstanarm::posterior_predict`
#' and return a tidy tibble of long-form posterior
#' predictive draws.
#'
#' @param forecast output of `rstanarm::posterior_predict`
#' to pivot.
#' @param signal_name name of the signal being forecast.
#' Default 'hosp'.
#' @param time_name name for the column that will contain
#' the values of the `forecast$time` vector. Default
#' "date".
#' @return the forecast pivoted to tidy long format
#' @export
pivot_forecast_to_long <- function(forecast,
                                   signal_name = "hosp",
                                   time_name = "date") {
  draws_wide <- tibble::tibble(as.data.frame(forecast$draws))
  names(draws_wide) <- forecast$time
  draws_long <- draws_wide |>
    dplyr::mutate(
      .draw = dplyr::row_number()
    ) |>
    tidyr::pivot_longer(
      cols = -.draw,
      names_to = time_name,
      values_to = signal_name
    )
  return(draws_long)
}

#' Produce a daily forecast for a given state
#'
#' @param state_abbr State to forecast, as
#' two-letter USPS abbreviation
#' @param results_list A list of `epimodel`` objects,
#' each corresponding to a single state fit.
#' @param start_date First date to forecast
#' @param end_date Last date to forecast
#' @param fitting_data data used to fit the model
#' @param seed seed for the pseudorandom number
#' generator for posterior prediction (passed to
#' [epidemia::posterior_predict()]. Default `NULL`.
#' @param verbose Boolean. Give verbose output
#' to the terminal? Default `FALSE`.
#' @return state forecast, as a .draw-indexed, tidy
#' [tibble::tibble] object.
#' @export
daily_state_forecast <- function(state_abbr,
                                 results_list,
                                 start_date,
                                 end_date,
                                 fitting_data,
                                 seed = NULL,
                                 verbose = FALSE) {
  state_result <- results_list[[state_abbr]]
  state_data <- fitting_data |>
    dplyr::filter(location == !!state_abbr)


  if (!all(state_result$data$location == state_abbr)) {
    cli::cli_abort(paste0(
      "Could not find a result in the provided ",
      "`results_list` whose fitting data matched ",
      "the provided state abbreviation ",
      "{state_abbr}. Check that the abbreviation ",
      "is correct and that the indices of ",
      "`results_list` and the values of ",
      "`data$location` for the entries of that ",
      "list are two-letter USPS state abbreviations."
    ))
  }

  if (verbose) {
    cli::cli_inform("Forecasting for {state_abbr}\n")
  }

  state_forecast <- forecast(
    state_result,
    start_date,
    end_date,
    seed = seed
  ) |>
    pivot_forecast_to_long() |>
    dplyr::filter(date >= as.Date(!!start_date))

  return(state_forecast)
}


#' Forecast hospitalizations and
#' output them in FluSight format
#'
#' @param data all data (unfilted by location)
#' data used to fit the models, as a `tibble`
#' @param results epidemia results, saved
#' as a list of `epimodel` objects
#' @param output_path Path to save the FluSight formatted
#' .csv file.
#' @param reference_date reference date for the forecast.
#' Should typically be the Saturday that concludes an
#' epiweek.
#' @param horizons vector of forecast horizons to compute,
#' in weeks ahead of the reference_date.
#' Default `-1:3` (FluSight 2023/24 requested horizons)
#' @param seed seed for the posterior predictive
#' pseudorandom random number generator, passed
#' to [epidemia::posterior_predict()]. Default `NULL`.
#' @return The formatted output that has been
#' saved to disk, as a [tibble::tibble()], on
#' success.
#' @export
forecast_and_output_flusight <- function(
    data,
    results,
    output_path,
    reference_date,
    horizons = -1:3,
    seed = NULL) {
  set.seed(seed)

  nation_state_crosswalk <- forecasttools::flusight_location_table


  cli::cli_inform("Producing daily state forecasts...")

  start_date <- (
    lubridate::date(reference_date) +
      lubridate::weeks(min(horizons)) -
      lubridate::weeks(1) -
      lubridate::days(1)
  )

  end_date <- (
    lubridate::date(reference_date) +
      lubridate::weeks(max(horizons)) +
      lubridate::days(1)
  )

  state_vec <- names(results)
  names(state_vec) <- state_vec

  state_daily_forecast_list <- lapply(
    state_vec,
    daily_state_forecast,
    results_list = results,
    start_date = start_date,
    end_date = end_date,
    fitting_data = data,
    verbose = TRUE,
    seed = 62352
  )

  cli::cli_inform("Summarizing daily forecasts to epiweekly...")

  state_weekly_forecasts <- future.apply::future_lapply(
    state_daily_forecast_list,
    forecasttools::daily_to_epiweekly,
    value_col = "hosp",
    id_cols = c(".draw"),
    weekly_value_name = "weekly_hosp"
  )

  cli::cli_inform("Formatting output for FluSight...")

  state_flusight_tables <- list()
  full_table <- tibble::tibble()

  for (state in names(state_weekly_forecasts)) {
    state_flusight_table <- forecasttools::trajectories_to_quantiles(
      state_weekly_forecasts[[state]],
      timepoint_cols = c("epiweek", "epiyear"),
      value_col = "weekly_hosp"
    ) |>
      dplyr::mutate(
        location = forecasttools::loc_abbr_to_flusight_code(state)
      ) |>
      forecasttools:::get_flusight_table(
        reference_date,
        horizons = horizons
      )

    full_table <- dplyr::bind_rows(
      full_table,
      state_flusight_table
    )
  }

  full_table <- full_table |>
    dplyr::arrange(
      location,
      reference_date,
      horizon,
      output_type,
      output_type_id
    )
  readr::write_csv(
    full_table,
    output_path
  )
  return(full_table)
}
