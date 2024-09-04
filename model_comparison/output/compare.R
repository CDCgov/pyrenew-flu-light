library(forecasttools)
library(tibble)
library(dplyr)
library(readr)
library(lubridate)

# function for taking
run_forecast_with_csvs <- function(
    spread_draws_csv,
    fitting_data_csv,
    output_path,
    reference_date,
    horizons = -1:3,
    seed = NULL
) {
  set.seed(seed)
  # read data from pyrenew-flu-light run
  spread_draws <- read_csv(spread_draws_csv)
  fitting_data <- read_csv(fitting_data_csv)
  # retrieve locations from fitting data
  locations <- unique(fitting_data$location)
  # parse over and collect forecasts and fitting data
  state_daily_forecast_list <- list()
  for (loc in locations) {
    state_fitting_data <- fitting_data %>% filter(location == loc)
    state_forecast <- spread_draws %>% filter(negbinom_rv_dim_0_index == loc)
    # forecasttools convert spread draws tiddy
    state_forecast_long <- state_forecast %>%
      dplyr::mutate(.draw = draw) %>%
      dplyr::select(.draw, date = negbinom_rv_dim_0_index, hosp = negbinom_rv)
    # go to epiweekly from daily
    state_weekly_forecasts <- forecasttools::daily_to_epiweekly(
      tidy_daily_trajectories = state_forecast_long,
      value_col = "hosp",
      date_col = "date",
      id_cols = ".draw"
    )
    state_daily_forecast_list[[loc]] <- state_weekly_forecasts
  }
  cli::cli_inform("Formatting output for FluSight...")
  # flusight formatting
  state_flusight_tables <- list()
  full_table <- tibble::tibble()
  for (state in names(state_daily_forecast_list)) {
    state_flusight_table <- forecasttools::trajectories_to_quantiles(
      state_daily_forecast_list[[state]],
      timepoint_cols = c("epiweek", "epiyear"),
      value_col = "weekly_hosp"
    ) %>%
      dplyr::mutate(
        location = forecasttools::loc_abbr_to_flusight_code(state)
      ) %>%
      forecasttools:::get_flusight_table(
        reference_date,
        horizons = horizons
      )

    full_table <- dplyr::bind_rows(
      full_table,
      state_flusight_table
    )
  }
  full_table <- full_table %>%
    dplyr::arrange(
      location,
      reference_date,
      horizon,
      output_type,
      output_type_id
    )
  # save
  readr::write_csv(
    full_table,
    output_path
  )
  return(full_table)
}

spread_draws_csv <- "AL_2024-03-30_28_NegBinRv.csv"
fitting_data_csv <- "filtered_data_AL.csv"
output_path <- "flusight_output_AL.csv"
reference_date <- "2024-03-30"
horizons <- -1:3

run_forecast_with_csvs(
    spread_draws_csv,
    fitting_data_csv,
    output_path,
    reference_date,
    horizons,
    seed = 62352
)
