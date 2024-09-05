library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(forecasttools)

read_posterior_samples <- function(file_path) {
  posterior_samples <- read_csv(file_path)
  return(posterior_samples)
}

pivot_forecast_to_long <- function(
  posterior_samples,
  signal_name = "hosp",
  time_name = "date"
) {
  draws_wide <- tibble::tibble(as.data.frame(posterior_samples))
  names(draws_wide) <- posterior_samples$time
  draws_long <- draws_wide %>%
    dplyr::mutate(.draw = dplyr::row_number()) %>%
    tidyr::pivot_longer(cols = -.draw, names_to = time_name, values_to = signal_name)
  return(draws_long)
}
aggregate_to_epiweekly <- function(tidy_daily_trajectories) {
  epiweekly_forecasts <- forecasttools::daily_to_epiweekly(
    tidy_daily_trajectories,
    value_col = "hosp",
    id_cols = c(".draw"),
    weekly_value_name = "weekly_hosp"
  )
  return(epiweekly_forecasts)
}

output_flusight_table <- function(
  weekly_forecasts,
  reference_date,
  horizons,
  output_path
) {
  formatted_output <- forecasttools::trajectories_to_quantiles(
    weekly_forecasts,
    timepoint_cols = c("epiweek", "epiyear"),
    value_col = "weekly_hosp"
  ) %>%
    dplyr::mutate(
      location = forecasttools::loc_abbr_to_flusight_code(state)
    ) %>%
    forecasttools::get_flusight_table(
      reference_date,
      horizons = horizons
    )

  readr::write_csv(formatted_output, output_path)
  return(formatted_output)
}

generate_flusight_output <- function(file_paths, reference_date, horizons, output_path) {
  for (file_path in file_paths) {
    posterior_samples <- read_posterior_samples(file_path)
    daily_forecasts <- pivot_forecast_to_long(posterior_samples)
    weekly_forecasts <- aggregate_to_epiweekly(daily_forecasts)
    formatted_output <- output_flusight_table(weekly_forecasts, reference_date, horizons, output_path)
  }
}


file_paths <- c(
  "posterior_predictive_forecasts_test_NY_2024-01-20.csv")
reference_date <- "2024-01-20"
horizons <- -1:3
output_path <- "flusight_forecast_output_PFL_test_NY_2024-01-20.csv"

generate_flusight_output(
  file_paths,
  reference_date,
  horizons,
  output_path)







# library(forecasttools)
# library(tibble)
# library(dplyr)
# library(readr)
# library(lubridate)

# pyrenew_flusight_forecast_from_csv <- function(
#     spread_draws_csv,
#     fitting_data_csv,
#     output_path,
#     reference_date,
#     horizons = -1:3,
#     seed = NULL
# ) {
#   set.seed(seed)
#   # read data from pyrenew-flu-light run
#   spread_draws <- read_csv(spread_draws_csv)
#   fitting_data <- read_csv(fitting_data_csv)
#   # retrieve locations from fitting data
#   locations <- unique(fitting_data$location)
#   # parse over and collect forecasts and fitting data
#   state_daily_forecast_list <- list()
#   for (loc in locations) {
#     state_fitting_data <- fitting_data %>% filter(location == loc)
#     state_forecast <- spread_draws %>% filter(negbinom_rv_dim_0_index == loc)
#     # forecasttools convert spread draws tiddy
#     state_forecast_long <- state_forecast %>%
#       dplyr::mutate(.draw = draw) %>%
#       dplyr::select(.draw, date = negbinom_rv_dim_0_index, hosp = negbinom_rv)
#     # go to epiweekly from daily
#     state_weekly_forecasts <- forecasttools::daily_to_epiweekly(
#       tidy_daily_trajectories = state_forecast_long,
#       value_col = "hosp",
#       date_col = "date",
#       id_cols = ".draw"
#     )
#     state_daily_forecast_list[[loc]] <- state_weekly_forecasts
#   }
#   cli::cli_inform("Formatting output for FluSight...")
#   # flusight formatting
#   state_flusight_tables <- list()
#   full_table <- tibble::tibble()
#   for (state in names(state_daily_forecast_list)) {
#     state_flusight_table <- forecasttools::trajectories_to_quantiles(
#       state_daily_forecast_list[[state]],
#       timepoint_cols = c("epiweek", "epiyear"),
#       value_col = "weekly_hosp"
#     ) %>%
#       dplyr::mutate(
#         location = forecasttools::loc_abbr_to_flusight_code(state)
#       ) %>%
#       forecasttools:::get_flusight_table(
#         reference_date,
#         horizons = horizons
#       )

#     full_table <- dplyr::bind_rows(
#       full_table,
#       state_flusight_table
#     )
#   }
#   full_table <- full_table %>%
#     dplyr::arrange(
#       location,
#       reference_date,
#       horizon,
#       output_type,
#       output_type_id
#     )
#   # save
#   readr::write_csv(
#     full_table,
#     output_path
#   )
#   return(full_table)
# }

# spread_draws_csv <- "AL_2024-03-30_28_NegBinRv.csv"
# fitting_data_csv <- "filtered_data_AL.csv"
# output_path <- "flusight_output_AL.csv"
# reference_date <- "2024-03-30"
# horizons <- -1:3

# run_forecast_with_csvs(
#     spread_draws_csv,
#     fitting_data_csv,
#     output_path,
#     reference_date,
#     horizons,
#     seed = 62352
# )
