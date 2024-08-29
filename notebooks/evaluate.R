# Start of simple eval. script.

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_local("../../cfa-forecasttools")

library(cfaforecasttools)
library(tibble)
library(dplyr)
library(ggplot2)


forecast_data <- read.csv("samples.csv")
