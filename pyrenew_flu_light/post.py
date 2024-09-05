"""
Save information regarding model performance
and nature of model input and output.
"""


def generate_flusight_formatted_output_forecasttools(input_csv):
    pass


def read_to_df_flusight_formatted_output(input_csv_path: str):
    pass


def compare_flusight_formatted_forecast_kstest():
    pass


# def convert_quantiles_to_draws(
#     input_csv_path: str, states: list[str], output_csv_path: str
# ):
#     df = pl.read_csv(input_csv_path, infer_schema_length=55000)
#     df_filtered = df.filter(pl.col("location").is_in(states))
#     num_draws = 2000
#     result_rows = []
#     for _, group in df_filtered.group_by(
#         ["reference_date", "target", "target_end_date", "location"]
#     ):
#         values = group["value"].to_numpy()
#         quantiles = group["output_type_id"].to_numpy()
#         # assumption normal = placeholder
#         samples = norm.ppf(quantiles, loc=values.mean(), scale=values.std())
#         for draw_index in range(num_draws):
#             for i, sample in enumerate(samples):
#                 result_rows.append([draw_index, i, sample])
#     df_result = pl.DataFrame(result_rows, schema=["draw", "index", "value"])
#     df_result.write_csv(output_csv_path)


# convert_quantiles_to_draws(
#     input_csv_path="../model_comparison/data/2024-03-30/2024-03-30-cfarenewal-cfaepimlight.csv",
#     states=["02"],
#     output_csv_path="test.csv",
# )
