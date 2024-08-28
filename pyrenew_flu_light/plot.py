"""
Plotting utilities for using pyrenew-flu-light.
"""

import arviz as az
import matplotlib.pyplot as plt


def plot_lm_arviz_fit(idata):  # numpydoc ignore=GL08
    fig, ax = plt.subplots()
    az.plot_lm(
        "negbinom_rv",
        idata=idata,
        kind_pp="hdi",
        y_kwargs={"color": "black"},
        y_hat_fill_kwargs={"color": "C0"},
        axes=ax,
    )
    ax.set_title("Posterior Predictive Plot")
    ax.set_ylabel("Hospital Admissions")
    ax.set_xlabel("Days")
    plt.show()


def compute_eti(dataset, eti_prob):  # numpydoc ignore=GL08
    eti_bdry = dataset.quantile(
        ((1 - eti_prob) / 2, 1 / 2 + eti_prob / 2), dim=("chain", "draw")
    )
    return eti_bdry.values.T


def plot_hdi_arviz_for(idata, forecast_days):  # numpydoc ignore=GL08
    x_data = idata.posterior_predictive["negbinom_rv_dim_0"] + forecast_days
    y_data = idata.posterior_predictive["negbinom_rv"]
    fig, axes = plt.subplots(figsize=(6, 5))
    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.9),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.3},
        ax=axes,
    )

    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.5),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.6},
        ax=axes,
    )
    median_ts = y_data.median(dim=["chain", "draw"])
    plt.plot(
        x_data,
        median_ts,
        color="C0",
        label="Median",
    )
    plt.scatter(
        idata.observed_data["negbinom_rv_dim_0"] + forecast_days,
        idata.observed_data["negbinom_rv"],
        color="black",
    )
    axes.legend()
    axes.set_title("Posterior Predictive Admissions, w/ Forecast", fontsize=10)
    axes.set_xlabel("Time", fontsize=10)
    axes.set_ylabel("Hospital Admissions", fontsize=10)
    plt.show()
