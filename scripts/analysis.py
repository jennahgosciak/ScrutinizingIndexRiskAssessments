import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt
from tqdm import tqdm
from scripts.utils import *

# Define for plotting
colorblind_cmap = sns.color_palette("colorblind", 3).as_hex()

##############################
# HVI Replication Functions
##############################


def produce_hvi_alternatives(df, additive_factors, subtracted_factors):
    """Produce simple HVI additive formula (adding and subtracting relevant inputs)"""
    assert min([1 if "_z" in x else 0 for x in additive_factors]) == 1
    assert min([1 if "_z" in x else 0 for x in subtracted_factors]) == 1
    return df[additive_factors].sum(axis=1) - df[subtracted_factors].sum(axis=1)


def rank_all_specifications(df, nta_geo, alt_specifications, rank_method):
    """Rank all specifications"""
    print("------------------------")
    print("Ranking all specifications")
    df = df.copy()

    # producing rankings and quintiles for all specifications
    for var in alt_specifications:
        print(f"Ranking alt specification for {var}")
        df[var + "_rank"], df[var + "_q5"] = custom_qcut_function(
            df[var], method=rank_method
        )
    # check differences
    print()
    print("Comparing differences between original and replicated HVI score")
    print(df[["HVI_RANK", "HVI_repl_q5"]].value_counts())
    print(
        f"Accuracy of replicated HVI (5-pt scale): {100*(df['HVI_RANK'] == df['HVI_repl_q5']).mean().round(2)}%"
    )

    # merge to nta data
    gdf = nta_geo.merge(
        df,
        on="nta2020",
    )
    return gdf


def produce_all_specifications(df, health_zscore_cols):
    """Produce all alternative specifications as enumerated in the main paper"""
    print("------------------------")
    print("Producing all specifications")
    print("Reproducing original")
    df["HVI_repl"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "PCT_BLACK_POP_z"],
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    print("Producing environmental only")
    df["HVI_env"] = produce_hvi_alternatives(df, ["SURFACE_TEMP_z"], ["GREENSPACE_z"])

    print("Producing age and poverty prioritized")
    df["HVI_age"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "pct_inpoverty_z", "pct_over65_z"],
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z"],
    )
    print("Producing health comorbidities added")
    df["HVI_health"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "PCT_BLACK_POP_z"] + health_zscore_cols,
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    print("Producing all (combined)")
    df["HVI_all"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "PCT_BLACK_POP_z", "pct_inpoverty_z", "pct_over65_z"]
        + health_zscore_cols,
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    print("Producing minority status instead of race")
    df["HVI_min"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "nonwhite_nh_dec_pct_z"],
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    print("Producing comorbidities (with max)")
    df["HVI_health_alt"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "PCT_BLACK_POP_z", "max_cdc_health_vars_z"],
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    print("Producing all (with max comorbidities)")
    df["HVI_all_alt"] = produce_hvi_alternatives(
        df,
        [
            "SURFACE_TEMP_z",
            "PCT_BLACK_POP_z",
            "pct_inpoverty_z",
            "pct_over65_z",
            "max_cdc_health_vars_z",
        ],
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    return df


def produce_correlations(df, vars, correlation_method, latex=True):
    """Produce correlation matrix for a set of input variables"""
    print("------------------------")
    print(f"Correlation using {correlation_method}")
    corr_matrix = df[vars].corr(method=correlation_method)
    if latex == True:
        print(corr_matrix.round(3).astype(str).to_latex())
    return corr_matrix


def compute_risk_increase(df, vars):
    """Identify where any of the input variables lead to an increase in HVI risk to HVI=4 or 5"""
    df = df.copy()
    for var in vars:
        df[var + "_increase"] = (df["HVI_repl_q5"].astype(int) < 4) & (
            df[var + "_q5"].isin([4, 5]).astype(int)
        )
    return df


def summarize_agreement(df, vars, latex=False):
    """Produce agreement summaries for different HVI specifications compard to original HVI risk scores"""
    df = df.copy()
    for var in vars:
        df[var + "_match"] = df[var + "_q5"] == df["HVI_repl_q5"]

    agreement_summary = df[[x + "_match" for x in vars]].mean() * 100

    if latex == True:
        print(agreement_summary.round(2).to_latex())
    else:
        print(agreement_summary)
    return agreement_summary


def produce_risk_increase_map(gdf, vars, nyc_boros, titles):
    """Produce maps corresponding to increases in risk (HVI risk = 4 or 5)"""
    gdf = gdf.copy()
    vars = [x for x in vars if x != "HVI_repl"]
    for i, var in enumerate(vars):
        gdf[var + "_q5"] = gdf[var + "_q5"].astype(str).str.replace(".0", "")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        gdf.plot(
            column=var + "_q5",
            ax=axes[0],
            cmap="rocket_r",
            legend=True,
            edgecolor="none",
        )
        gdf[gdf[var + "_increase"] == True].plot(
            facecolor=colorblind_cmap[1], ax=axes[1], legend=True, edgecolor="none"
        )
        axes[0].set_axis_off()
        axes[1].set_axis_off()

        axes[0].set_title(titles[i])
        axes[1].set_title("NTAs with increased risk scores of 4 or 5")

        nyc_boros.plot(ax=axes[0], facecolor="none", edgecolor="gray", lw=0.3)
        nyc_boros.plot(ax=axes[1], facecolor="none", edgecolor="gray", lw=0.3)
        plt.tight_layout()
        plt.savefig(
            f"./_figures/hvi_{var}.pdf",
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )
        plt.show()

        print(gdf.sort_values(var + "_rank", ascending=False)["ntaname"].head(10))


def prep_for_plot(df, vars, orig_var, id_var):
    """Produce dataframe that's pivoted long to prepare for plot"""
    print("------------------------")
    print("Prepping data for plotting")
    print(f"id var: {id_var}")
    print(f"vars to pivot: {', '.join(vars)}")
    rank_vars = [x + "_rank" for x in vars]
    q5_vars = [x + "_q5" for x in vars]
    df_melt = df.copy()

    # pivot rank long
    df_melt = df_melt[rank_vars + [orig_var + "_rank", id_var]].melt(
        id_vars=[id_var, orig_var + "_rank"], value_name="rank"
    )
    df_melt["variable"] = df_melt["variable"].str.replace("_rank", "")

    # pivot q5 long
    df_melt_q5 = df[q5_vars + [orig_var + "_rank", orig_var + "_q5", id_var]].melt(
        id_vars=[id_var, orig_var + "_rank", orig_var + "_q5"], value_name="q5"
    )
    df_melt_q5["variable"] = df_melt_q5["variable"].str.replace("_q5", "")
    df_melt_q5 = df_melt_q5.merge(df_melt, on=[orig_var + "_rank", id_var, "variable"])

    # add data label
    df_melt_q5["label"] = np.select(
        [
            df_melt_q5[f"{orig_var}_q5"] < df_melt_q5["q5"],
            df_melt_q5[f"{orig_var}_q5"] > df_melt_q5["q5"],
        ],
        ["Increased HVI score", "Decreased HVI score"],
        "Unchanged HVI score",
    )

    df_melt_q5 = df_melt_q5[
        [
            id_var,
            "rank",
            "q5",
            "variable",
            f"{orig_var}_rank",
            f"{orig_var}_q5",
            "label",
        ]
    ]
    print(f"Data size: {df.shape}")

    check_unique_id(df_melt_q5, [id_var, "variable"])
    return df_melt_q5


def min_max_summary(df, id_vars, value_vars):
    """Produce summary of max/min values compared to original HVI"""
    df_summary = (
        df[id_vars + value_vars]
        .melt(id_vars=id_vars)
        .groupby(id_vars, as_index=False)["value"]
        .agg(["min", "max", "mean", "std"])
    )
    return df_summary


########################
# NRI Analysis
########################


def plot_nri(df, tract_geo, nyc_boros):
    """Plots NRI data in NYC"""
    df_geo = tract_geo[["geoid", "geometry"]].merge(df, on="geoid")
    df_geo[["HWAV_EALT_q5", "HWAV_EALTxSVIxRESL_q5"]] = (
        df_geo[["HWAV_EALT_q5", "HWAV_EALTxSVIxRESL_q5"]].astype(int).astype(str)
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    df_geo.plot(column="HWAV_EALT_q5", cmap="rocket_r", ax=ax[0], edgecolor="none")
    df_geo.plot(
        column="HWAV_EALTxSVIxRESL_q5",
        cmap="rocket_r",
        ax=ax[1],
        legend=True,
        edgecolor="none",
        legend_kwds={"loc": "upper right", "bbox_to_anchor": (1.2, 1)},
    )
    nyc_boros.plot(ax=ax[0], facecolor="none", edgecolor="gray", lw=0.3)
    nyc_boros.plot(ax=ax[1], facecolor="none", edgecolor="gray", lw=0.3)

    ax[0].set_title("EAL")
    ax[1].set_title("EAL x f(SV / CR)")
    ax[0].axis("off")
    ax[1].axis("off")

    plt.savefig(
        "./_figures/nri_comparison.pdf", bbox_inches="tight", pad_inches=0, dpi=300
    )
    plt.show()


def produce_scatter(df, orig_var, ax):
    """Add scatter plot with formatting"""
    for line in df["variable"].unique():
        df_temp = df[df["variable"] == line]
        print(f"Plotting {line}")
        print(f"df temp size: {df_temp.shape}")

        sns.scatterplot(
            x=df_temp[orig_var + "_rank"],
            y=df_temp["rank"],
            hue=df_temp["label"],
            hue_order=[
                "Decreased HVI score",
                "Unchanged HVI score",
                "Increased HVI score",
            ],
            style_order=[
                "Decreased HVI score",
                "Unchanged HVI score",
                "Increased HVI score",
            ],
            palette=[colorblind_cmap[1], "#808080", colorblind_cmap[2]],
            style=df_temp["label"],
            markers=True,
            legend=False,
            s=50,
            alpha=0.6,
            linewidth=0,
            ax=ax,
        )


def patches(ax, colorblind_cmap):
    decrease_patch = mpl.lines.Line2D(
        [0],
        [0],
        marker="o",
        markerfacecolor=colorblind_cmap[1],
        markersize=10,
        markeredgecolor="none",
        ls="",
        label="Decreased score",
    )
    increase_patch = mpl.lines.Line2D(
        [0],
        [0],
        marker="s",
        markerfacecolor=colorblind_cmap[2],
        markersize=10,
        markeredgecolor="none",
        ls="",
        label="Increased score",
    )
    nochange_patch = mpl.lines.Line2D(
        [0],
        [0],
        marker="X",
        markerfacecolor="gray",
        ls="",
        markersize=10,
        markeredgecolor="none",
        label="Unchanged score",
    )
    ax.legend(
        handles=[increase_patch, nochange_patch, decrease_patch],
        loc="lower right",
        fontsize=10,
    )


def scatter_plot_formatting(fig, ax):
    """Default formatting for scatter plot"""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))

    ax.text(
        19,
        95,
        "1",
        horizontalalignment="right",
        color="black",
        fontweight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", pad=0, alpha=0.6),
    )
    ax.text(
        39,
        95,
        "2",
        horizontalalignment="right",
        color="black",
        fontweight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", pad=0, alpha=0.6),
    )
    ax.text(
        59,
        95,
        "3",
        horizontalalignment="right",
        color="black",
        fontweight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", pad=0, alpha=0.6),
    )
    ax.text(
        79,
        95,
        "4",
        horizontalalignment="right",
        color="black",
        fontweight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", pad=0, alpha=0.6),
    )
    ax.text(
        99,
        95,
        "5",
        horizontalalignment="right",
        color="black",
        fontweight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", pad=0, alpha=0.6),
    )
    fig.supylabel("New Percentile Ranking")
    fig.supxlabel("Original HVI prioritizations (percentile ranking)")


def produce_facet_plot(df_hvi, df_tract_hvi, df_nri, id_vars, filename):
    """Produces main facet plot"""
    print("------------------------")
    print("Producing main facet plot (fig. 2)")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.65), layout="constrained")

    for i, ax in enumerate(axes):
        ax.axvline(20, color="gray", linestyle="dashed")
        ax.axvline(40, color="gray", linestyle="dashed")
        ax.axvline(60, color="gray", linestyle="dashed")
        ax.axvline(80, color="gray", linestyle="dashed")
        ax.axvline(100, color="gray", linestyle="dashed")

        if i == 0:
            produce_scatter(df_hvi, id_vars[i], ax)
            ax.set_title("(a) Alternative model specifications")
        elif i == 1:
            produce_scatter(df_tract_hvi, id_vars[i], ax)
            ax.set_title("(b) HVI prioritization (tract-level)")
        elif i == 2:
            produce_scatter(df_nri, id_vars[2], ax)
            ax.set_title("(c) NRI specifications")
            patches(ax, colorblind_cmap)
        default_plot(ax)
        scatter_plot_formatting(fig, ax)

    plt.savefig(f"./_figures/{filename}", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


##############################################
# Check Sensitivity of Results
# For Selecting Health/Comorbidity Columns
##############################################


def test_sensitivity_health_specification(
    df, health_cols, rank_method, correlation_method
):
    """
    Tests the sensitivity of the choice of comorbidities for the health specification.
    For any combination of health columns, produces rankings and quintiles.
    """
    df = df.copy()
    df["HVI_health_sens"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "PCT_BLACK_POP_z"] + list(health_cols),
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )

    df["HVI_health_sens_rank"], df["HVI_health_sens_q5"] = custom_qcut_function(
        df["HVI_health_sens"], method=rank_method
    )

    rank_corr = (
        df[["HVI_repl_rank", "HVI_health_sens_rank"]]
        .corr(method=correlation_method)
        .iloc[0, 1]  # extract relevant correlation value
    )
    q5_corr = (
        df[["HVI_repl_q5", "HVI_health_sens_q5"]]
        .corr(method=correlation_method)
        .iloc[0, 1]  # extract relevant correlation value
    )
    return rank_corr, q5_corr


def health_specification_correlations(df, health_cols, rank_method, correlation_method):
    """
    Produce all possible ranking- and quintile-based correlations for different comorbidities
    included in the health specification HVI.
    """
    # create all combinations of health cols
    health_cols_combinations = []
    for i in range(1, len(health_cols) + 1):
        health_cols_combinations += itertools.combinations(health_cols, r=i)

    corr_vals = []
    q5_corr_vals = []
    for col in tqdm(health_cols_combinations):
        rank_corr, q5_corr = test_sensitivity_health_specification(
            df, col, rank_method, correlation_method
        )
        corr_vals += [rank_corr]
        q5_corr_vals += [q5_corr]
    return corr_vals, q5_corr_vals, health_cols_combinations


def print_health_cols_corr(corr_vals):
    """Print average, minimum, and maxium values for different rankings"""
    print(f"Average value: {round(np.mean(corr_vals), 4)}")
    print(f"Minimum: {round(np.min(corr_vals), 4)}")
    print(f"Maximum: {round(np.max(corr_vals), 4)}")
    print(
        f"Median and IQR: {round(np.quantile(corr_vals, 0.5), 4)} ({round(np.quantile(corr_vals, 0.25), 4)}-{round(np.quantile(corr_vals, 0.75), 4)})"
    )
