import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scripts.utils import *

# Define for plotting
colorblind_cmap = sns.color_palette("colorblind", 3).as_hex()


def produce_hvi_alternatives(df, additive_factors, subtracted_factors):
    assert min([1 if "_z" in x else 0 for x in additive_factors]) == 1
    assert min([1 if "_z" in x else 0 for x in subtracted_factors]) == 1
    return df[additive_factors].sum(axis=1) - df[subtracted_factors].sum(axis=1)


def rank_all_specifications(df, nta_geo, alt_specifications, rank_method):
    """Rank all specifications"""
    print("------------------------")
    print("Ranking all specifications")
    # producing rankings and quintiles for all specifications
    for var in alt_specifications:
        print(f"Producing alt specification for {var}")
        df[var + "_rank"], df[var + "_q5"] = custom_qcut_function(
            df[var], method=rank_method
        )
    # check differences
    print()
    print("Comparing differences between original and replicated HVI score")
    print(df[["HVI_RANK", "HVI_raw_q5"]].value_counts())
    print(
        f"Accuracy of replicated HVI (5-pt scale): {100*(df['HVI_RANK'] == df['HVI_raw_q5']).mean().round(2)}%"
    )

    # merge to nta data
    gdf = nta_geo.merge(
        df,
        on="nta2020",
    )
    return gdf


def produce_all_specifications(df, health_zscore_cols):
    """Produce all specifications via formula"""

    print("------------------------")
    print("Producing all specifications")
    print("Reproducing original")
    df["HVI_raw"] = produce_hvi_alternatives(
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
    print("Producing comorbidities (with average)")
    df["HVI_health_alt"] = produce_hvi_alternatives(
        df,
        ["SURFACE_TEMP_z", "PCT_BLACK_POP_z", "avg_cdc_health_vars_z"],
        ["GREENSPACE_z", "PCT_HOUSEHOLDS_AC_z", "MEDIAN_INCOME_z"],
    )
    print("Producing all (with averaged comorbidities)")
    df["HVI_all_alt"] = produce_hvi_alternatives(
        df,
        [
            "SURFACE_TEMP_z",
            "PCT_BLACK_POP_z",
            "pct_inpoverty_z",
            "pct_over65_z",
            "avg_cdc_health_vars_z",
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
    for var in vars:
        df[var + "_increase"] = (
            df["HVI_RANK"].astype(int) < df[var + "_q5"].astype(int)
        ) & (df[var + "_q5"].isin([4, 5]).astype(int))
    return df


def produce_risk_increase_map(gdf, vars, nyc_boros, titles):
    """Produce maps corresponding to increases in risk (risk scores of 4 or 5)"""
    vars = [x for x in vars if x != "HVI_raw"]
    for i, var in enumerate(vars):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        gdf.plot(
            column=var + "_q5",
            ax=axes[0],
            cmap="rocket_r",
            legend=True,
            edgecolor="none",
        )
        gdf[gdf[var + "_increase"] == True].to_crs(2263).plot(
            facecolor=colorblind_cmap[1], ax=axes[1], legend=True, edgecolor="none"
        )
        axes[0].set_axis_off()
        axes[1].set_axis_off()

        axes[0].set_title(titles[i])
        axes[1].set_title("NTAs with increased risk of 4 or 5")

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
    """Produce dataframe that's pivoted long as prep for plotting"""
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

    # add color label
    df_melt_q5["color"] = np.select(
        [
            df_melt_q5[f"{orig_var}_q5"] < df_melt_q5["q5"],
            df_melt_q5[f"{orig_var}_q5"] > df_melt_q5["q5"],
        ],
        [colorblind_cmap[2], colorblind_cmap[1]],
        "gray",
    )

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
            "color",
            "label",
        ]
    ]
    print(f"Data size: {df.shape}")

    check_unique_id(df_melt_q5, [id_var, "variable"])
    return df_melt_q5


def min_max_summary(df, id_vars, value_vars):
    """Produce summary of max/min values"""
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
    """Plots NRI data onto NYC maps"""
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
            palette=[colorblind_cmap[2], "#808080", colorblind_cmap[1]],
            style=df_temp["label"],
            markers=True,
            legend=False,
            s=40,
            alpha=0.6,
            linewidth=0,
            ax=ax,
        )
