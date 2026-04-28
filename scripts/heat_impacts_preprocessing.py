import pandas as pd
import geopandas as gpd
import numpy as np
import os

from scripts.utils import *


def create_grid(tract_ids, dates, spatial_id_name, date_var_name="week"):
    """Create evenly spaced grid of dates and tract IDs"""
    grid_data = []
    for tract_id in tract_ids:
        for date in dates:
            grid_data.append({spatial_id_name: tract_id, date_var_name: date})

    grid_df = pd.DataFrame(grid_data)
    return grid_df


def generate_weekly_date(df, date_var):
    """Produce week, date, month, and year variables"""
    df[date_var] = pd.to_datetime(df[date_var])
    df["date"] = df[date_var].dt.date
    df["week"] = df[date_var].dt.isocalendar().week
    df["month"] = df[date_var].dt.month
    df["year"] = df[date_var].dt.year.astype(int)
    df["week"] = pd.to_datetime(
        df["year"].astype(str) + df["week"].astype(str) + "1",
        format="%Y%W%w",
    )
    return df


##############################
# 311 Complaints
##############################
def load_311(tract_geo, load_impacts=True):
    """Load 311 data on hydrant complaints"""
    print("------------------------")
    print("Loading 311 hydrant data via API")

    if load_impacts:
        df_311 = pd.read_csv(
            "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=100000000&$where=created_date>%272021-01-01%27%20and%20created_date<%272025-12-31%27%20and%20contains(descriptor,%27Hydrant%27)"
        )
        assert df_311.shape[0] < 100000000
        print(f"Initial size of 311 data: {df_311.shape}")
        # drop duplicates via resolution description
        df_311["resolution_description"] = df_311["resolution_description"].fillna("")
        df_311 = df_311[~df_311["resolution_description"].str.contains("duplicate")]
        print(f"Data after dropping duplicates: {df_311.shape}")

        # print out resolution descriptions
        df_311["resolution_description"].drop_duplicates().to_csv(
            "resolution descriptions.csv"
        )
        print(
            f"Number of unique resolution descriptions: {df_311['resolution_description'].unique().shape[0]}"
        )
        print(f"311 data shape after dropping duplicates: {df_311.shape}")
        df_311 = generate_weekly_date(df_311, "created_date")

        # create geodataframe
        gdf_311 = gpd.GeoDataFrame(
            df_311,
            geometry=gpd.points_from_xy(
                df_311["longitude"], df_311["latitude"], crs=4326
            ),
        ).to_crs(2263)
        # spatial join
        gdf_311_tracts = gdf_311.sjoin(tract_geo[["geoid", "geometry"]])

        # filter for only 5 categories
        gdf_311_tracts = gdf_311_tracts[
            gdf_311_tracts["descriptor"].isin(
                [
                    "Hydrant Running Full (WA4)",
                    "Hydrant Running (WC3)",
                    "Illegal Use Of A Hydrant (CIN)",
                    "Request To Open A Hydrant (WC4)",
                    "Remove Hydrant Locking Device (WC6)",
                ]
            )
        ]

        # filter data after 2021
        gdf_311_tracts = gdf_311_tracts[
            (gdf_311_tracts["year"] >= 2021)
            & (gdf_311_tracts["year"] <= 2025)
            & (gdf_311_tracts["month"].isin([5, 6, 7, 8, 9]))
        ]

        # count complaints by week and geoid
        gdf_311_summ = gdf_311_tracts[["geoid", "week"]].value_counts().reset_index()
        gdf_311_summ.to_parquet("./_data/311_data.parquet")
    else:
        gdf_311_summ = pd.read_parquet("./_data/311_data.parquet")
    return gdf_311_summ


def create_311_grid(df, tract_geo, dec_gdf):
    """Create number of hydrant complaints per 1000 people"""
    weekly_dates = pd.date_range(
        start=df["week"].min(), end=df["week"].max(), freq="W-MON"
    ).tolist()

    # merge to grid, will fill in missing values
    gdf_311_summ = (
        create_grid(tract_geo["geoid"], weekly_dates, "geoid")
        .merge(df, on=["geoid", "week"], how="left")
        # add in total population info
        .merge(dec_gdf[["geoid", "totalpop_dec"]], on="geoid", how="left")
    )
    # fill in missing values
    gdf_311_summ["count"] = gdf_311_summ["count"].fillna(0)
    # remove tracts with 0 population
    gdf_311_summ = gdf_311_summ[gdf_311_summ["totalpop_dec"] > 0]

    # produce rate per 1000 people
    gdf_311_summ["count_pp_hydrant"] = (
        gdf_311_summ["count"] * 1000 / gdf_311_summ["totalpop_dec"]
    )

    assert gdf_311_summ["count_pp_hydrant"].notna().all()
    assert np.isfinite(gdf_311_summ["count_pp_hydrant"]).all() # check finite

    # take the average across all weeks in the data
    gdf_311_summ = gdf_311_summ.groupby(["geoid"], as_index=False)[
        "count_pp_hydrant"
    ].mean()

    gdf_311_summ["count_pp_hydrant_rank"], gdf_311_summ["count_pp_hydrant_q5"] = (
        custom_qcut_function(gdf_311_summ["count_pp_hydrant"])
    )

    # check there are no duplicates
    assert gdf_311_summ.shape[0] == gdf_311_summ['geoid'].unique().shape[0]
    return gdf_311_summ


##############################
# EMS PROCESSING
##############################

def load_ems(load_impacts=True):
    """Load EMS data via Open Data API"""
    print("------------------------")
    print("Loading EMS data via API")
    if load_impacts:
        df_ems = pd.read_csv(
            "https://data.cityofnewyork.us/resource/76xm-jjuj.csv?$limit=100000000&$where=first_activation_datetime>%272021-01-01%27%20and%20(initial_call_type=%27HEAT%27%20or%20final_call_type%20=%20%27HEAT%27)"
        )
        assert df_ems.shape[0] < 100000000
        # keep only final call type equal to heat
        df_ems = df_ems[df_ems["final_call_type"] == "HEAT"]
        df_ems = df_ems[df_ems["first_activation_datetime"].notna()]

        df_ems = generate_weekly_date(df_ems, "first_activation_datetime")
        df_ems_summ = (
            df_ems[["week", "zipcode"]]
            .value_counts()
            .reset_index()
            .rename(columns={"zipcode": "zcta"})
        )
        df_ems_summ["zcta"] = df_ems_summ["zcta"].astype(str).str[:5]
        df_ems_summ.to_parquet("./_data/ems_data.parquet")
    else:
        df_ems_summ = pd.read_parquet("./_data/ems_data.parquet")
    return df_ems_summ


def create_ems_grid(df, zcta_geo):
    """Create grid of heat-related EMS incidents for each week 2021 - 2025, May - September"""
    df["month"] = df["week"].dt.month
    df["year"] = df["week"].dt.year

    # filter to data in the study period
    df = df[
        (df["year"] >= 2021)
        & (df["year"] <= 2025)
        & (df["month"].isin([5, 6, 7, 8, 9]))
    ]
    
    # produce weekly dates (as opposed to daily)
    weekly_dates = pd.date_range(
        start=df["week"].min(), end=df["week"].max(), freq="W-MON"
    ).tolist()

    # merge to grid, will add in 0s
    df_ems_summ = create_grid(zcta_geo["zcta"].unique(), weekly_dates, "zcta").merge(
        df, on=["zcta", "week"], how="left"
    )
    # fill in missing count values
    df_ems_summ["count"] = df_ems_summ["count"].fillna(0)

    # group by zcta
    df_ems_summ = df_ems_summ.groupby(["zcta"], as_index=False)["count"].mean()

    # rank zctas by ems count
    df_ems_summ["ems_count_rank"], df_ems_summ["ems_count_q5"] = custom_qcut_function(
        df_ems_summ["count"]
    )

    # check there are no duplicates
    assert df_ems_summ.shape[0] == df_ems_summ['zcta'].unique().shape[0]
    return df_ems_summ


##############################
# DPS PROCESSING
##############################


def clean_dps(dps_geo, xwalk):
    """Load DPS data"""
    print("------------------------")
    print("Loading DPS Data on Power Outages")
    # load dps data
    df_dps = pd.concat(
        [
            pd.read_csv(f"_data/_foil/EORS_Outage_Data_{year}.csv")
            for year in range(2021, 2026)
        ]
    )
    df_dps = df_dps[
        df_dps["COUNTY"].isin(["Queens", "Bronx", "Kings", "Richmond", "New York"])
    ]
    df_dps["PRIME_DPS_"] = (
        df_dps["DPS_ID"].str.split(".").apply(lambda x: x[0]).str.strip()
    )

    # for DPS service localities that need to merged, summ total customers and total customers out
    # need to pivot wide to ensure that we are filling in missing values
    df_dps["SUBMIT_DATETIME"] = pd.to_datetime(
        df_dps["SUBMIT_DATE"] + " " + df_dps["SUBMIT_TIME"]
    )
    # some places have two DPS localities, need to align customers and outages
    df_dps["DPS_ORDER"] = (
        df_dps["DPS_ID"].str.split(".").apply(lambda x: x[-1]).str.strip()
    )
    print(f"Unique values of DPS_ORDER var: {df_dps['DPS_ORDER'].unique()}")
    df_dps_pivot = df_dps.pivot(
        index=["SUBMIT_DATETIME", "PRIME_DPS_"],
        columns=["DPS_ORDER"],
        values=["CUSTOMERS_OUT", "TOTAL_CUSTOMERS"],
    ).reset_index()

    # remove hierarchical multiindex, rename columns
    df_dps_pivot.columns = [
        "_".join(col).strip() for col in df_dps_pivot.columns.values
    ]

    # pivot data wide
    df_dps_pivot = df_dps_pivot.rename(
        columns={"SUBMIT_DATETIME_": "SUBMIT_DATETIME", "PRIME_DPS__": "PRIME_DPS_"}
    )

    for col in ["TOTAL_CUSTOMERS_0", "TOTAL_CUSTOMERS_2"]:
        # first: fill in total customers chronologically (earlier dates to later)
        df_dps_pivot[col + "_MOD"] = (
            df_dps_pivot.sort_values("SUBMIT_DATETIME", ascending=False)
            .groupby("PRIME_DPS_", as_index=False)[col]
            .transform(lambda x: x.bfill())
        )
        # second: fill in total customers from later dates to earlier
        df_dps_pivot[col + "_MOD"] = (
            df_dps_pivot.sort_values("SUBMIT_DATETIME", ascending=True)
            .groupby("PRIME_DPS_", as_index=False)[col + "_MOD"]
            .transform(lambda x: x.bfill())
        )

    # fill missing values with 0
    df_dps_pivot[["CUSTOMERS_OUT_0", "CUSTOMERS_OUT_2"]] = df_dps_pivot[
        ["CUSTOMERS_OUT_0", "CUSTOMERS_OUT_2"]
    ].fillna(0)

    # produce customers out column
    df_dps_pivot["CUSTOMERS_OUT"] = df_dps_pivot[
        ["CUSTOMERS_OUT_0", "CUSTOMERS_OUT_2"]
    ].sum(axis=1)

    # produce total customers column
    df_dps_pivot['TOTAL_CUSTOMERS'] = df_dps_pivot[
        ["TOTAL_CUSTOMERS_0_MOD", "TOTAL_CUSTOMERS_2_MOD"]
    ].sum(
        axis=1
    )

    df_dps_pivot["CUSTOMERS_OUT_RATE"] = df_dps_pivot[
        "CUSTOMERS_OUT"
    ] / df_dps_pivot[
        "TOTAL_CUSTOMERS"
    ]

    # create week, date, month, year
    df_dps_pivot = generate_weekly_date(df_dps_pivot, "SUBMIT_DATETIME")

    # filter to 2021, may through september
    df_dps_pivot = df_dps_pivot[
        (df_dps_pivot["year"] >= 2021) & (df_dps_pivot["month"].isin([5, 6, 7, 8, 9]))
    ]

    # create list of dates
    dates = pd.date_range(
        start=df_dps_pivot["date"].min(), end=df_dps_pivot["date"].max(), freq="D"
    ).tolist()

    # compute the daily max value
    df_dps_summ = df_dps_pivot.groupby(["date", "PRIME_DPS_"], as_index=False)[
        "CUSTOMERS_OUT_RATE"
    ].max()

    df_dps_summ["date"] = pd.to_datetime(df_dps_summ["date"])

    # create grid and then left join (will fill in 0s)
    df_dps_summ = create_grid(
        df_dps_summ["PRIME_DPS_"], dates, "PRIME_DPS_", "date"
    ).merge(df_dps_summ, on=["PRIME_DPS_", "date"], how="left")

    # fill in 0s
    df_dps_summ["CUSTOMERS_OUT_RATE"] = df_dps_summ["CUSTOMERS_OUT_RATE"].fillna(0)

    # check all rate values are >= 0 and <= 1 (no inf values)
    assert ((df_dps_summ["CUSTOMERS_OUT_RATE"] <= 1) & (
        df_dps_summ["CUSTOMERS_OUT_RATE"] >= 0
    )).all()
    assert df_dps_summ["CUSTOMERS_OUT_RATE"].notna().all()

    # take the mean across the study period
    df_dps_summ = df_dps_summ.groupby("PRIME_DPS_", as_index=False)[
        "CUSTOMERS_OUT_RATE"
    ].mean()

    # produce ranking values
    df_dps_summ["CUSTOMERS_OUT_RATE_rank"], df_dps_summ["CUSTOMERS_OUT_RATE_q5"] = (
        custom_qcut_function(df_dps_summ["CUSTOMERS_OUT_RATE"])
    )

    print(f"Dataset size prior to xwalk with census tracts: {df_dps_summ.shape[0]}")
    # merge onto crosswalk data to be at tract level
    df_dps_tract_summ = df_dps_summ.merge(
        xwalk[["PRIME_DPS_", "geoid"]], on="PRIME_DPS_"
    )
    print(f"Dataset size after merging with census tracts: {df_dps_tract_summ.shape[0]}")

    # check there are no duplicates
    assert df_dps_tract_summ.shape[0] == df_dps_tract_summ[['PRIME_DPS_', 'geoid']].drop_duplicates().shape[0]
    return df_dps_tract_summ
