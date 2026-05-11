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


def generate_month_year(df, date_var):
    """Produce month and year variables"""
    df[date_var] = pd.to_datetime(df[date_var])
    df["month"] = df[date_var].dt.month
    df["year"] = df[date_var].dt.year.astype(int)
    return df


##############################
# 311 Complaints
##############################
def load_311(tract_geo, load_impacts=True):
    """Load 311 data on hydrant complaints"""
    print("------------------------")

    if load_impacts:
        print("Loading 311 hydrant data via API")
        df_311 = pd.read_csv(
            "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=100000000&$where=created_date>%272021-01-01%27%20and%20created_date<%272025-12-31%27%20and%20contains(descriptor,%27Hydrant%27)"
        )
        assert df_311.shape[0] < 100000000
        print(f"Initial size of 311 data: {df_311.shape}")

        # drop duplicates via resolution description
        df_311["resolution_description"] = df_311["resolution_description"].fillna("")
        df_311 = df_311[
            ~df_311["resolution_description"].str.contains("duplicat", case=False)
        ]
        print(f"Data after dropping duplicates: {df_311.shape}")

        print(
            f"Number of unique resolution descriptions: {df_311['resolution_description'].unique().shape[0]}"
        )
        print(f"311 data shape after dropping duplicates: {df_311.shape}")

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
                ]
            )
        ]

        gdf_311_tracts = filter_data(gdf_311_tracts, "created_date")

        # count complaints by week and geoid
        gdf_311_summ = gdf_311_tracts[["geoid"]].value_counts().reset_index()
        gdf_311_summ.to_parquet("./_data/311_data.parquet")
    else:
        print("Loading 311 hydrant data from cache")
        gdf_311_summ = pd.read_parquet("./_data/311_data.parquet")
    return gdf_311_summ


def create_date_range(df, date_var, date_freq):
    """Create list of unique date range based on min and max in data"""
    dates = pd.date_range(
        start=df[date_var].min(), end=df[date_var].max(), freq=date_freq
    ).tolist()
    print(f"\nStart date: {df[date_var].min()}")
    print(f"End date: {df[date_var].max()}")
    print(f"Producing list of dates with {date_freq} frequency")
    print(f"Number of unique dates in grid: {len(dates)}")
    return dates


def filter_data(df, date_var):
    """Filter data for correct date range 2021 - 2025, May through September"""
    # filter data after 2021
    print(
        f"Filtering data for 2021 - 2025, May through September for date variable: {date_var}"
    )
    df = generate_month_year(df, date_var)
    df = df[
        (df["year"] >= 2021)
        & (df["year"] <= 2025)
        & (df["month"].isin([5, 6, 7, 8, 9]))
    ]
    print(f"\nMinimum date (after filtering): {df[date_var].min()}")
    print(f"Maximum date (after filtering): {df[date_var].max()}")
    print(f"Unique number of dates: {df[date_var].dt.date.unique().shape[0]}")
    return df


def rank_311(df, dec_gdf, rank_method, date_var="week", date_freq="W-MON"):
    """Create number of hydrant complaints per 1000 people"""

    # merge to grid, will fill in missing values
    gdf_311_summ = dec_gdf[["geoid", "totalpop_dec"]].merge(
        df, on=["geoid"], how="left"
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
    assert np.isfinite(gdf_311_summ["count_pp_hydrant"]).all()  # check finite

    gdf_311_summ["count_pp_hydrant_rank"], gdf_311_summ["count_pp_hydrant_q5"] = (
        custom_qcut_function(gdf_311_summ["count_pp_hydrant"], method=rank_method)
    )

    # check there are no duplicates
    assert gdf_311_summ.shape[0] == gdf_311_summ["geoid"].unique().shape[0]
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

        # check there are no duplicates
        assert (
            df_ems.shape[0]
            == df_ems[
                [
                    "cad_incident_id",
                    "incident_datetime",
                    "zipcode",
                    "first_activation_datetime",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

        # filter for correct time range (2021 - 2025, May through Sept.)
        df_ems = filter_data(df_ems, "first_activation_datetime")

        df_ems_summ = (
            df_ems[["zipcode"]]
            .value_counts()
            .reset_index()
            .rename(columns={"zipcode": "zcta"})
        )

        # checking for missing values
        print(
            f"Percent missing zipcode: {round(100*df_ems_summ['zcta'].isna().mean(), 3)}"
        )
        df_ems_summ["zcta"] = df_ems_summ["zcta"].astype(str).str[:5]
        df_ems_summ.to_parquet("./_data/ems_data.parquet")
    else:
        df_ems_summ = pd.read_parquet("./_data/ems_data.parquet")
    return df_ems_summ


def rank_ems(df, zcta_geo, rank_method):
    """Compute count of heat-related EMS incidents for each week 2021 - 2025, May - September"""

    # merge to full list of zctas, add in zeros
    df_ems_summ = zcta_geo.merge(df, on=["zcta"], how="left")
    # fill in missing count values
    df_ems_summ["count"] = df_ems_summ["count"].fillna(0)

    # rank zctas by ems count
    df_ems_summ["ems_count_rank"], df_ems_summ["ems_count_q5"] = custom_qcut_function(
        df_ems_summ["count"], method=rank_method
    )

    # check there are no duplicates
    assert df_ems_summ.shape[0] == df_ems_summ["zcta"].unique().shape[0]
    return df_ems_summ


##############################
# DPS PROCESSING
##############################


def clean_dps(dps_geo, xwalk):
    """Load and clean DPS data for 2021 - 2025"""
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
        # filter for 5 boroughs via county
        df_dps["COUNTY"].isin(["Queens", "Bronx", "Kings", "Richmond", "New York"])
    ]

    # check none are missing
    assert df_dps["DPS_ID"].notna().all()

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

    # pivot data wide, rename cols
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

    # if both the radial and non-radial outages are the same, only count the non-radial
    print(
        f"\n% perfect duplicates between radial and network outages: {(100*((df_dps_pivot["CUSTOMERS_OUT_0"] ==  df_dps_pivot["CUSTOMERS_OUT_2"]) & (df_dps_pivot["CUSTOMERS_OUT_0"] > 10)).mean()).round(3)}"
    )
    df_dps_pivot["CUSTOMERS_OUT_2"] = np.where(
        (df_dps_pivot["CUSTOMERS_OUT_0"] == df_dps_pivot["CUSTOMERS_OUT_2"])
        & (df_dps_pivot["CUSTOMERS_OUT_0"] > 10),
        0,
        df_dps_pivot["CUSTOMERS_OUT_2"],
    )

    # produce total customers out column
    df_dps_pivot["CUSTOMERS_OUT"] = df_dps_pivot[
        ["CUSTOMERS_OUT_0", "CUSTOMERS_OUT_2"]
    ].sum(axis=1)

    # produce total customers column
    df_dps_pivot["TOTAL_CUSTOMERS"] = df_dps_pivot[
        ["TOTAL_CUSTOMERS_0_MOD", "TOTAL_CUSTOMERS_2_MOD"]
    ].sum(axis=1)

    # calculate customers out rate
    df_dps_pivot["CUSTOMERS_OUT_RATE"] = (
        df_dps_pivot["CUSTOMERS_OUT"] / df_dps_pivot["TOTAL_CUSTOMERS"]
    )

    # create date column
    df_dps_pivot["date"] = pd.to_datetime(
        df_dps_pivot["SUBMIT_DATETIME"]
    ).dt.normalize()

    # create list of dates
    dates = create_date_range(df_dps_pivot, "date", "D")

    # compute the daily max value
    df_dps_summ = df_dps_pivot.groupby(["date", "PRIME_DPS_"], as_index=False)[
        "CUSTOMERS_OUT_RATE"
    ].max()

    # df_dps_summ["date"] = pd.to_datetime(df_dps_summ["date"])

    # create grid and then left join (will fill in 0s)
    df_dps_summ = create_grid(
        dps_geo["PRIME_DPS_"].unique(), dates, "PRIME_DPS_", "date"
    ).merge(df_dps_summ, on=["PRIME_DPS_", "date"], how="left")

    # fill in 0s
    df_dps_summ["CUSTOMERS_OUT_RATE"] = df_dps_summ["CUSTOMERS_OUT_RATE"].fillna(0)

    # filter for correct time range (2021 - 2025, May through Sept.)
    df_dps_summ = filter_data(df_dps_summ, "date")

    # check all rate values are >= 0 and <= 1 (no inf values)
    assert (
        (df_dps_summ["CUSTOMERS_OUT_RATE"] <= 1)
        & (df_dps_summ["CUSTOMERS_OUT_RATE"] >= 0)
    ).all()
    assert df_dps_summ["CUSTOMERS_OUT_RATE"].notna().all()

    # take the mean across the study period
    df_dps_locality_summ = df_dps_summ.groupby("PRIME_DPS_", as_index=False)[
        "CUSTOMERS_OUT_RATE"
    ].mean()

    # write data to parquet file
    df_dps_locality_summ.to_parquet("./_data/dps_summary.parquet")
    return df_dps_locality_summ


def create_dps_rankings(xwalk, rank_method):
    """Create rankings of DPS outage data from locality level summary file"""
    # load data from file
    df_dps = pd.read_parquet("./_data/dps_summary.parquet")

    # produce ranking values
    df_dps["CUSTOMERS_OUT_RATE_rank"], df_dps["CUSTOMERS_OUT_RATE_q5"] = (
        custom_qcut_function(df_dps["CUSTOMERS_OUT_RATE"], method=rank_method)
    )

    print(f"Dataset size prior to xwalk with census tracts: {df_dps.shape[0]}")
    # merge onto crosswalk data to be at tract level
    df_dps_tract_summ = df_dps.merge(xwalk[["PRIME_DPS_", "geoid"]], on="PRIME_DPS_")
    print(
        f"Dataset size after merging with census tracts: {df_dps_tract_summ.shape[0]}"
    )

    # check there are no duplicates
    assert (
        df_dps_tract_summ.shape[0]
        == df_dps_tract_summ[["PRIME_DPS_", "geoid"]].drop_duplicates().shape[0]
    )
    return df_dps_tract_summ
