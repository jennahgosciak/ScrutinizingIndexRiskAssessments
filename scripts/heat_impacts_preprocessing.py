import pandas as pd
import geopandas as gpd
import numpy as np

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
def load_311():
    """Load 311 data on hydrant complaints"""
    print("------------------------")
    print("Loading 311 hydrant data via API")

    if not os.path.isfile("./_data/311_data.parquet"):
        df_311 = pd.read_csv(
            "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$limit=100000000&$where=created_date>%272021-01-01%27%20and%20created_date<%272025-12-31%27%20and%20contains(descriptor,%27Hydrant%27)"
        )
        print(f"311 data shape: {df_311.shape}")
        # drop duplicates via resolution description
        df_311["resolution_description"] = df_311["resolution_description"].fillna("")
        df_311 = df_311[~df_311["resolution_description"].str.contains("duplicate")]
        
        # print out resolution descriptions
        df_311["resolution_description"].drop_duplicates().to_csv("resolution descriptions.csv")
        print(f"Number of unique resolution descriptions: {df_311['resolution_description'].unique().shape[0]}")
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
            (gdf_311_tracts["year"] >= 2021) & (gdf_311_tracts["year"] <= 2025)
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

    gdf_311_summ = (
        create_grid(tract_geo["geoid"], weekly_dates, "geoid")
        .merge(df, on=["geoid", "week"], how="left")
        .merge(dec_gdf[["geoid", "totalpop_dec"]], on="geoid", how="left")
    )
    gdf_311_summ["count"] = gdf_311_summ["count"].fillna(0)
    gdf_311_summ = gdf_311_summ[gdf_311_summ["totalpop_dec"] > 0]

    gdf_311_summ["count_pp_hydrant"] = (
        gdf_311_summ["count"] * 1000 / gdf_311_summ["totalpop_dec"]
    )
    gdf_311_summ = gdf_311_summ.groupby(["geoid"], as_index=False)[
        "count_pp_hydrant"
    ].mean()
    gdf_311_summ["count_pp_hydrant_rank"], gdf_311_summ["count_pp_hydrant_q5"] = (
        custom_qcut_function(gdf_311_summ["count_pp_hydrant"])
    )
    return gdf_311_summ

##############################
# EMS PROCESSING
##############################


def load_ems():
    """Load EMS data via Open Data API"""
    print("------------------------")
    print("Loading 311 hydrant data via API")
    if not os.path.isfile("./_data/ems_data.parquet"):
        df_ems = pd.read_csv(
            "https://data.cityofnewyork.us/resource/76xm-jjuj.csv?$limit=100000000&$where=first_activation_datetime>%272021-01-01%27%20and%20(initial_call_type=%27HEAT%27%20or%20final_call_type%20=%20%27HEAT%27)"
        )
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
    df = df[(df["year"] >= 2021) & (df["year"] <= 2025) & (df["month"].isin([5, 6, 7, 8, 9]))]
    weekly_dates = pd.date_range(
        start=df["week"].min(), end=df["week"].max(), freq="W-MON"
    ).tolist()

    df_ems_summ = create_grid(zcta_geo["zcta"].unique(), weekly_dates, "zcta").merge(
        df, on=["zcta", "week"], how="left"
    )
    df_ems_summ["count"] = df_ems_summ["count"].fillna(0)
    df_ems_summ = df_ems_summ.groupby(["zcta"], as_index=False)["count"].mean()
    df_ems_summ["ems_count_rank"], df_ems_summ["ems_count_q5"] = custom_qcut_function(
        df_ems_summ["count"]
    )
    return df_ems_summ

##############################
# DPS PROCESSING
##############################

def clean_dps(dps_geo, compute_daily_max=False):
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
    df_dps["PRIME_DPS_"] = df_dps["DPS_ID"].str.split(".").apply(lambda x: x[0])

    # for DPS service localities that need to merged, summ total customers and total customers out
    df_dps = df_dps.groupby(
        ["PRIME_DPS_", "SUBMIT_DATE", "SUBMIT_TIME"], as_index=False
    )[["TOTAL_CUSTOMERS", "CUSTOMERS_OUT"]].sum()
    df_dps = generate_weekly_date(df_dps, "SUBMIT_DATE")

    assert (df_dps["TOTAL_CUSTOMERS"] > 0).all()

    df_dps = df_dps[(df_dps["year"] >= 2021) & (df_dps["month"].isin([5, 6, 7, 8, 9]))]
    dates = pd.date_range(
        start=df_dps["date"].min(), end=df_dps["date"].max(), freq="D"
    ).tolist()

    if compute_daily_max == True:
        # compute outage rate at 30 min intervals
        df_dps["CUSTOMERS_OUT_RATE"] = (
            df_dps["CUSTOMERS_OUT"] * 1000 / df_dps["TOTAL_CUSTOMERS"]
        )
        # compute the daily max value
        df_dps_summ = df_dps.groupby(["date", "PRIME_DPS_"], as_index=False)[
            "CUSTOMERS_OUT_RATE"
        ].max()

        df_dps_summ["date"] = pd.to_datetime(df_dps_summ["date"])
        df_dps_summ = create_grid(
            dps_geo["PRIME_DPS_"], dates, "PRIME_DPS_", "date"
        ).merge(df_dps_summ, on=["PRIME_DPS_", "date"], how="left")
        df_dps_summ["CUSTOMERS_OUT_RATE"] = df_dps_summ["CUSTOMERS_OUT_RATE"].fillna(0)
    else:
        df_dps_dedup = df_dps.sort_values(
            "CUSTOMERS_OUT", ascending=False
        ).drop_duplicates(subset=["date", "PRIME_DPS_"], keep="first")

        df_dps_summ = df_dps_dedup.groupby(
            ["PRIME_DPS_", "week", "year", "month"], as_index=False
        )["CUSTOMERS_OUT"].mean()
        dates = pd.date_range(
            start=df_dps_dedup["date"].min(),
            end=df_dps_dedup["date"].max(),
            freq="W-MON",
        ).tolist()
        df_dps_summ = (
            create_grid(dps_geo["PRIME_DPS_"], dates, "PRIME_DPS_")
            .merge(df_dps_summ, on=["PRIME_DPS_", "week"], how="left")
            .merge(
                df_dps_dedup.groupby(["PRIME_DPS_"], as_index=False)[
                    "TOTAL_CUSTOMERS"
                ].max()
            )
        )

        df_dps_summ["CUSTOMERS_OUT_RATE"] = (
            df_dps_summ["CUSTOMERS_OUT"] * 1000 / df_dps_summ["TOTAL_CUSTOMERS"]
        )
        df_dps_summ["CUSTOMERS_OUT_RATE"] = df_dps_summ["CUSTOMERS_OUT_RATE"].fillna(0)

    df_dps_summ = df_dps_summ.groupby("PRIME_DPS_", as_index=False)[
        "CUSTOMERS_OUT_RATE"
    ].mean()
    df_dps_summ["CUSTOMERS_OUT_RATE_rank"], df_dps_summ["CUSTOMERS_OUT_RATE_q5"] = (
        custom_qcut_function(df_dps_summ["CUSTOMERS_OUT_RATE"])
    )
    df_dps_tract_summ = df_dps_summ.merge(
        dps_tract_xwalk[["PRIME_DPS_", "geoid"]], on="PRIME_DPS_"
    )
    return df_dps_tract_summ