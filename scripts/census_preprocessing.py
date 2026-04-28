from census import Census
import geopandas as gpd
import pandas as pd
import numpy as np

from scripts.utils import *

# load acs data dictionary / cross-walk
# data dictioanry of variables and names to retrieve
acs_dict = (
    pd.read_excel("./_docs/acs_dict.xlsx").set_index("colname").to_dict()["textname"]
)

###############
# Load Data
###############


def load_acs(census, tract_geo, load_acs_data=True, year=2020):
    """Load ACS data via API or cached"""
    if load_acs_data == True:
        # load data from ACS
        print("------------------------")
        print(f"Loading data from the American Community Survey (ACS) for {year}")
        acs_df = pd.DataFrame(
            census.acs5.get(
                (tuple(["NAME", "GEO_ID"] + list(acs_dict.keys()))),
                geo={
                    "for": f"tract:*",
                    "in": f"state:36 county:061,005,081,085,047",
                },
                year=year,
            )
        ).rename(columns=acs_dict)

        acs_df["geoid"] = acs_df["GEO_ID"].str[9:]
        acs_df.to_parquet("./_data/acs_data.parquet")
    else:
        acs_df = pd.read_parquet("./_data/acs_data.parquet")
    acs_gdf = tract_geo.merge(acs_df, on="geoid")
    print(f"Data size: {acs_gdf.shape}")
    return acs_gdf


def load_dec(census, tract_geo, load_dec_data=True, year=2020):
    """Load Decennial 2020 census data via API or cached"""
    if load_dec_data == True:
        # get decennial census data
        dec_df = pd.DataFrame(
            census.pl.get(
                (tuple(["NAME", "GEO_ID"] + ["P1_001N", "P2_006N", "P2_005N"])),
                geo={
                    "for": f"tract:*",
                    "in": f"state:36 county:061,005,081,085,047",
                },
                year=year,
            )
        ).rename(
            columns={
                "P1_001N": "totalpop_dec",
                "P2_006N": "black_nh_dec",
                "P2_005N": "white_nh_dec",
            }
        )

        dec_df["geoid"] = dec_df["GEO_ID"].str[9:]
        dec_df.to_parquet("./_data/dec_data.parquet")
    else:
        dec_df = pd.read_parquet("./_data/dec_data.parquet")
    dec_gdf = tract_geo.merge(dec_df, on="geoid")
    print(f"Data size: {dec_gdf.shape}")
    return dec_gdf


def load_acs_zcta(census, zcta_geo, load_acs_data=True, year=2020):
    """Load data from the 2016-2020 ACS at ZCTA level"""
    if load_acs_data == True:
        # load data from ACS (ZCTA)
        acs_zcta = pd.DataFrame(
            census.acs5.get(
                (tuple(["NAME", "GEO_ID"] + list(acs_dict.keys()))),
                geo={"for": f"zip code tabulation area:*"},
                year=year,
            )
        ).rename(columns=acs_dict)
        acs_zcta.to_parquet("./_data/acs_data_zcta.parquet")
    else:
        acs_zcta = pd.read_parquet("./_data/acs_data_zcta.parquet")

    acs_zcta["zcta"] = acs_zcta["GEO_ID"].str[-5:]

    # left join because two are not merging
    acs_zcta_gdf = zcta_geo.merge(acs_zcta, on="zcta", how="left")
    print(
        f"Unmerged: {', '.join(zcta_geo[~zcta_geo['zcta'].isin(acs_zcta['zcta'])]['zcta'])}"
    )
    print(f"Data size: {acs_zcta_gdf.shape}")
    return acs_zcta_gdf


def load_census(tract_geo, zcta_geo, load_acs_data=True, load_dec_data=True, year=2020):
    """Load data from the Decennial Census (2020)"""
    census = Census("")

    acs_gdf = load_acs(census, tract_geo, load_acs_data=load_acs_data, year=year)
    acs_zcta_gdf = load_acs_zcta(
        census, zcta_geo, load_acs_data=load_acs_data, year=year
    )
    dec_gdf = load_dec(census, tract_geo, load_dec_data=load_dec_data, year=year)

    # merge acs and dec (subset)
    acs_gdf = acs_gdf.merge(
        dec_gdf[["geoid", "totalpop_dec", "black_nh_dec", "white_nh_dec"]],
        on="geoid",
        how="left",
    )

    return (acs_gdf, acs_zcta_gdf, dec_gdf)


##############
# Clean data
##############

# clean census data
age_vars = [
    "totalpop_female_65to66",
    "totalpop_female_67to69",
    "totalpop_female_70to74",
    "totalpop_female_75to79",
    "totalpop_female_80to84",
    "totalpop_female_over85",
    "totalpop_male_65to66",
    "totalpop_male_67to69",
    "totalpop_male_70to74",
    "totalpop_male_75to79",
    "totalpop_male_80to84",
    "totalpop_male_over85",
]
acs_pop_vars = [
    "totalpop",
    "poverty_status_inpoverty",
    "inpoverty_75over_male",
    "inpoverty_75over_female",
    "hh_gt65",
    "total_hh_age",
    "total_over75",
    "black",
]

dec_pop_vars = [
    "totalpop_dec",
    "black_nh_dec",
    "white_nh_dec",
]

census_cols_hvi = (
    [
        "median_hhinc",
        "geometry",
        "geoid",
        "nta2020",
    ]
    + age_vars
    + acs_pop_vars
    + dec_pop_vars
)

pct_vars = [
    "pct_over65",
    "pct_inpoverty",
    "pct_inpoverty_75over",
    "pct_hh_gt65",
    "pct_black",
    "pct_over_75",
    "nonwhite_nh_dec_pct",
]


def produce_pct(df):
    """Produce percentages for relevant ACS variabels"""
    # compute percentages
    df.loc[:, "pct_black"] = df["black_nh_dec"] / df["totalpop_dec"]
    df.loc[:, "pct_hh_gt65"] = df["hh_gt65"] / df["total_hh_age"]  # this is the denom
    # percent individuals over 75 and in poverty / total pop
    df.loc[:, "pct_inpoverty_75over"] = (
        df[["inpoverty_75over_male", "inpoverty_75over_female"]].sum(axis=1)
    ) / df["totalpop"]

    # pct in poverty / total pop
    df.loc[:, "pct_inpoverty"] = (df["poverty_status_inpoverty"]) / df["totalpop"]

    # create pct over 65 variable
    df.loc[:, "pct_over65"] = (df[age_vars].sum(axis=1)) / df["totalpop"]
    df.loc[:, "pct_over_75"] = (df["total_over75"]) / df["totalpop"]

    df.loc[:, "nonwhite_nh_dec_pct"] = 1 - (
        df.loc[:, "white_nh_dec"] / df.loc[:, "totalpop_dec"]
    )

    # check percent vars are within 0 and 1
    assert (df[pct_vars].min() >= 0).all()
    assert (df[pct_vars].max() <= 1).all()
    return df


def clean_acs_hvi(acs_gdf):
    """Clean ACS data for HVI replication"""
    acs_gdf_hvi = acs_gdf[census_cols_hvi].copy()

    # drop census tracts with 0 pop
    acs_gdf_hvi = acs_gdf_hvi[acs_gdf_hvi["totalpop_dec"] > 0]

    # check that no values are missing and/or negative
    check_missing_negative_value(acs_gdf_hvi)

    acs_gdf_hvi.loc[:, "median_hhinc"] = np.where(
        acs_gdf_hvi["median_hhinc"] < 0, 0, acs_gdf_hvi["median_hhinc"]
    )
    # check again
    check_missing_negative_value(acs_gdf_hvi)

    acs_gdf_hvi = produce_pct(acs_gdf_hvi)
    return acs_gdf_hvi
