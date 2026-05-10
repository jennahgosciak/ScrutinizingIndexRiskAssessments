import pandas as pd
from scripts.utils import *

measures_list_2024 = [
    "Stroke among adults",
    "Diagnosed diabetes among adults",
    "High cholesterol among adults who have ever been screened",
    "No leisure-time physical activity among adults",
    "Cholesterol screening among adults",
    "Current asthma among adults",
    "Frequent mental distress among adults",
    "Obesity among adults",
    "Frequent physical distress among adults",
    "Coronary heart disease among adults",
    "Chronic obstructive pulmonary disease among adults",
    "High blood pressure among adults",
    "Taking medicine to control high blood pressure among adults with high blood pressure",
]

########################
# Load CDC Places Data
########################


def load_cdc_places(zcta_geo, nyc_counties, load_cdc_places_data=True):
    """Loading data from CDC Places via API or cache"""
    print("------------------------")
    print("Loading CDC Places Data")

    if load_cdc_places_data:
        endpoint_path = "https://data.cdc.gov/resource"

        tract_endpoint = "ai6z-tcin"
        zcta_endpoint = "4r2x-hcfq"

        # Load tract data
        df_cdc = pd.read_csv(
            f"{endpoint_path}/{tract_endpoint}.csv?$limit=1000000000&$where=STATEABBR='NY'"
        ).rename(columns={"locationname": "geoid"})

        # subset to nyc counties
        df_cdc = df_cdc[df_cdc["countyfips"].isin(nyc_counties)]
        print(
            f"Number of unique tracts in 2024 data: {df_cdc['geoid'].unique().shape[0]}"
        )

        # Load ZCTA data
        df_cdc_zcta = pd.read_csv(
            f"{endpoint_path}/{zcta_endpoint}.csv?$limit=1000000000"
        ).rename(columns={"locationname": "zcta"})
        df_cdc_zcta["zcta"] = df_cdc_zcta["zcta"].astype(str)
        df_cdc_zcta = df_cdc_zcta[df_cdc_zcta["zcta"].isin(zcta_geo["zcta"])]
        print(
            f"Number of unique ZCTAs in 2024 data: {df_cdc_zcta['zcta'].unique().shape[0]}"
        )

        # save data
        df_cdc.to_parquet("./_data/cdc_places_tract.parquet")
        df_cdc_zcta.to_parquet("./_data/cdc_places_zcta.parquet")
    else:
        df_cdc = pd.read_parquet("./_data/cdc_places_tract.parquet")
        df_cdc_zcta = pd.read_parquet("./_data/cdc_places_zcta.parquet")

        # print unique values of census tract ID and ZCTA ID
        print(
            f"Number of census tracts in CDC Places: {df_cdc['geoid'].unique().shape[0]}"
        )
        print(f"Number of ZCTAs in CDC Places: {df_cdc_zcta['zcta'].unique().shape[0]}")
    return df_cdc, df_cdc_zcta


def clean_cdc_places(df, id_var="geoid"):
    """Cleans CDC places data via API and converts data from long to wide"""
    print("------------------------")
    print("Cleaning CDC Places Data")
    print(f"id var = {id_var}")

    # convert id variable to string if numeric
    if pd.api.types.is_numeric_dtype(df[id_var]):
        df[id_var] = df[id_var].astype(str)

    # ensure that we only keep the relevant health measures in list
    df_unhealthy = df[df["measure"].isin(measures_list_2024)]

    # check that we have correct number of health measures
    assert df_unhealthy["measure"].drop_duplicates().shape[0] == len(measures_list_2024)
    print(
        f"Number of unique health measures: {df_unhealthy[['measure', 'measureid']].drop_duplicates().shape[0]}"
    )
    print(f"Conditions: {df_unhealthy[['measure', 'measureid']].drop_duplicates()}")

    # pivot data wide by measure id
    df_unhealthy_wide = df_unhealthy.pivot(
        index=[id_var], columns=["measureid"], values=["data_value"]
    ).reset_index()

    # remove hierarchical index
    df_unhealthy_wide.columns = [
        "_".join(col) for col in df_unhealthy_wide.columns.values
    ]

    # check there are no duplicate population values
    assert (
        df_unhealthy[[id_var, "totalpop18plus"]].drop_duplicates().shape[0]
        == df_unhealthy[id_var].unique().shape[0]
    )
    df_unhealthy_wide = df_unhealthy_wide.rename(columns={id_var + "_": id_var}).merge(
        df_unhealthy[[id_var, "totalpop18plus"]].drop_duplicates(),
        how="left",
        on=id_var,
    )
    print(f"\nShape of wide data: {df_unhealthy_wide.shape[0]}")
    print(f"# of unique locations: {df_unhealthy_wide[id_var].unique().shape[0]}")

    # collect columns for health measures
    data_cols = [x for x in df_unhealthy_wide.columns if "data_value_" in x]
    # for all column with health measures, produce estimate of the numerator
    for col in data_cols:
        if col not in ["data_value_HIGHCHOL", "data_value_BPMED"]:
            print(f"Computing numerator estimate for {col}")
            df_unhealthy_wide[col + "_total"] = (
                df_unhealthy_wide[col] / 100
            ) * df_unhealthy_wide["totalpop18plus"]

    # for high cholesterol denominator must be those with cholscreen
    df_unhealthy_wide["data_value_HIGHCHOL_total"] = (
        df_unhealthy_wide["data_value_HIGHCHOL"]
        / 100
        * df_unhealthy_wide["data_value_CHOLSCREEN_total"]
    )

    # for taking blood pressure med, denominator is those with high blood presssure
    df_unhealthy_wide["data_value_BPMED_total"] = (
        df_unhealthy_wide["data_value_BPMED"]
        / 100
        * df_unhealthy_wide["data_value_BPHIGH_total"]
    )

    total_cols = [x for x in df_unhealthy_wide.columns if "_total" in x]

    # checking averages overall, and that these seem reasonable
    print("\nPrinting average numerator values:")
    print(df_unhealthy_wide[total_cols].mean().round(3).astype(str))

    # check for unique ID
    check_unique_id(df_unhealthy_wide, id_var)
    return df_unhealthy_wide


def cdc_nta_cleaning(df, health_cols: list[str]):
    """Produce percentage estimates for CDC places data at the NTA level"""
    # create a copy before modifying
    df_nta = df.copy()

    """Produce CDC Places percentage estimates at the NTA level"""
    for col in health_cols:
        if ("totalpop" not in col.lower()) and (
            col not in ["data_value_HIGHCHOL_total", "data_value_BPMED_total"]
        ):
            print(f"Recomputing NTA percentage for {col}")
            df_nta[col.replace("_total", "_pct")] = (
                df_nta[col] / df_nta["totalpop18plus"]
            )

    df_nta["data_value_HIGHCHOL_pct"] = (
        df_nta["data_value_HIGHCHOL_total"] / df_nta["data_value_CHOLSCREEN_total"]
    )
    df_nta["data_value_BPMED_pct"] = (
        df_nta["data_value_BPMED_total"] / df_nta["data_value_BPHIGH_total"]
    )

    # return list of all health columns in the data
    health_cdc_pct_cols = [
        x.replace("_total", "_pct") for x in health_cols if "totalpop" not in x
    ]

    # check percentages are all within 0, 1 range
    assert (df_nta[health_cdc_pct_cols].min() >= 0).all()
    assert (df_nta[health_cdc_pct_cols].max() <= 1).all()

    # produce a version that is just the max percent value across all health conditions
    df_nta["max_cdc_health_vars"] = (
        df_nta[health_cdc_pct_cols].fillna(0).max(axis=1, skipna=True).fillna(0)
    )
    return df_nta, health_cdc_pct_cols
