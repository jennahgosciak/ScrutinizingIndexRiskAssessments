import pandas as pd

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


def clean_cdc_places(df, id_var="geoid"):
    """Load data from CDC Places via API"""
    """Cleans CDC places data and converts data from long to wide"""
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
        f"Number of unique health measures: {df_unhealthy[["measure", "measureid"]].drop_duplicates().shape[0]}"
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
    df_unhealthy_wide = df_unhealthy_wide.rename(columns={id_var + "_": id_var}).merge(
        df_unhealthy[[id_var, "totalpop18plus"]].drop_duplicates(),
        how="left",
        on=id_var,
    )
    print(f"\nShape of wide data: {df_unhealthy_wide.shape[0]}")
    print(f"# of unique locations: {df_unhealthy_wide[id_var].unique().shape[0]}")

    # collect columns for health measrues
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
    print("\nPrinting average total values:")
    print(df_unhealthy_wide[total_cols].mean().round(3).astype(str))
    return df_unhealthy_wide


def cdc_nta_cleaning(df, health_cols):
    """Produce CDC Places percentage estimates at the NTA level"""
    for col in health_cols:
        if ("totalpop" not in col.lower()) and (col not in ["data_value_HIGHCHOL_total", "data_value_BPMED"_total]):
            print(f"Recomputing percentage for {col}")
            df[col.replace("_total", "_pct")] = df[col] / df["totalpop18plus"]

    df["data_value_HIGHCHOL_pct"] = (
        df["data_value_HIGHCHOL_total"] / df["data_value_CHOLSCREEN_total"]
    )
    df["data_value_BPMED_pct"] = (
        df["data_value_BPMED_total"] / df["data_value_BPHIGH_total"]
    )

    # return list of all health columns in the data
    health_cdc_pct_cols = [
        x.replace("_total", "_pct") for x in health_cols if "totalpop" not in x
    ]

    # check percentages are all within 0, 1 range
    assert (df[health_cdc_pct_cols].min() >= 0).all()
    assert (df[health_cdc_pct_cols].max() <= 1).all()

    # produce a version that is just the max percent value across all health conditions
    df["max_cdc_health_vars"] = (
        df[health_cdc_pct_cols].max(axis=1, skipna=True).fillna(0)
    )
    return df, health_cdc_pct_cols
