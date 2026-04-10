import pandas as pd

measures_list_2020 = [
    "Stroke among adults aged >=18 years",
    "Diagnosed diabetes among adults aged >=18 years",
    "High cholesterol among adults aged >=18 years who have been screened in the past 5 years",
    "No leisure-time physical activity among adults aged >=18 years",
    "Cholesterol screening among adults aged >=18 years",
    "Current asthma among adults aged >=18 years",
    "Mental health not good for >=14 days among adults aged >=18 years",
    "Obesity among adults aged >=18 years",
    "Physical health not good for >=14 days among adults aged >=18 years",
    "Coronary heart disease among adults aged >=18 years",
    "Chronic obstructive pulmonary disease among adults aged >=18 years",
    "High blood pressure among adults aged >=18 years",
    "Taking medicine for high blood pressure control among adults aged >=18 years with high blood pressure",
]

measures_list_2024 = [
    "Stroke among adults",
    "Diagnosed diabetes among adults",
    "High cholesterol among adults who have ever been screened",
    "No leisure-time physical activity among adults",
    "Cholesterol screening among adults",
    "Current asthma among adults",
    "Mobility disability among adults",
    "Frequent mental distress among adults",
    "Obesity among adults",
    "Frequent physical distress among adults",
    "Coronary heart disease among adults",
    "Self-care disability among adults",
    "Chronic obstructive pulmonary disease among adults",
    "High blood pressure among adults",
    "Taking medicine to control high blood pressure among adults with high blood pressure",
]


def clean_cdc_places(df, id_var="geoid"):
    """Cleans CDC places data and converts data from long to wide"""
    print("------------------------")
    print("Cleaning CDC Places Data")
    print(f"id var = {id_var}")

    if pd.api.types.is_numeric_dtype(df[id_var]):
        df[id_var] = df[id_var].astype(str)

    df_unhealthy = df[df["measure"].isin(measures_list_2024)]
    print(
        f"Number of unique health measures: {df_unhealthy[["measure", "measureid"]].drop_duplicates().shape[0]}"
    )
    print(f"Conditions: {df_unhealthy[['measure', 'measureid']].drop_duplicates()}")

    df_unhealthy_wide = df_unhealthy.pivot(
        index=[id_var], columns=["measureid"], values=["data_value"]
    ).reset_index()
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

    data_cols = [x for x in df_unhealthy_wide.columns if "data_value_" in x]

    for col in data_cols:
        df_unhealthy_wide[col + "_total"] = (
            df_unhealthy_wide[col] / 100 * df_unhealthy_wide["totalpop18plus"]
        )
    df_unhealthy_wide["data_value_HIGHCHOL_total"] = (
        df_unhealthy_wide["data_value_HIGHCHOL"]
        / 100
        * df_unhealthy_wide["data_value_CHOLSCREEN_total"]
    )
    df_unhealthy_wide["data_value_BPMED_total"] = (
        df_unhealthy_wide["data_value_BPMED"]
        / 100
        * df_unhealthy_wide["data_value_BPHIGH_total"]
    )

    total_cols = [x for x in df_unhealthy_wide.columns if "_total" in x]
    print("Printing average total values:")
    print(df_unhealthy_wide[total_cols].mean().round(3).astype(str))
    return df_unhealthy_wide


def cdc_nta_cleaning(df, health_cols):
    """Produce percentages at the NTA level"""
    for var in health_cols:
        if "totalpop" not in var.lower():
            df[var.replace("_total", "_pct")] = df[var] / df["totalpop18plus"]
    df["data_value_HIGHCHOL_pct"] = (
        df["data_value_HIGHCHOL_total"] / df["data_value_CHOLSCREEN_total"]
    )
    df["data_value_BPMED_pct"] = (
        df["data_value_BPMED_total"] / df["data_value_BPHIGH_total"]
    )

    health_cdc_pct_cols = [
        x.replace("_total", "_pct") for x in health_cols if "totalpop" not in x
    ]

    assert (df[health_cdc_pct_cols].min() >= 0).all()
    assert (df[health_cdc_pct_cols].max() <= 1).all()

    df["avg_cdc_health_vars"] = (
        df[health_cdc_pct_cols].mean(axis=1, skipna=True).fillna(0)
    )
    return df, health_cdc_pct_cols
