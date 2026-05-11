import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import zipfile
import os

from matplotlib import pyplot as plt


def default_plot(ax):
    """Default plot style"""
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()

    xlab.set_size(14)
    ylab.set_size(14)


#####################################################
# Load spatial data and other spatial oeprations
#####################################################


def load_zcta_rel_files(nyc_counties):
    """Load county- and tract-level ZCTA relationship files"""
    df_zcta_rel_file_tract = pd.read_csv(
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt",
        sep="|",
    )
    df_zcta_rel_file_tract["county_code"] = (
        df_zcta_rel_file_tract["GEOID_TRACT_20"].astype(str).str[:5].astype(int)
    )
    df_zcta_rel_file_tract = df_zcta_rel_file_tract[
        df_zcta_rel_file_tract["county_code"].isin(nyc_counties)
    ]
    df_zcta_rel_file_tract = df_zcta_rel_file_tract[
        df_zcta_rel_file_tract["OID_ZCTA5_20"].notna()
    ]

    df_zcta_rel_file_county = pd.read_csv(
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt",
        sep="|",
    )

    df_zcta_rel_file_county = df_zcta_rel_file_county[
        df_zcta_rel_file_county["GEOID_COUNTY_20"].isin(nyc_counties)
    ]
    df_zcta_rel_file_county = df_zcta_rel_file_county[
        df_zcta_rel_file_county["OID_ZCTA5_20"].notna()
    ]

    print(
        f"Number of unique ZCTAs {df_zcta_rel_file_county['OID_ZCTA5_20'].unique().shape[0]} in 2020 relfile"
    )
    if (
        df_zcta_rel_file_county["OID_ZCTA5_20"].unique().shape[0]
        != df_zcta_rel_file_tract["OID_ZCTA5_20"].unique().shape[0]
    ):
        print("ZCTA shape mismatched")
    return (df_zcta_rel_file_county, df_zcta_rel_file_tract)


# load nta crosswalk from open data
def load_nta_xwalk(open_data_path, nyc_tracts):
    """Loads the NYC NTA x Tract crosswalk from NYC Open Data"""
    nta_xwalk = pd.read_csv(f"{open_data_path}/hm78-6dwm.csv?$limit=10000")
    nta_xwalk["geoid"] = nta_xwalk["geoid"].astype(str)

    # check unmerged
    unmgd = nta_xwalk[(~nta_xwalk["geoid"].isin(nyc_tracts["geoid"]))]
    print(f'There are {unmgd.shape[0]} unmerged tracts: {", ".join(unmgd['geoid'])}')
    nta_xwalk = nta_xwalk[(nta_xwalk["geoid"].isin(nyc_tracts["geoid"]))]
    nta_count = nta_xwalk["geoid"].unique().shape[0]
    print(f"There are {nta_count} unique tracts (for all NTAs)")
    return nta_xwalk


def load_geospatial(open_data_path, nyc_counties, load_geospatial_data=True):
    """Loads all geospatial data for NYC"""
    print("------------------------")
    print("Loading NTA data")
    if load_geospatial_data:
        # nta geo
        nta_geo = gpd.read_file(f"{open_data_path}/9nt8-h7nd.geojson?$limit=100000").to_crs(
            2263
        )
        nta_geo.to_file("_data/nta_geo.geojson")
    else:
        nta_geo = gpd.read_file("_data/nta_geo.geojson")
    
    nta_count = nta_geo["nta2020"].unique().shape[0]
    print(f"There are {nta_count} unique neighborhoods")

    # load zcta data
    print("------------------------")
    print("Loading 2020 and 2010 ZCTA data")
    if load_geospatial_data:
        zcta_geo = gpd.read_file(
            "https://www2.census.gov/geo/tiger/TIGER2020/ZCTA520/tl_2020_us_zcta520.zip"
        ).to_crs(2263)

        zcta_2010_geo = gpd.read_file(
            "https://www2.census.gov/geo/tiger/TIGER2010/ZCTA5/2010/tl_2010_36_zcta510.zip"
        ).to_crs(2263)

         # cleaning ZCTA 2020
        zcta_geo = zcta_geo.rename(columns={"ZCTA5CE20": "zcta"})
        zcta_2010_geo = zcta_2010_geo.rename(columns={"ZCTA5CE10": "zcta"})

        # load all zcta values within 5 boros
        county_relfile, _ = load_zcta_rel_files(nyc_counties)
        county_relfile["GEOID_ZCTA5_20"] = (
            county_relfile["GEOID_ZCTA5_20"].astype(int).astype(str)
        )
        # subset zctas just to those in the county relfile
        zcta_geo = zcta_geo[zcta_geo["GEOID20"].isin(county_relfile["GEOID_ZCTA5_20"])]
        # exclude single zcta in Long Island
        zcta_geo = zcta_geo[~zcta_geo["zcta"].isin(["11040"])]

        # cleaning ZCTA 2010
        # load 2010 relfile
        df_zcta_rel_file_county = pd.read_csv(
            "https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_county_rel_10.txt"
        )
        df_zcta_rel_file_county = df_zcta_rel_file_county[
            df_zcta_rel_file_county["GEOID"].isin(nyc_counties)
        ]
        df_zcta_rel_file_county["ZCTA5"] = (
            df_zcta_rel_file_county["ZCTA5"]
            .astype(str)
            .str.pad(width=5, side="left", fillchar="0")
        )
        zcta_2010_geo = zcta_2010_geo[
            zcta_2010_geo["zcta"].isin(df_zcta_rel_file_county["ZCTA5"])
        ]
        # remove some areas that are technically naussau/Long Island
        zcta_2010_geo = zcta_2010_geo[~zcta_2010_geo["zcta"].isin(["11040"])]
        
        zcta_geo.to_file("_data/zcta_geo.geojson")
        zcta_2010_geo.to_file("_data/zcta_2010_geo.geojson")
    else:
        zcta_geo = gpd.read_file("_data/zcta_geo.geojson")
        zcta_2010_geo = gpd.read_file("_data/zcta_2010_geo.geojson")

    zcta_count = zcta_geo["zcta"].unique().shape[0]
    print(f"There are {zcta_count} unique ZCTAs in 2020")

    zcta_2010_count = zcta_2010_geo["zcta"].unique().shape[0]
    print(f"There are {zcta_2010_count} unique ZCTAs in 2010")
    if (zcta_geo["zcta"] != zcta_geo["GEOID20"]).any():
        raise Exception("zcta column does not match GEOID column")

    # load nyc tracts
    print("------------------------")
    print("Loading NYC Census Tracts")
    if load_geospatial_data:
        tract_geo = gpd.read_file(
            f"{open_data_path}/63ge-mke6.geojson?$limit=10000"
        ).to_crs(2263)
        tract_geo.to_file("_data/tract_geo.geojson")
    else:
        tract_geo = gpd.read_file("_data/tract_geo.geojson")
    tract_count = tract_geo["geoid"].unique().shape[0]
    print(f"There are {tract_count} unique tracts")

    # load boros
    print("------------------------")
    print("Loading NYC Boroughs")
    if load_geospatial_data:
        boros_geo = gpd.read_file(f"{open_data_path}/gthc-hcne.geojson").to_crs(2263)
        boros_geo.to_file("_data/boros_geo.geojson")
    else:
        boros_geo = gpd.read_file("_data/boros_geo.geojson")
    print(f"There are {boros_geo['borocode'].unique().shape[0]} unique boroughs")
    return (nta_geo, zcta_geo, zcta_2010_geo, tract_geo, boros_geo)


def produce_tract_points(tract_geo, method="centroid"):
    """Produce point data for census tracts based on the centroid or representative point"""
    tract_pt = tract_geo.copy()
    if method == "centroid":
        tract_pt["geometry"] = tract_pt["geometry"].centroid
    elif method == "representative_point":
        tract_pt["geometry"] = tract_pt["geometry"].representative_point()
    elif method == "spatial_overlap":
        print("No points for spatial overlap method")
    else:
        raise Exception("Method unknown")
    return tract_pt


# merge tracts to zcta, get zcta information at tract level
def tract_spatial_join(gdf, tract_data, method, spatial_id):
    """ZCTA to tract spatial join based on specified method"""
    if method in ["centroid", "representative_point"]:
        # produce tract pts (either centroid or representative point)
        tract_pt = produce_tract_points(tract_data, method)

        mgd_data = (
            tract_pt[["geoid", "geometry"]]
            .sjoin(gdf[[spatial_id, "geometry"]], how="left")
            .merge(
                gdf.drop(columns="geometry"),
                on=spatial_id,
                how="left",
            )
        )
    elif method == "spatial_overlap":
        print(f"Tract CRS: {tract_data.crs}")
        print(f"ZCTA CRS: {gdf.crs}")
        assert tract_data["geometry"].is_valid.all()
        assert gdf["geometry"].is_valid.all()

        overlap = gpd.overlay(
            tract_data, gdf, how="intersection", make_valid=True, keep_geom_type=False
        )
        assert (overlap["geometry"].is_valid).all()
        assert (
            overlap[
                (overlap.geometry.is_empty) | (overlap.geometry.is_empty.isna())
            ].shape[0]
            == 0
        )
        print(f"Overlap shape: {overlap.shape[0]}")

        overlap["area_overlap"] = overlap["geometry"].area
        overlap = overlap.sort_values(
            ["geoid", "area_overlap"], ascending=False
        ).drop_duplicates(subset="geoid", keep="first")
        print(f"Taking the max area from the intersection: {overlap.shape[0]}")
        xwalk = overlap[["geoid", spatial_id]]
        mgd_data = xwalk.merge(gdf.drop(columns="geometry"), on=spatial_id, how="left")
    else:
        raise Exception(f"{method} not recognized!")

    print(f"Merged data size: {mgd_data.shape}")
    print(f"Tract data size: {tract_data.shape}")

    unmgd = tract_data.loc[~tract_data["geoid"].isin(mgd_data["geoid"])]
    print(f"Census tracts that didn't merge to ZCTA/DPS (num = {unmgd.shape[0]}):")
    for i, row in unmgd.iterrows():
        print(
            f"      GEOID: {row['geoid']}, CDTA name: {row['cdtaname']}, and NTA name: {row['ntaname']}"
        )
    print(f"ZCTA/DPS data size: {gdf.shape}")
    return mgd_data


def check_census_relfile_matches(nyc_counties, zcta_geo, tract_geo):
    """Comparison  of the Census relationship file to a simple spatial overlap method"""
    _, tract_rel_file = load_zcta_rel_files(nyc_counties)
    tract_rel_file = tract_rel_file.rename(
        columns={"GEOID_TRACT_20": "geoid", "GEOID_ZCTA5_20": "zcta_relfile"}
    )
    tract_rel_file["geoid"] = tract_rel_file["geoid"].astype(int).astype(str)

    # remove rows where zcta is missing
    tract_rel_file = tract_rel_file[(tract_rel_file['zcta_relfile'].astype(int).notna())]
    print(f"Number of rows in rel file with non-missing ZCTA: {tract_rel_file.shape[0]}")

    zcta_tract_xwalk = tract_spatial_join(
        zcta_geo, tract_geo, method="spatial_overlap", spatial_id="zcta"
    )
    df_comparison = zcta_tract_xwalk.merge(tract_rel_file, on="geoid", how="inner")
    
    df_comparison["zcta_relfile"] = (
        df_comparison["zcta_relfile"].astype(int).astype(str)
    )
    df_comparison = df_comparison[df_comparison["OID_TRACT_20"].notna()]
    df_comparison = df_comparison.sort_values(
        by="AREALAND_PART", ascending=False
    ).drop_duplicates(subset=["OID_TRACT_20"], keep="first")

    print(
        f"\n% incorrect assignments: {(100*(df_comparison['zcta'] != df_comparison['zcta_relfile']).mean()).round(3)}%"
    )
    print(f"Dataset size: {df_comparison.shape[0]}")
    


####################
# Ranking
####################
def custom_qcut_function(var, method):
    """Creates quintiles based on percentile ranking"""
    if var.isna().sum() > 0:
        raise Exception("NA values in ranking var!")

    rank_var = var.rank(pct=True, method=method) * 100

    return rank_var, np.select(
        [
            rank_var <= 20,
            rank_var <= 40,
            rank_var <= 60,
            rank_var <= 80,
            rank_var <= 100,
        ],
        [1, 2, 3, 4, 5],
        np.nan,
    )


####################
# Load HVI
####################


def load_hvi_data(open_data_path, zcta_geo, nta_geo, load_data=True):
    """Loading datasets specific to the HVI"""
    print("------------------------")
    print("Loading HVI (ZCTA) data")
    if load_data == True:
        df_hvi_zcta = pd.read_csv(f"{open_data_path}/4mhf-duep.csv?$limit=100000")
        df_hvi_zcta.to_parquet("./_data/hvi_zcta.parquet")
    else:
        df_hvi_zcta = pd.read_parquet("./_data/hvi_zcta.parquet")

    print(
        f"There are {df_hvi_zcta['zcta20'].unique().shape[0]} unique ZCTAs in the data"
    )
    df_hvi_zcta = df_hvi_zcta.rename(columns={"zcta20": "zcta"})
    df_hvi_zcta["zcta"] = df_hvi_zcta["zcta"].astype(str)

    # compare to zcta geo, check nothing is unmerged
    if df_hvi_zcta[~df_hvi_zcta["zcta"].isin(zcta_geo["zcta"].unique())].shape[0] != 0:
        print(df_hvi_zcta[~df_hvi_zcta["zcta"].isin(zcta_geo["zcta"].unique())])

    print("------------------------")
    print("Loading HVI (NTA) data")
    if load_data == True:
        df_hvi_nta = pd.read_csv(
            "https://a816-dohbesp.nyc.gov/IndicatorPublic/data-features/hvi/hvi-nta-2020.csv"
        )
        df_hvi_nta.to_parquet("./_data/hvi_nta.parquet")
    else:
        df_hvi_nta = pd.read_parquet("./_data/hvi_nta.parquet")
    df_hvi_nta = (
        df_hvi_nta.drop(columns="GEOCODE")
        .drop_duplicates()
        .rename(columns={"NTACode": "nta2020"})
    )
    print(f"Data size: {df_hvi_nta.shape}")
    print(
        f"There are {df_hvi_nta['nta2020'].unique().shape[0]} unique NTAs in the data"
    )
    return df_hvi_zcta, df_hvi_nta


####################
# Load NRI
####################


def load_nri_data(nyc_counties, method, download_nri_data=True):
    """Loads NRI Data for NYC"""
    print("------------------------")
    print("Loading NRI data (tract-level)")
    if download_nri_data == True:
        nri_data = pd.read_csv(
            "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Census_Tracts/FeatureServer/replicafilescache/National_Risk_Index_Census_Tracts_-2131777716435920328.csv"
        )
        nri_data = nri_data[
            nri_data["State-County FIPS Code"].astype(int).isin(nyc_counties)
        ]
        if nri_data["State-County FIPS Code"].unique().shape[0] != 5:
            raise Exception("Not enough unique counties for NYC")
        nri_data.to_parquet("./_data/nri_data.parquet")
    else:
        nri_data = pd.read_parquet("./_data/nri_data.parquet")
    nri_data["geoid"] = nri_data["State-County FIPS Code"].astype(str) + nri_data[
        "Census Tract"
    ].astype(str).str.pad(width=6, fillchar="0", side="left")
    print(
        f"There are {nri_data['geoid'].unique().shape[0]} unique census tracts in the data"
    )

    print("Renaming columns and ranking")
    nri_data = nri_data.rename(
        columns={
            "Heat Wave - Expected Annual Loss - Total": "HWAV_EALT",
            "Heat Wave - Hazard Type Risk Index Value": "HWAV_EALTxSVIxRESL",
            "Heat Wave - Expected Annual Loss Rate - National Percentile": "HWAV_EALT_NatlPct",
            "Heat Wave - Hazard Type Risk Index Score": "HWAV_Score",
            "Heat Wave - Hazard Type Risk Index Rating": "HWAV_Rating",
        }
    )
    nri_data["HWAV_EALT_rank"], nri_data["HWAV_EALT_q5"] = custom_qcut_function(
        nri_data["HWAV_EALT"], method
    )
    nri_data["HWAV_EALTxSVIxRESL_rank"], nri_data["HWAV_EALTxSVIxRESL_q5"] = (
        custom_qcut_function(nri_data["HWAV_EALTxSVIxRESL"], method)
    )

    print("\nDistribution of quintiles")
    print(nri_data["HWAV_EALT_q5"].value_counts())
    print(nri_data["HWAV_EALTxSVIxRESL_q5"].value_counts())
    return nri_data


def load_uri():
    """Load URI data via URL"""
    url = "https://services1.arcgis.com/8cuieNI8NbqQZQVJ/arcgis/rest/services/NTA_URI_Data_View_3/FeatureServer/1//query?where=OBJECTID%3E0&objectIds=&geometry=&geometryType=esriGeometryEnvelope&inSR=&spatialRel=esriSpatialRelIntersects&resultType=none&distance=0.0&units=esriSRUnit_Meter&outDistance=&relationParam=&returnGeodetic=false&outFields=*&returnGeometry=true&returnCentroid=false&returnEnvelope=false&featureEncoding=esriDefault&multipatchOption=xyFootprint&maxAllowableOffset=&geometryPrecision=&outSR=&defaultSR=&datumTransformation=&applyVCSProjection=false&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnExtentOnly=false&returnQueryGeometry=false&returnDistinctValues=false&cacheHint=false&collation=&orderByFields=&groupByFieldsForStatistics=&returnAggIds=false&outStatistics=&having=&resultOffset=&resultRecordCount=&returnZ=false&returnM=false&returnTrueCurves=false&returnExceededLimitFeatures=true&quantizationParameters=&sqlFormat=none&f=pgeojson&token="
    r = requests.get(url)
    df_uri = r.json()
    return gpd.GeoDataFrame.from_features(df_uri, crs=4326).to_crs(2263)


########################
# Load CDC HHI Data
########################


def load_cdc_hhi_from_url():
    """Load HHI data from url and unzip folder locally"""
    print("------------------------")
    print("Loading CDC HHI Data")
    # download via url and save
    url = "https://www.atsdr.cdc.gov/place-health/media/files/2024/08/HHI_Data.zip"
    r = requests.get(url)
    zip_path = os.path.join("./_data/", "HHI_Data.zip")

    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("./_data/HHI_Data/")


def check_cdc_hhi_merge(df_cdc, zcta_geo, nyc_counties):
    """Load ZCTA 2010 county relationship file and check CDC HHI ZCTA merge"""
    df_zcta_rel_file_county = pd.read_csv(
        "https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_county_rel_10.txt"
    )
    df_zcta_rel_file_county = df_zcta_rel_file_county[
        df_zcta_rel_file_county["GEOID"].isin(nyc_counties)
    ]
    df_zcta_rel_file_county["zcta"] = (
        df_zcta_rel_file_county["ZCTA5"]
        .astype(str)
        .str.pad(width=5, side="left", fillchar="0")
    )

    # for those not merging, how many are in the relationship file? should only be one that was manually excluded
    assert (
        df_cdc[~df_cdc["zcta"].isin(zcta_geo["zcta"])].merge(
            df_zcta_rel_file_county, on="zcta"
        )["zcta"]
        == ["11040"]
    ).all()


def load_and_clean_hhi(zcta_geo, tract_geo, nyc_counties, join_method, rank_method):
    """Load HHI data from local folder and clean"""
    print("------------------------")
    print("Loading CDC HHI data from local file and cleaning")
    df_cdc_hhi = pd.read_excel("./_data/HHI_Data/HHI Data 2024 United States.xlsx")
    df_cdc_hhi = df_cdc_hhi[df_cdc_hhi["STATE"].isin(["NY"])].rename(
        columns={"ZCTA": "zcta"}
    )
    df_cdc_hhi["zcta"] = (
        df_cdc_hhi["zcta"].astype(str).str.pad(width=5, side="left", fillchar="0")
    )
    assert (df_cdc_hhi["zcta"].astype(str).apply(lambda x: len(x)).unique() == 5).all()

    # check cdc hhi merge to zcta
    check_cdc_hhi_merge(df_cdc_hhi, zcta_geo, nyc_counties)

    # subset to local data, join to zcta 2010 data
    df_cdc_hhi_geo = zcta_geo[["zcta", "geometry"]].merge(df_cdc_hhi, on="zcta")
    df_cdc_hhi_geo["PR_HRI"] = df_cdc_hhi_geo["PR_HRI"].astype(float)

    # produce quintiles and other rankings
    df_cdc_hhi_geo["PR_HRI_rank"], df_cdc_hhi_geo["PR_HRI_q5"] = custom_qcut_function(
        df_cdc_hhi_geo["PR_HRI"], method=rank_method
    )
    df_cdc_hhi_geo["OVERALL_SCORE_rank"], df_cdc_hhi_geo["OVERALL_SCORE_q5"] = (
        custom_qcut_function(df_cdc_hhi_geo["OVERALL_SCORE"], method=rank_method)
    )

    print(f"Data size: {df_cdc_hhi_geo.shape}")

    print("Producing tract -> ZCTA spatial join (for comparisons)")
    df_cdc_hhi_tract = tract_spatial_join(
        df_cdc_hhi_geo, tract_geo, method=join_method, spatial_id="zcta"
    )

    return df_cdc_hhi_geo, df_cdc_hhi_tract


###############
# Plotting
###############
def plot_simple_map(df, boros_geo, col, filename, categorical=False):
    df = df.copy()
    """Plot a simple map of NYC with boroughs"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    if categorical == False:
        df.plot(
            column=col,
            ax=ax,
            cmap="rocket_r",
            edgecolor="none",
            legend=True,
            legend_kwds={"shrink": 0.7},
        )
    else:
        df[col] = df[col].astype(int).astype(str)
        df.plot(
            column=col,
            ax=ax,
            cmap="rocket_r",
            edgecolor="none",
            legend=True,
        )
    boros_geo.plot(ax=ax, facecolor="none", edgecolor="gray", lw=0.3)
    ax.set_axis_off()
    plt.savefig(f"./_figures/{filename}", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.show()


def plot_all_indices(hvi, nri, cdc, boros_geo):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    hvi = hvi.copy()
    hvi["HVI_RANK"] = hvi["HVI_RANK"].astype(int).astype(str)
    hvi.plot(
        column="HVI_RANK",
        ax=axes[0],
        cmap="rocket_r",
        edgecolor="none",
        legend=True,
        legend_kwds={"loc": "upper left", "bbox_to_anchor": (0.1, 1)},
    )

    nri = nri.copy()
    nri["HWAV_EALTxSVIxRESL_q5"] = (
        nri["HWAV_EALTxSVIxRESL_q5"].astype(int).astype(str)
    )
    nri.plot(
        column="HWAV_EALTxSVIxRESL_q5",
        cmap="rocket_r",
        ax=axes[1],
        legend=False,
        edgecolor="none",
    )

    axes[1].set_title("(b) NRI")
    cdc = cdc.copy()
    cdc["OVERALL_SCORE_q5"] = cdc["OVERALL_SCORE_q5"].astype(int).astype(str)
    cdc.plot(
        column="OVERALL_SCORE_q5",
        ax=axes[2],
        cmap="rocket_r",
        edgecolor="none",
        legend=False,
    )
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    axes[2].set_axis_off()

    axes[0].set_title("(a) NYC HVI")
    axes[2].set_title("(c) CDC HHI")

    boros_geo.plot(ax=axes[0], facecolor="none", edgecolor="gray", lw=0.3)
    boros_geo.plot(ax=axes[1], facecolor="none", edgecolor="gray", lw=0.3)
    boros_geo.plot(ax=axes[2], facecolor="none", edgecolor="gray", lw=0.3)
    plt.savefig(
        "./_figures/all_index_tools.pdf", bbox_inches="tight", pad_inches=0, dpi=300
    )
    plt.tight_layout()
    plt.show()


def plot_lst(gdf_lst, boros_geo):
    """Plot average land surface temperature"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf_lst.plot(
        column="_mean_f",
        ax=ax,
        cmap="rocket_r",
        edgecolor="none",
        legend=True,
        legend_kwds={"shrink": 0.8},
    )
    ax.set_axis_off()
    boros_geo.plot(ax=ax, facecolor="none", edgecolor="gray", lw=0.3)
    plt.savefig(
        "./_figures/average_lst.pdf", bbox_inches="tight", pad_inches=0, dpi=300
    )
    plt.show()


######################
# Load ECOSTRESS Data
######################
def convert_temp_units(df, cols):
    """Converts Kelvin to Farhenheit"""
    for col in cols:
        df[col + "_f"] = (df[col] * 9 / 5) - 459.67
    return df


def load_ecostress_data(filename, id_col="geoid"):
    """Loads land surface temperature data pre-cleaned from raster"""
    print("------------------------")
    print("Loading ECOSTRESS land surface temperature")

    # load ecostress data
    df = gpd.read_file(f"./_data/{filename}")
    # subset cols
    df = df[
        [
            id_col,
            "_mean",
            "_min",
            "_max",
            "_range",
            "geometry",
        ]
    ]

    # convert kelvin temp to fahrenheit
    df = convert_temp_units(df, ["_mean", "_min", "_max", "_range"])

    if pd.api.types.is_numeric_dtype(df[id_col]):
        df[id_col] = df[id_col].astype(str)

    print(f"Data size: {df.shape}")
    return df


######################
# Load Vegetation Data
######################


def load_veg_data(filename, rank_method, id_col="geoid"):
    """Loads zonal histogram of vegetation data (generated via R code)"""
    print("------------------------")
    print("Loading vegetation data")

    df = pd.read_csv(f"./_data/{filename}").copy()

    df["pct_vegetation"] = 100 * (df["frac_1"] + df["frac_2"])
    df["pct_vegetation_rank"], df["pct_vegetation_q5"] = custom_qcut_function(
        df["pct_vegetation"], method=rank_method
    )

    if pd.api.types.is_numeric_dtype(df[id_col]):
        df[id_col] = df[id_col].astype(str)

    print(f"Data size: {df.shape}")
    return df[[id_col, "pct_vegetation", "pct_vegetation_rank", "pct_vegetation_q5"]]


##########################
# Data Manipulation/QA
##########################


def check_missing_negative_value(df):
    """Check missing or negative values of columns"""
    # check missing and negative values
    for var in df.columns:
        if pd.api.types.is_numeric_dtype(df[var]):
            missing_val = df[var].isna().sum()
            val_lt0 = (df[var] < 0).sum()
            notfinite = (np.isfinite(df[var])==False).sum()

            if missing_val > 0:
                print(f"Number of missing values for {var} is: {missing_val}")
            if val_lt0 > 0:
                print(f"Number of values < 0 for {var} is: {val_lt0}")
            if notfinite > 0:
                print(f"Number of values not finite for {var} is: {notfinite}")


def check_unique_id(df, id):
    """Check id variables uniquely identify data"""
    if df.shape[0] != df[id].drop_duplicates().shape[0]:
        raise Exception(f"Data not uniquely identified by {id}")


def standardize_values(df, vars, rank_method):
    """Produce z-scores, percentiles, and quintiles for inputs"""
    df = df.copy()
    for var in vars:
        df[var + "_z"] = (df[var] - df[var].mean()) / df[var].std()
        df[var + "_rank"], df[var + "_q5"] = custom_qcut_function(
            df[var], method=rank_method
        )
    return df


##########################
# Merging data
##########################
def merge_dfs(acs, veg, lst, cdc_places, id_col):
    """Merging data together"""
    print("------------------------")
    print("Merging datasets together")
    print(f"ACS data size: {acs.shape[0]}")
    df_mgd = acs.merge(veg, on=id_col, how="left").merge(lst, on=id_col, how="left")
    if cdc_places is not None:
        df_mgd = df_mgd.merge(cdc_places, on=id_col, how="left")

    print(f"Merged data size: {df_mgd.shape[0]}")
    # check uniqueness
    check_unique_id(df_mgd, id_col)
    return df_mgd


def produce_nta_summary(df, vars):
    """Produce NTA summary from tract data"""
    print("------------------------")
    print("Producing NTA summary from tract data")

    df_nta_summary = df.groupby("nta2020", as_index=False)[vars].sum()
    print(f"Number of NTAs: {df_nta_summary.shape}")
    check_unique_id(df_nta_summary, "nta2020")
    return df_nta_summary


def merge_tract_nta(nta, tract):
    """Merge tract- and NTA-level data"""
    df_tract_hvi = tract[
        [
            "geoid",
            "_mean_f",
            "_mean",
            "_max",
            "median_hhinc",
            "pct_black",
            "pct_vegetation",
            "nta2020",
        ]
    ].merge(
        nta[["nta2020", "PCT_HOUSEHOLDS_AC"]].drop_duplicates(),
        how="left",
        on="nta2020",
    )
    print(f"Data shape: {df_tract_hvi.shape}")

    check_unique_id(df_tract_hvi, "geoid")

    # drop cases where pct households w/ AC is missing
    df_tract_hvi = df_tract_hvi[df_tract_hvi["PCT_HOUSEHOLDS_AC"].notna()]
    print(f"After dropping missing AC values, data shape is: {df_tract_hvi.shape}")
    return df_tract_hvi
