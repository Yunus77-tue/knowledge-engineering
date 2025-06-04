#import libraries

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
import geopandas as gpd


def load_and_clean_gemeente_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w_]', '', regex=True)
        .str.rstrip('_')
    )
    return df

def load_provincie_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[['gemeente', 'provincie']]
    
    
    return df

def load_proximity_data_merge(path1,path2, sep=';', encoding='utf-8') -> pd.DataFrame:
    #load proximity data from two csv files
    df1 = pd.read_csv(path1, sep=sep, encoding=encoding)
    df2 = pd.read_csv(path2, sep=sep, encoding=encoding)    
    df1 = df1[df1['DistrictsAndNeighbourhoods'].astype(str).str.startswith('GM')].copy()
    df2 = df2[df2['DistrictsAndNeighbourhoods'].astype(str).str.startswith('GM')].copy()

    #merge the two dataframes
    merged_df  = df1.merge(df2, left_on='DistrictsAndNeighbourhoods', right_on='DistrictsAndNeighbourhoods', how='left')

    #rename columns for clarity
    merged_df.rename(columns={
        'DistrictsAndNeighbourhoods': 'gemeente_code',
        'MunicipalityName_1_x': 'gemeente',
        'DistanceToGPPractice_5': 'afstand_tot_huisartsenpraktijk',
        'Within1Km_6': 'huisartsenpraktijken_binnen_1km',
        'Within3Km_7': 'huisartsenpraktijken_binnen_3km',
        'Within5Km_8': 'huisartsenpraktijken_binnen_5km',
        'DistanceToHospital_11': 'afstand_tot_ziekenhuis',
        'Within5Km_12': 'ziekenhuizen_binnen_5km',
        'Within10Km_13': 'ziekenhuizen_binnen_10km',
        'Within20Km_14': 'ziekenhuizen_binnen_20km',
        'DistanceToLargeSupermarket_24': 'afstand_tot_supermarkt',
        'Within1Km_25': 'supermarkten_binnen_1km',
        'Within3Km_26': 'supermarkten_binnen_3km',
        'Within5Km_27': 'supermarkten_binnen_5km',
        'DistanceToRestaurant_44': 'afstand_tot_restaurant',
        'Within1Km_45': 'restaurants_binnen_1km',
        'Within3Km_46': 'restaurants_binnen_3km',
        'Within5Km_47': 'restaurants_binnen_5km',
        'Within5Km_49': 'sportvoorzieningen_binnen_5km',   
        'Within105km_50': 'sportvoorzieningen_binnen_10km',   
        'Within20Km_51': 'sportvoorzieningen_binnen_20km',
        'ID_x': 'id_oorspronkelijk',
        'ID_y': 'id_extra',
        'MunicipalityName_1_y': 'gemeente_extra',
        'DistanceToDaycareCentres_52': 'afstand_tot_kinderopvang',
        'Within1Km_53': 'kinderopvang_binnen_1km',
        'Within3Km_54': 'kinderopvang_binnen_3km',
        'Within5Km_55': 'kinderopvang_binnen_5km',
        'DistanceToSchool_60': 'afstand_tot_basisschool',
        'Within1Km_61': 'basisschool_binnen_1km',
        'Within3Km_62': 'basisschool_binnen_3km',
        'Within5Km_63': 'basisschool_binnen_5km',
        'DistanceToSchool_64': 'afstand_tot_voortgezet_onderwijs',
        'Within3Km_65': 'voortgezet_onderwijs_binnen_3km',
        'Within5Km_66': 'voortgezet_onderwijs_binnen_5km',
        'Within10Km_67': 'voortgezet_onderwijs_binnen_10km',
        'DistanceToMainRoadEntrance_89': 'afstand_tot_oprit_snelweg',
        'DistanceToTrainStationAllTypes_90': 'afstand_tot_treinstation',
        'DistanceToImportantTransferStation_91': 'afstand_tot_belangrijk_knooppunt'
            }, inplace=True)
    
    
    #remove the following columns
    columns_to_remove = ['gemeente_extra', 'id_oorspronkelijk', 'id_extra']
    merged_df.drop(columns=columns_to_remove, inplace=True)
    
    return merged_df


def merge_gemeente_with_provincie(df_km: pd.DataFrame, df_prov: pd.DataFrame) -> pd.DataFrame:
    return df_km.merge(df_prov, on='gemeente', how='left')


def apply_manual_province_mapping(df: pd.DataFrame) -> pd.DataFrame:
    manual_mapping = {
    'Beek': 'Limburg',
    'Den Haag': 'Zuid-Holland',
    'Hengelo': 'Overijssel',
    'Laren': 'Noord-Holland',
    'Middelburg': 'Zeeland',
    'Rijswijk': 'Zuid-Holland',
    'Stein': 'Limburg'
    }
    mask = df['provincie'].isna()
    df.loc[mask, 'provincie'] = df.loc[mask, 'gemeente'].map(manual_mapping)
    return df

def replace_zeros_with_nan(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].replace(0, np.nan)
    return df

def fill_missing_with_province_median(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        df[col] = df.groupby('provincie')[col].transform(lambda x: x.fillna(x.median()))
    return df

def normalize_gemeente_column(df, col='gemeente'):
    df[col] = df[col].astype(str).str.rstrip()
    return df


# datasets
km_path = "Datasets/Regionale_klimaat_monitor_2023.xlsx"
prov_path = "Datasets/provincie_gemeente.xlsx"
proximity_fp = "Datasets/proximity_first_part.csv"
proximity_sp = "Datasets/proximity_second_part.csv"



# Verwerk de data
df_km = load_and_clean_gemeente_data(km_path)

df_proximity_merge = load_proximity_data_merge(proximity_fp, proximity_sp)
df_km = normalize_gemeente_column(df_km)
df_proximity_merge = normalize_gemeente_column(df_proximity_merge)
df_prov = load_provincie_data(prov_path)
df_km = merge_gemeente_with_provincie(df_km, df_prov)
df_km = apply_manual_province_mapping(df_km)

# Kolommen waarin 0 betekent: 'missing'
income_columns = ['gemiddeld_inkomen_per_inwoner', 'gemiddeld_inkomen_per_huishouden']
df_km = replace_zeros_with_nan(df_km, income_columns)
df_km = fill_missing_with_province_median(df_km, income_columns)

#merge df_km with proximity data

df_final = df_km.merge(df_proximity_merge, left_on='gemeente', right_on='gemeente', how='left')
print(df_final.columns)

#save the cleaned data
output_path = "Datasets/final_dataset.csv"
df_final.to_csv(output_path, index=False)

