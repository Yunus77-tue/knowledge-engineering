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

# Bestanden
km_path = "Datasets/Regionale_klimaat_monitor_2023.xlsx"
prov_path = "Datasets/provincie_gemeente.xlsx"

# Verwerk de data
df_km = load_and_clean_gemeente_data(km_path)
df_prov = load_provincie_data(prov_path)
df_km = merge_gemeente_with_provincie(df_km, df_prov)
df_km = apply_manual_province_mapping(df_km)

# Kolommen waarin 0 betekent: 'missing'
income_columns = ['gemiddeld_inkomen_per_inwoner', 'gemiddeld_inkomen_per_huishouden']
df_km = replace_zeros_with_nan(df_km, income_columns)
df_km = fill_missing_with_province_median(df_km, income_columns)

