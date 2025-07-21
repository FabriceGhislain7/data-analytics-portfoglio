"""
src/utils/data_loader.py
Funzioni per il caricamento e gestione dei dati Titanic
"""

import pandas as pd
import streamlit as st
import os
from src.config import DATA_FILE, DATA_URL

# ----------------1. Caricamento Dataset (da notebook sezione 2.1 - Structure of dataset)
@st.cache_data(ttl=3600)
def load_titanic_data():
    """
    Carica il dataset Titanic da file locale o URL remoto
    Implementa la logica di caricamento dal notebook sezione 2.1
    """
    try:
        # Prova prima a caricare da file locale
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            st.success(f"Dati caricati da file locale: {DATA_FILE}")
        else:
            # Carica da URL GitHub come nel notebook
            df = pd.read_csv(DATA_URL)
            st.info("Dati caricati da repository GitHub")
        
        return df
    
    except Exception as e:
        st.error(f"Errore nel caricamento dati: {str(e)}")
        return None

# ----------------2. Informazioni Dataset (da notebook sezione 2.1 - Dataset information)
def get_data_summary(df):
    """
    Calcola statistiche riassuntive del dataset
    Basato sull'analisi del notebook sezione 2.1
    """
    if df is None:
        return None
    
    # Calcoli base dal notebook
    total_passengers = len(df)
    survived_count = df['Survived'].sum()
    died_count = total_passengers - survived_count
    survival_rate = (survived_count / total_passengers) * 100
    missing_values = df.isnull().sum().sum()
    
    return {
        'rows': total_passengers,
        'columns': len(df.columns),
        'survived': survived_count,
        'died': died_count,
        'survival_rate': survival_rate,
        'missing_values': missing_values
    }

# ----------------3. Controllo Qualita Dati (da notebook sezione 2.2 - Missing values)
def get_missing_values_info(df):
    """
    Analizza i valori mancanti come nel notebook sezione 2.2
    """
    if df is None:
        return None
    
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Colonna': missing_counts.index,
        'Valori_Mancanti': missing_counts.values,
        'Percentuale': missing_percentages.values
    })
    
    return missing_info[missing_info['Valori_Mancanti'] > 0].sort_values('Valori_Mancanti', ascending=False)

# ----------------4. Controllo Duplicati (da notebook sezione 2.3 - Check duplicates)
def check_duplicates(df):
    """
    Verifica presenza duplicati come nel notebook sezione 2.3
    """
    if df is None:
        return 0
    
    return df.duplicated().sum()

# ----------------5. Preparazione Dati Base (da notebook sezione 3 - Data Cleaning)
def prepare_basic_dataset(df):
    """
    Applica pulizia base dei dati seguendo notebook sezione 3
    """
    if df is None:
        return None
    
    # Copia del dataframe originale
    df_clean = df.copy()
    
    # Rimuovi righe duplicate (sezione 3.1)
    df_clean = df_clean.drop_duplicates()
    
    # Rimuovi colonne con troppi valori mancanti (sezione 3.2)
    # Cabin ha 77% di missing values, la rimuoviamo
    if 'Cabin' in df_clean.columns:
        df_clean = df_clean.drop('Cabin', axis=1)
    
    return df_clean

# ----------------6. Dataset Statistics (da notebook sezione 4.1.1 - Descriptive Statistics)
def get_descriptive_statistics(df):
    """
    Calcola statistiche descrittive come nel notebook sezione 4.1.1
    """
    if df is None:
        return None
    
    # Seleziona solo colonne numeriche
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    stats = df[numeric_columns].describe()
    
    return stats