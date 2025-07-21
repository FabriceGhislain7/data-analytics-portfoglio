"""
src/utils/feature_engineering.py
Funzioni per feature engineering avanzato
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# ----------------1. Estrazione Titolo dal Nome (estensione analisi nomi)
def extract_title_from_name(df):
    """
    Estrae il titolo dal nome del passeggero
    Estende l'analisi dei nomi dal notebook
    """
    if df is None or 'Name' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Estrai titolo usando regex
    df_copy['Title'] = df_copy['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Raggruppa titoli rari in categorie comuni
    title_mapping = {
        'Mr': 'Mr',
        'Mrs': 'Mrs', 
        'Miss': 'Miss',
        'Master': 'Master',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Mlle': 'Miss',
        'Countess': 'Royalty',
        'Ms': 'Miss',
        'Lady': 'Royalty',
        'Jonkheer': 'Royalty',
        'Don': 'Royalty',
        'Dona': 'Royalty',
        'Mme': 'Mrs',
        'Capt': 'Officer',
        'Sir': 'Royalty'
    }
    
    df_copy['Title'] = df_copy['Title'].map(title_mapping)
    df_copy['Title'].fillna('Other', inplace=True)
    
    st.info(f"Estratti {df_copy['Title'].nunique()} titoli unici dal nome")
    
    return df_copy

# ----------------2. Estrazione Deck dalla Cabina (analisi posizione nave)
def extract_deck_from_cabin(df):
    """
    Estrae il deck dalla cabina per analisi posizione sulla nave
    """
    if df is None or 'Cabin' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Estrai prima lettera della cabina come deck
    df_copy['Deck'] = df_copy['Cabin'].str[0]
    
    # Sostituisci NaN con 'Unknown'
    df_copy['Deck'].fillna('Unknown', inplace=True)
    
    # Raggruppa deck rari
    deck_counts = df_copy['Deck'].value_counts()
    rare_decks = deck_counts[deck_counts < 10].index
    df_copy.loc[df_copy['Deck'].isin(rare_decks), 'Deck'] = 'Other'
    
    st.info(f"Estratti {df_copy['Deck'].nunique()} deck dalla cabina")
    
    return df_copy

# ----------------3. Calcolo Tariffa per Persona (analisi economica)
def calculate_fare_per_person(df):
    """
    Calcola tariffa per persona considerando la famiglia
    """
    if df is None or 'Fare' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Calcola dimensione famiglia se non esiste
    if 'Family_Size' not in df_copy.columns:
        df_copy['Family_Size'] = df_copy['SibSp'] + df_copy['Parch'] + 1
    
    # Calcola tariffa per persona
    df_copy['Fare_Per_Person'] = df_copy['Fare'] / df_copy['Family_Size']
    
    # Gestisci divisioni per zero o valori nulli
    df_copy['Fare_Per_Person'].fillna(df_copy['Fare_Per_Person'].median(), inplace=True)
    
    st.info("Calcolata tariffa per persona basata sulla dimensione famiglia")
    
    return df_copy

# ----------------4. Creazione Fasce Età Avanzate (da notebook gruppi età)
def create_advanced_age_groups(df):
    """
    Crea fasce di età più dettagliate
    Estende l'analisi gruppi età del notebook
    """
    if df is None or 'Age' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Fasce età dettagliate
    df_copy['Age_Group_Detailed'] = pd.cut(
        df_copy['Age'],
        bins=[0, 5, 12, 18, 25, 35, 50, 65, 100],
        labels=['Infant', 'Child', 'Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior', 'Elderly'],
        include_lowest=True
    )
    
    # Indicatori binari per fasce critiche
    df_copy['Is_Child'] = (df_copy['Age'] <= 12).astype(int)
    df_copy['Is_Senior'] = (df_copy['Age'] >= 60).astype(int)
    df_copy['Is_Adult_Prime'] = ((df_copy['Age'] >= 25) & (df_copy['Age'] <= 45)).astype(int)
    
    return df_copy

# ----------------5. Analisi Nome Avanzata (pattern nei nomi)
def analyze_name_patterns(df):
    """
    Analizza pattern nei nomi per feature engineering
    """
    if df is None or 'Name' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Lunghezza nome
    df_copy['Name_Length'] = df_copy['Name'].str.len()
    
    # Numero di parole nel nome
    df_copy['Name_Word_Count'] = df_copy['Name'].str.split().str.len()
    
    # Presenza di nickname (parentesi)
    df_copy['Has_Nickname'] = df_copy['Name'].str.contains(r'\(.*\)').astype(int)
    
    # Presenza di titoli nobiliari
    nobility_pattern = r'(Count|Countess|Lady|Sir|Don|Dona|Jonkheer)'
    df_copy['Is_Nobility'] = df_copy['Name'].str.contains(nobility_pattern, case=False).astype(int)
    
    return df_copy

# ----------------6. Feature Interazioni (combinazioni variabili)
def create_interaction_features(df):
    """
    Crea feature di interazione tra variabili
    """
    if df is None:
        return df
    
    df_copy = df.copy()
    
    # Interazione Classe-Genere
    if 'Pclass' in df_copy.columns and 'Sex' in df_copy.columns:
        df_copy['Class_Sex'] = df_copy['Pclass'].astype(str) + '_' + df_copy['Sex']
    
    # Interazione Età-Classe
    if 'Age' in df_copy.columns and 'Pclass' in df_copy.columns:
        df_copy['Age_Class_Ratio'] = df_copy['Age'] / df_copy['Pclass']
    
    # Interazione Famiglia-Classe
    if 'Family_Size' in df_copy.columns and 'Pclass' in df_copy.columns:
        df_copy['Family_Class_Score'] = df_copy['Family_Size'] * (4 - df_copy['Pclass'])  # Peso inverso classe
    
    # Indicatore viaggiatore di lusso
    if 'Fare' in df_copy.columns and 'Pclass' in df_copy.columns:
        fare_threshold = df_copy['Fare'].quantile(0.8)  # Top 20% fare
        df_copy['Is_Luxury_Traveler'] = ((df_copy['Fare'] > fare_threshold) & (df_copy['Pclass'] == 1)).astype(int)
    
    return df_copy

# ----------------7. Feature Economiche Avanzate (analisi tariffe)
def create_economic_features(df):
    """
    Crea feature economiche avanzate
    """
    if df is None or 'Fare' not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Fare relativo alla classe
    if 'Pclass' in df_copy.columns:
        class_fare_mean = df_copy.groupby('Pclass')['Fare'].transform('mean')
        df_copy['Fare_Relative_To_Class'] = df_copy['Fare'] / class_fare_mean
    
    # Fare percentile
    df_copy['Fare_Percentile'] = df_copy['Fare'].rank(pct=True)
    
    # Indicatori economici
    fare_q25 = df_copy['Fare'].quantile(0.25)
    fare_q75 = df_copy['Fare'].quantile(0.75)
    
    df_copy['Is_Economy_Fare'] = (df_copy['Fare'] <= fare_q25).astype(int)
    df_copy['Is_Premium_Fare'] = (df_copy['Fare'] >= fare_q75).astype(int)
    
    return df_copy

# ----------------8. Feature Familiari Avanzate (da notebook famiglia)
def create_advanced_family_features(df):
    """
    Crea feature familiari avanzate
    Estende analisi famiglia del notebook
    """
    if df is None:
        return df
    
    df_copy = df.copy()
    
    # Tipo famiglia dettagliato
    if 'SibSp' in df_copy.columns and 'Parch' in df_copy.columns:
        df_copy['Family_Type'] = 'Single'
        df_copy.loc[(df_copy['SibSp'] > 0) & (df_copy['Parch'] == 0), 'Family_Type'] = 'Couple'
        df_copy.loc[(df_copy['SibSp'] == 0) & (df_copy['Parch'] > 0), 'Family_Type'] = 'Parent'
        df_copy.loc[(df_copy['SibSp'] > 0) & (df_copy['Parch'] > 0), 'Family_Type'] = 'Full_Family'
        
        # Indicatori specifici
        df_copy['Has_Spouse'] = (df_copy['SibSp'] > 0).astype(int)
        df_copy['Has_Children'] = (df_copy['Parch'] > 0).astype(int)
        df_copy['Has_Siblings'] = (df_copy['SibSp'] > 1).astype(int)  # Più di 1 significa anche fratelli
        
        # Famiglia ottimale (2-4 membri)
        if 'Family_Size' in df_copy.columns:
            df_copy['Is_Optimal_Family'] = ((df_copy['Family_Size'] >= 2) & (df_copy['Family_Size'] <= 4)).astype(int)
    
    return df_copy

# ----------------9. Feature Categoriche Avanzate
def create_advanced_categorical_features(df):
    """
    Crea versioni avanzate delle feature categoriche
    """
    if df is None:
        return df
    
    df_copy = df.copy()
    
    # Encoding ordinale per classe (inverso per importanza)
    if 'Pclass' in df_copy.columns:
        df_copy['Class_Rank'] = 4 - df_copy['Pclass']  # 1st class = 3, 3rd class = 1
    
    # Binario sesso
    if 'Sex' in df_copy.columns:
        df_copy['Is_Female'] = (df_copy['Sex'] == 'female').astype(int)
    
    # Porto di imbarco con logica
    if 'Embarked' in df_copy.columns:
        # S = Southampton (più comune), C = Cherbourg (ricchi), Q = Queenstown (poveri)
        df_copy['Embarked_Wealth_Score'] = df_copy['Embarked'].map({'S': 2, 'C': 3, 'Q': 1}).fillna(2)
    
    return df_copy

# ----------------10. Feature Engineering Completo
def apply_full_feature_engineering(df):
    """
    Applica tutto il feature engineering in sequenza
    """
    if df is None:
        return df
    
    st.info("Iniziando feature engineering completo...")
    
    # Pipeline completa
    df_engineered = df.copy()
    
    # 1. Feature da nomi
    df_engineered = extract_title_from_name(df_engineered)
    df_engineered = analyze_name_patterns(df_engineered)
    
    # 2. Feature da cabine
    df_engineered = extract_deck_from_cabin(df_engineered)
    
    # 3. Feature economiche
    df_engineered = calculate_fare_per_person(df_engineered)
    df_engineered = create_economic_features(df_engineered)
    
    # 4. Feature età
    df_engineered = create_advanced_age_groups(df_engineered)
    
    # 5. Feature famiglia
    df_engineered = create_advanced_family_features(df_engineered)
    
    # 6. Feature categoriche
    df_engineered = create_advanced_categorical_features(df_engineered)
    
    # 7. Feature interazioni
    df_engineered = create_interaction_features(df_engineered)
    
    # Rimuovi colonne con troppi NaN
    for col in df_engineered.columns:
        null_pct = df_engineered[col].isnull().sum() / len(df_engineered)
        if null_pct > 0.8:  # Rimuovi se >80% NaN
            df_engineered.drop(col, axis=1, inplace=True)
            st.warning(f"Rimossa colonna {col} (troppi valori mancanti)")
    
    new_features_count = len(df_engineered.columns) - len(df.columns)
    st.success(f"Feature engineering completato! Aggiunte {new_features_count} nuove feature")
    
    return df_engineered