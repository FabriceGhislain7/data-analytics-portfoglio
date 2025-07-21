"""
src/components/metrics.py
Componenti per visualizzare metriche e KPI del dashboard
"""

import streamlit as st
import pandas as pd
from src.config import format_percentage, VALUE_MAPPINGS

# ----------------1. Metriche Overview Principali (da notebook sezione 4.2.2 - Survival Analysis)
def create_overview_metrics(df):
    """
    Crea le metriche principali per la dashboard
    Basato sull'analisi di sopravvivenza del notebook sezione 4.2.2
    """
    if df is None:
        return
    
    # Calcoli base della sopravvivenza
    total_passengers = len(df)
    survived = df['Survived'].sum()
    died = total_passengers - survived
    survival_rate = (survived / total_passengers) * 100
    
    # Layout a 4 colonne per le metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Passeggeri Totali",
            value=f"{total_passengers:,}",
            help="Numero totale di passeggeri nel dataset"
        )
    
    with col2:
        st.metric(
            label="Sopravvissuti",
            value=f"{survived:,}",
            delta=f"{format_percentage(survival_rate)}",
            delta_color="normal",
            help="Numero e percentuale di sopravvissuti"
        )
    
    with col3:
        st.metric(
            label="Morti",
            value=f"{died:,}",
            delta=f"{format_percentage(100-survival_rate)}",
            delta_color="inverse",
            help="Numero e percentuale di morti"
        )
    
    with col4:
        # ----------------2. Eta Media (da notebook sezione 4.2.1 - Age Analysis)
        avg_age = df['Age'].mean()
        st.metric(
            label="Eta Media",
            value=f"{avg_age:.1f} anni",
            help="Eta media dei passeggeri"
        )

# ----------------3. Metriche per Classe (da notebook sezione 4.2.2.2 - Survival by Class)
def create_class_metrics(df):
    """
    Metriche di sopravvivenza per classe passeggeri
    Da notebook sezione 4.2.2.2
    """
    if df is None:
        return
    
    # Calcola sopravvivenza per classe
    class_survival = df.groupby('Pclass').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    class_survival.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza']
    class_survival = class_survival.reset_index()
    
    st.subheader("Sopravvivenza per Classe")
    
    for _, row in class_survival.iterrows():
        pclass = int(row['Pclass'])
        class_name = VALUE_MAPPINGS['Pclass'][pclass]
        survival_rate = row['Tasso_Sopravvivenza'] * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{class_name} - Totale", f"{int(row['Totale'])}")
        with col2:
            st.metric(f"{class_name} - Sopravvissuti", f"{int(row['Sopravvissuti'])}")
        with col3:
            st.metric(f"{class_name} - Tasso", f"{survival_rate:.1f}%")

# ----------------4. Metriche per Genere (da notebook sezione 4.2.2.3 - Survival by Gender)
def create_gender_metrics(df):
    """
    Metriche di sopravvivenza per genere
    Da notebook sezione 4.2.2.3
    """
    if df is None:
        return
    
    # Calcola sopravvivenza per genere
    gender_survival = df.groupby('Sex').agg({
        'Survived': ['sum', 'count', 'mean']
    }).round(3)
    
    gender_survival.columns = ['Sopravvissuti', 'Totale', 'Tasso_Sopravvivenza']
    gender_survival = gender_survival.reset_index()
    
    st.subheader("Sopravvivenza per Genere")
    
    col1, col2 = st.columns(2)
    
    for i, (_, row) in enumerate(gender_survival.iterrows()):
        gender = row['Sex']
        gender_name = VALUE_MAPPINGS['Sex'][gender]
        survival_rate = row['Tasso_Sopravvivenza'] * 100
        
        with col1 if i == 0 else col2:
            st.metric(
                label=f"{gender_name}",
                value=f"{int(row['Sopravvissuti'])}/{int(row['Totale'])}",
                delta=f"{survival_rate:.1f}%",
                help=f"Sopravvissuti/Totale per {gender_name.lower()}"
            )

# ----------------5. Metriche Statistiche Base (da notebook sezione 4.1.1 - Descriptive Statistics)
def create_statistical_metrics(df):
    """
    Mostra statistiche descrittive principali
    Da notebook sezione 4.1.1
    """
    if df is None:
        return
    
    st.subheader("Statistiche Descrittive")
    
    # Variabili numeriche principali
    age_stats = df['Age'].describe()
    fare_stats = df['Fare'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Eta Min", f"{age_stats['min']:.0f} anni")
        st.metric("Eta Max", f"{age_stats['max']:.0f} anni")
    
    with col2:
        st.metric("Eta Mediana", f"{age_stats['50%']:.0f} anni")
        st.metric("Eta Media", f"{age_stats['mean']:.1f} anni")
    
    with col3:
        st.metric("Prezzo Min", f"${fare_stats['min']:.0f}")
        st.metric("Prezzo Max", f"${fare_stats['max']:.0f}")
    
    with col4:
        st.metric("Prezzo Mediano", f"${fare_stats['50%']:.0f}")
        st.metric("Prezzo Medio", f"${fare_stats['mean']:.0f}")