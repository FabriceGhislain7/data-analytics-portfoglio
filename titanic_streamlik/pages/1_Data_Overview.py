"""
pages/1_Data_Overview.py
Panoramica completa del dataset Titanic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import *
from src.utils.data_loader import load_titanic_data, get_data_summary, get_missing_values_info, check_duplicates
from src.utils.data_processor import clean_dataset_basic, detect_outliers_summary
from src.components.charts import create_missing_values_heatmap, create_data_types_chart

# ----------------1. Configurazione pagina (da config.py)
st.set_page_config(**PAGE_CONFIG)

# ----------------2. Caricamento dati (da notebook sezione 2.1 - Structure of dataset)
df_original = load_titanic_data()
if df_original is None:
    st.error("Impossibile caricare i dati")
    st.stop()

# ----------------3. Header pagina
st.title("Panoramica Dataset Titanic")
st.markdown("Analisi completa della struttura, qualità e caratteristiche del dataset")

# ----------------4. Sidebar controlli
with st.sidebar:
    st.header("Controlli Visualizzazione")
    
    # Opzioni di visualizzazione
    show_raw_data = st.checkbox("Mostra dati grezzi", value=False)
    show_cleaned_data = st.checkbox("Mostra dati puliti", value=True)
    show_statistics = st.checkbox("Mostra statistiche", value=True)
    
    # Numero righe da visualizzare
    n_rows = st.slider("Righe da visualizzare", 5, 50, 10)

# ----------------5. Informazioni generali dataset (da notebook sezione 2.1)
st.header("1. Informazioni Generali")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Numero Righe", f"{len(df_original):,}")
with col2:
    st.metric("Numero Colonne", len(df_original.columns))
with col3:
    st.metric("Dimensione Dataset", f"{df_original.memory_usage().sum() / 1024:.1f} KB")
with col4:
    duplicates_count = check_duplicates(df_original)
    st.metric("Righe Duplicate", duplicates_count)

# ----------------6. Struttura delle colonne (da notebook sezione 2.1)
st.subheader("Struttura delle Colonne")

# Informazioni sui tipi di dati
col1, col2 = st.columns(2)

with col1:
    st.write("**Informazioni colonne:**")
    column_info = pd.DataFrame({
        'Colonna': df_original.columns,
        'Tipo': df_original.dtypes.astype(str),
        'Valori Non-Null': df_original.count(),
        'Valori Null': df_original.isnull().sum(),
        'Percentuale Null': (df_original.isnull().sum() / len(df_original) * 100).round(2)
    })
    st.dataframe(column_info, use_container_width=True)

with col2:
    # ----------------7. Grafico tipi di dati (da notebook - Data type proportions)
    st.write("**Distribuzione tipi di dati:**")
    data_types = df_original.dtypes.value_counts()
    
    # Converti i tipi di dati in stringhe per evitare errori JSON
    data_types_str = data_types.astype(str)
    names_str = [str(name) for name in data_types.index]
    
    fig_types = px.pie(
        values=data_types_str.values,
        names=names_str,
        title="Distribuzione Tipi di Dati"
    )
    st.plotly_chart(fig_types, use_container_width=True)

# ----------------8. Analisi valori mancanti (da notebook sezione 2.2 - Missing values)
st.header("2. Analisi Valori Mancanti")

missing_info = get_missing_values_info(df_original)

if missing_info is not None and not missing_info.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Riepilogo Valori Mancanti")
        st.dataframe(missing_info, use_container_width=True)
    
    with col2:
        # ----------------9. Visualizzazione missing values (da notebook sezione 2.2)
        st.subheader("Visualizzazione Missing Values")
        fig_missing = px.bar(
            missing_info,
            x='Colonna',
            y='Percentuale',
            title="Percentuale Valori Mancanti per Colonna",
            color='Percentuale',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    
    # Heatmap valori mancanti
    st.subheader("Heatmap Valori Mancanti")
    fig_heatmap = create_missing_values_heatmap(df_original)
    st.plotly_chart(fig_heatmap, use_container_width=True)
else:
    st.success("Nessun valore mancante rilevato nel dataset!")

# ----------------10. Data Cleaning Preview (da notebook sezione 3 - Data Cleaning)
st.header("3. Anteprima Pulizia Dati")

# Applica pulizia base
df_cleaned = clean_dataset_basic(df_original)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Originale")
    st.write(f"Righe: {len(df_original)}")
    st.write(f"Colonne: {len(df_original.columns)}")
    st.write(f"Valori mancanti: {df_original.isnull().sum().sum()}")

with col2:
    st.subheader("Dataset Pulito")
    st.write(f"Righe: {len(df_cleaned)}")
    st.write(f"Colonne: {len(df_cleaned.columns)}")
    st.write(f"Valori mancanti: {df_cleaned.isnull().sum().sum()}")

# ----------------11. Visualizzazione dati (da notebook sezione 2.1)
if show_raw_data:
    st.header("4. Dati Grezzi")
    st.subheader("Prime righe del dataset originale")
    st.dataframe(df_original.head(n_rows), use_container_width=True)
    
    st.subheader("Ultime righe del dataset")
    st.dataframe(df_original.tail(n_rows), use_container_width=True)

if show_cleaned_data:
    st.header("5. Dati Puliti")
    st.subheader("Dataset dopo pulizia base")
    st.dataframe(df_cleaned.head(n_rows), use_container_width=True)

# ----------------12. Statistiche descrittive (da notebook sezione 4.1.1 - Descriptive Statistics)
if show_statistics:
    st.header("6. Statistiche Descrittive")
    
    # Statistiche per variabili numeriche
    st.subheader("Variabili Numeriche")
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df_cleaned[numeric_cols].describe(), use_container_width=True)
    
    # Statistiche per variabili categoriche
    st.subheader("Variabili Categoriche")
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'Name']  # Escludi Name
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            with st.expander(f"Analisi {COLUMN_LABELS.get(col, col)}"):
                value_counts = df_cleaned[col].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Conteggi:**")
                    st.dataframe(value_counts.reset_index())
                
                with col2:
                    st.write("**Distribuzione:**")
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribuzione {COLUMN_LABELS.get(col, col)}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ----------------13. Rilevamento outliers (da notebook sezione 4.1.1 - Outlier detection)
st.header("7. Rilevamento Outliers")

outliers_summary = detect_outliers_summary(df_cleaned)
if outliers_summary is not None:
    st.subheader("Riepilogo Outliers per Variabile")
    st.dataframe(outliers_summary, use_container_width=True)
    
    # Boxplot per visualizzare outliers
    st.subheader("Visualizzazione Outliers")
    numeric_cols_for_outliers = [col for col in numeric_cols if col not in ['PassengerId']]
    
    if len(numeric_cols_for_outliers) > 0:
        selected_col = st.selectbox(
            "Seleziona variabile per analisi outliers:",
            numeric_cols_for_outliers,
            format_func=lambda x: COLUMN_LABELS.get(x, x)
        )
        
        fig_box = px.box(
            df_cleaned,
            y=selected_col,
            title=f"Boxplot per {COLUMN_LABELS.get(selected_col, selected_col)}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ----------------14. Summary finale
st.header("8. Riepilogo Qualità Dati")

quality_metrics = {
    "Completezza": f"{((df_cleaned.count().sum() / (len(df_cleaned) * len(df_cleaned.columns))) * 100):.1f}%",
    "Duplicati": f"{duplicates_count} righe",
    "Outliers rilevati": f"{outliers_summary['Outliers_Count'].sum() if outliers_summary is not None else 0} valori",
    "Variabili numeriche": f"{len(numeric_cols)} colonne",
    "Variabili categoriche": f"{len(categorical_cols)} colonne"
}

col1, col2, col3 = st.columns(3)
metrics_items = list(quality_metrics.items())

for i, (metric, value) in enumerate(metrics_items):
    with [col1, col2, col3][i % 3]:
        st.metric(metric, value)

# ----------------15. Note metodologiche (da notebook)
with st.expander("Note Metodologiche"):
    st.markdown("""
    **Metodologia di analisi basata su:**
    
    - **Sezione 2.1 del notebook**: Struttura iniziale del dataset
    - **Sezione 2.2 del notebook**: Analisi valori mancanti
    - **Sezione 2.3 del notebook**: Controllo duplicati
    - **Sezione 3 del notebook**: Metodi di pulizia dati
    - **Sezione 4.1.1 del notebook**: Statistiche descrittive
    
    **Trasformazioni applicate:**
    - Rimozione colonna 'Cabin' (77% valori mancanti)
    - Rimozione righe duplicate
    - Rilevamento outliers con metodo IQR
    """)