"""
pages/2_Univariate_Analysis.py
Analisi univariata delle singole variabili del dataset Titanic
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import *
from src.utils.data_loader import load_titanic_data
from src.utils.data_processor import clean_dataset_basic, detect_outliers_iqr, handle_outliers
from src.components.charts import create_age_distribution_chart, create_survival_overview_chart
from src.components.univariate_charts import *

# ----------------1. Configurazione pagina (da config.py)
st.set_page_config(**PAGE_CONFIG)

# ----------------2. Caricamento e pulizia dati (da notebook sezione 2.1 e 3)
df_original = load_titanic_data()
if df_original is None:
    st.error("Impossibile caricare i dati")
    st.stop()

df = clean_dataset_basic(df_original)

# ----------------3. Header pagina
st.title("Analisi Univariata - Variabili Singole")
st.markdown("Esplorazione dettagliata delle caratteristiche individuali di ogni variabile")

# ----------------4. Sidebar controlli
with st.sidebar:
    st.header("Controlli Analisi")
    
    # Selezione variabile da analizzare
    numeric_variables = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if col not in ['PassengerId']]
    categorical_variables = [col for col in df.select_dtypes(include=['object']).columns 
                           if col not in ['Name', 'Ticket']]
    
    analysis_type = st.selectbox(
        "Tipo di Analisi:",
        ["Panoramica Generale", "Variabili Numeriche", "Variabili Categoriche", "Focus su Età", "Focus su Sopravvivenza"]
    )
    
    # Opzioni visualizzazione
    show_statistics = st.checkbox("Mostra statistiche dettagliate", value=True)
    show_outliers = st.checkbox("Analisi outliers", value=True)

# ----------------5. Panoramica Generale (da notebook sezione 4.1.1)
if analysis_type == "Panoramica Generale":
    st.header("1. Panoramica Generale Dataset")
    
    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Passeggeri Totali", f"{len(df):,}")
    with col2:
        st.metric("Variabili Numeriche", len(numeric_variables))
    with col3:
        st.metric("Variabili Categoriche", len(categorical_variables))
    with col4:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Tasso Sopravvivenza", f"{survival_rate:.1f}%")
    
    # ----------------6. Distribuzioni principali
    st.subheader("Distribuzioni Principali")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuzione età
        fig_age = create_age_distribution_detailed(df)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Distribuzione sopravvivenza
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)
    
    # ----------------7. Summary statistiche tutte variabili numeriche
    if show_statistics:
        st.subheader("Statistiche Descrittive Variabili Numeriche")
        stats_df = df[numeric_variables].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

# ----------------8. Analisi Variabili Numeriche (da notebook sezione 4.2.1)
elif analysis_type == "Variabili Numeriche":
    st.header("2. Analisi Variabili Numeriche")
    
    # Selezione variabile specifica
    selected_var = st.selectbox(
        "Seleziona variabile numerica:",
        numeric_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    
    if selected_var:
        st.subheader(f"Analisi dettagliata: {COLUMN_LABELS.get(selected_var, selected_var)}")
        
        # ----------------9. Statistiche base (da notebook sezione 4.2.1 - Age analysis)
        col1, col2, col3, col4 = st.columns(4)
        
        var_data = df[selected_var].dropna()
        
        with col1:
            st.metric("Media", f"{var_data.mean():.2f}")
        with col2:
            st.metric("Mediana", f"{var_data.median():.2f}")
        with col3:
            st.metric("Deviazione Standard", f"{var_data.std():.2f}")
        with col4:
            st.metric("Valori Unici", f"{var_data.nunique()}")
        
        # ----------------10. Visualizzazioni multiple (da notebook sezione 4.2.1)
        fig = create_numerical_analysis_charts(df, selected_var)
        st.plotly_chart(fig, use_container_width=True)
        
        # ----------------11. Analisi outliers (da notebook sezione 4.2.1 - Outlier detection)
        if show_outliers:
            st.subheader("Analisi Outliers")
            
            outliers, lower_bound, upper_bound = detect_outliers_iqr(var_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outliers Rilevati", len(outliers))
            with col2:
                st.metric("Limite Inferiore", f"{lower_bound:.2f}")
            with col3:
                st.metric("Limite Superiore", f"{upper_bound:.2f}")
            
            if len(outliers) > 0:
                st.write("**Valori outliers:**")
                outliers_df = pd.DataFrame({
                    'Indice': outliers.index,
                    'Valore': outliers.values
                }).sort_values('Valore')
                st.dataframe(outliers_df, use_container_width=True)

# ----------------12. Analisi Variabili Categoriche (da notebook sezione 4.2.2)
elif analysis_type == "Variabili Categoriche":
    st.header("3. Analisi Variabili Categoriche")
    
    # Selezione variabile categorica
    selected_cat_var = st.selectbox(
        "Seleziona variabile categorica:",
        categorical_variables,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    
    if selected_cat_var:
        st.subheader(f"Analisi dettagliata: {COLUMN_LABELS.get(selected_cat_var, selected_cat_var)}")
        
        # ----------------13. Statistiche categoriche (da notebook sezione 4.2.2)
        value_counts = df[selected_cat_var].value_counts()
        value_props = df[selected_cat_var].value_counts(normalize=True) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Categorie Uniche", len(value_counts))
        with col2:
            st.metric("Categoria Più Frequente", value_counts.index[0])
        with col3:
            st.metric("Frequenza Massima", f"{value_props.iloc[0]:.1f}%")
        
        # ----------------14. Tabella frequenze
        freq_table = pd.DataFrame({
            'Categoria': value_counts.index,
            'Conteggio': value_counts.values,
            'Percentuale': value_props.values.round(1)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tabella Frequenze:**")
            st.dataframe(freq_table, use_container_width=True)
        
        with col2:
            # ----------------15. Visualizzazione categorica
            fig_cat = create_categorical_analysis_chart(df, selected_cat_var)
            st.plotly_chart(fig_cat, use_container_width=True)

# ----------------16. Focus su Età (da notebook sezione 4.2.1 - Age Analysis completa)
elif analysis_type == "Focus su Età":
    st.header("4. Analisi Approfondita dell'Età")
    
    # ----------------17. Statistiche età (da notebook sezione 4.2.1)
    age_data = df['Age'].dropna()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Età Media", f"{age_data.mean():.1f} anni")
    with col2:
        st.metric("Età Mediana", f"{age_data.median():.1f} anni")
    with col3:
        st.metric("Età Minima", f"{age_data.min():.1f} anni")
    with col4:
        st.metric("Età Massima", f"{age_data.max():.1f} anni")
    with col5:
        st.metric("Deviazione Standard", f"{age_data.std():.1f} anni")
    
    # ----------------18. Visualizzazioni età multiple (da notebook sezione 4.2.1)
    fig_age_complete = create_age_complete_analysis(df)
    st.plotly_chart(fig_age_complete, use_container_width=True)
    
    # ----------------19. Gruppi di età (da notebook sezione 4.2.2.4)
    st.subheader("Analisi per Gruppi di Età")
    
    # Crea gruppi età
    df['Age_Group'] = pd.cut(df['Age'], 
                            bins=[0, 12, 25, 40, 100], 
                            labels=['Bambini (0-12)', 'Giovani (13-25)', 'Adulti (26-40)', 'Anziani (41+)'])
    
    age_group_stats = df['Age_Group'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribuzione per Gruppi:**")
        age_groups_df = pd.DataFrame({
            'Gruppo': age_group_stats.index,
            'Conteggio': age_group_stats.values,
            'Percentuale': (age_group_stats.values / len(df) * 100).round(1)
        })
        st.dataframe(age_groups_df, use_container_width=True)
    
    with col2:
        fig_age_groups = px.bar(
            x=age_group_stats.index,
            y=age_group_stats.values,
            title="Distribuzione per Gruppi di Età",
            labels={'x': 'Gruppo Età', 'y': 'Numero Passeggeri'}
        )
        st.plotly_chart(fig_age_groups, use_container_width=True)
    
    # ----------------20. Trattamento outliers età (da notebook sezione 4.2.1)
    if show_outliers:
        st.subheader("Gestione Outliers Età")
        
        outliers, lower_bound, upper_bound = detect_outliers_iqr(age_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Outliers rilevati:** {len(outliers)}")
            st.write(f"**Limite inferiore:** {lower_bound:.1f} anni")
            st.write(f"**Limite superiore:** {upper_bound:.1f} anni")
            
            if len(outliers) > 0:
                outliers_list = sorted(outliers.values)
                st.write("**Età outliers:**", outliers_list)
        
        with col2:
            # Opzioni trattamento outliers
            outlier_method = st.selectbox(
                "Metodo trattamento outliers:",
                ["Nessuno", "Rimozione", "Sostituzione con mediana", "Clipping ai limiti"]
            )
            
            if outlier_method != "Nessuno":
                if outlier_method == "Rimozione":
                    df_processed = handle_outliers(df, method='remove', columns=['Age'])
                elif outlier_method == "Sostituzione con mediana":
                    df_processed = handle_outliers(df, method='replace_median', columns=['Age'])
                else:  # Clipping
                    df_processed = handle_outliers(df, method='clip', columns=['Age'])
                
                st.write(f"**Dataset dopo trattamento:** {len(df_processed)} righe")
                
                # Confronto distribuzioni
                fig_comparison = create_outlier_comparison_chart(df, df_processed, 'Age')
                st.plotly_chart(fig_comparison, use_container_width=True)

# ----------------21. Focus su Sopravvivenza (da notebook sezione 4.2.2 - Survival Analysis)
elif analysis_type == "Focus su Sopravvivenza":
    st.header("5. Analisi Approfondita della Sopravvivenza")
    
    # ----------------22. Statistiche sopravvivenza (da notebook sezione 4.2.2)
    survival_stats = df['Survived'].value_counts()
    survival_props = df['Survived'].value_counts(normalize=True) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Totale Passeggeri", len(df))
    with col2:
        st.metric("Sopravvissuti", int(survival_stats[1]))
    with col3:
        st.metric("Morti", int(survival_stats[0]))
    with col4:
        st.metric("Tasso Sopravvivenza", f"{survival_props[1]:.1f}%")
    
    # ----------------23. Visualizzazioni sopravvivenza (da notebook sezione 4.2.2)
    col1, col2 = st.columns(2)
    
    with col1:
        # Grafico a torta
        fig_pie = create_survival_overview_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Grafico a barre
        fig_bar = px.bar(
            x=['Non Sopravvissuti', 'Sopravvissuti'],
            y=[survival_stats[0], survival_stats[1]],
            title="Conteggio Sopravvivenza",
            color=['Non Sopravvissuti', 'Sopravvissuti'],
            color_discrete_map={'Non Sopravvissuti': COLOR_PALETTES['survival'][0], 
                               'Sopravvissuti': COLOR_PALETTES['survival'][1]}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # ----------------24. Analisi distribuzione sopravvivenza
    st.subheader("Distribuzione Sopravvivenza nel Dataset")
    
    survival_interpretation = f"""
    **Interpretazione dei risultati:**
    
    - **Tasso di mortalità**: {survival_props[0]:.1f}% dei passeggeri non è sopravvissuto
    - **Tasso di sopravvivenza**: {survival_props[1]:.1f}% dei passeggeri è sopravvissuto
    - **Differenza**: La probabilità di morte era {survival_props[0]/survival_props[1]:.1f} volte superiore alla sopravvivenza
    
    Questo dato conferma la gravità del disastro del Titanic, dove la maggioranza dei passeggeri ha perso la vita.
    """
    
    st.markdown(survival_interpretation)

# ----------------25. Note metodologiche
with st.expander("Note Metodologiche"):
    st.markdown("""
    **Metodologia di analisi basata su:**
    
    - **Sezione 4.1.1 del notebook**: Statistiche descrittive generali
    - **Sezione 4.2.1 del notebook**: Analisi univariata dell'età
    - **Sezione 4.2.2 del notebook**: Analisi della sopravvivenza
    - **Sezione 4.1.1 del notebook**: Rilevamento e gestione outliers
    
    **Tecniche utilizzate:**
    - Statistiche descrittive (media, mediana, deviazione standard)
    - Visualizzazioni multiple (istogrammi, boxplot, grafici a torta)
    - Rilevamento outliers con metodo IQR
    - Analisi delle frequenze per variabili categoriche
    - Creazione di gruppi di età per analisi segmentata
    
    **Metriche calcolate:**
    - Percentili (25%, 50%, 75%)
    - Limiti per outliers (Q1-1.5*IQR, Q3+1.5*IQR)
    - Distribuzioni di frequenza
    - Tassi di sopravvivenza
    """)