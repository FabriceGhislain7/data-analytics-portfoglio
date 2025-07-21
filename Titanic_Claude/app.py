# File principale dell'applicazione Streamlit
"""
app.py - File principale dell'applicazione Streamlit
Titanic Survival Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import *
from src.utils.data_loader import load_titanic_data, get_data_summary
from src.components.metrics import create_overview_metrics
from src.components.charts import create_survival_overview_chart, create_class_distribution_chart

# ----------------1. Configurazione pagina principale (da config.py)
st.set_page_config(**PAGE_CONFIG)

def main():
    """Funzione principale dell'applicazione"""
    
    # ----------------2. Caricamento dati (da notebook sezione 2.1 - Dataset Loading)
    df = load_titanic_data()
    if df is None:
        st.error("Errore nel caricamento dei dati")
        return
    
    # ----------------3. Header principale (da notebook Project Overview)
    st.title(APP_TEXTS['main_title'])
    st.markdown(APP_TEXTS['subtitle'])
    
    # ----------------4. Sidebar informazioni
    with st.sidebar:
        st.header("Informazioni Dataset")
        st.info(f"Passeggeri totali: {len(df)}")
        st.info(f"Variabili: {len(df.columns)}")
        st.info(APP_TEXTS['data_source'])
    
    # ----------------5. Overview Metrics (da notebook sezione 4.2.2 - Survival Analysis)
    st.subheader("Panoramica Generale")
    create_overview_metrics(df)
    
    # ----------------6. Visualizzazioni principali dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        # ----------------7. Grafico sopravvivenza generale (da notebook sezione 4.2.2)
        st.subheader("Tasso di Sopravvivenza")
        fig_survival = create_survival_overview_chart(df)
        st.plotly_chart(fig_survival, use_container_width=True)
    
    with col2:
        # ----------------8. Distribuzione per classe (da notebook sezione 4.2.2.1)
        st.subheader("Distribuzione per Classe")
        fig_class = create_class_distribution_chart(df)
        st.plotly_chart(fig_class, use_container_width=True)
    
    # ----------------9. Informazioni dataset (da notebook sezione 2.1)
    with st.expander("Dettagli Dataset"):
        summary = get_data_summary(df)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Righe", summary['rows'])
            st.metric("Sopravvissuti", summary['survived'])
        
        with col2:
            st.metric("Colonne", summary['columns'])
            st.metric("Morti", summary['died'])
        
        with col3:
            st.metric("Valori Mancanti", summary['missing_values'])
            st.metric("Tasso Sopravvivenza", f"{summary['survival_rate']:.1f}%")
    
    # ----------------10. Footer (da config.py)
    st.markdown(APP_TEXTS['footer'])

if __name__ == "__main__":
    main()