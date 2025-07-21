"""
src/components/charts.py
Componenti per la creazione di grafici e visualizzazioni
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from src.config import COLOR_PALETTES, VALUE_MAPPINGS

# ----------------1. Grafico Sopravvivenza Generale (da notebook sezione 4.2.2 - Survival Analysis)
def create_survival_overview_chart(df):
    """
    Crea grafico a torta della sopravvivenza generale
    Basato sull'analisi del notebook sezione 4.2.2
    """
    if df is None:
        return None
    
    # Calcola conteggi sopravvivenza
    survival_counts = df['Survived'].value_counts().sort_index()
    
    # Mappa i valori alle etichette
    labels = [VALUE_MAPPINGS['Survived'][val] for val in survival_counts.index]
    values = survival_counts.values
    colors = COLOR_PALETTES['survival']
    
    # Crea grafico a torta
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Distribuzione Sopravvivenza",
        showlegend=True,
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------2. Distribuzione per Classe (da notebook sezione 4.2.2.1 - Tickets by Class)
def create_class_distribution_chart(df):
    """
    Crea grafico distribuzione passeggeri per classe
    Da notebook sezione 4.2.2.1
    """
    if df is None:
        return None
    
    # Conta passeggeri per classe
    class_counts = df['Pclass'].value_counts().sort_index()
    
    # Mappa le classi alle etichette
    labels = [VALUE_MAPPINGS['Pclass'][val] for val in class_counts.index]
    values = class_counts.values
    colors = COLOR_PALETTES['class']
    
    # Crea grafico a barre
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Distribuzione Passeggeri per Classe",
        xaxis_title="Classe",
        yaxis_title="Numero Passeggeri",
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------3. Sopravvivenza per Classe (da notebook sezione 4.2.2.2 - Survival by Class)
def create_survival_by_class_chart(df):
    """
    Grafico sopravvivenza per classe passeggeri
    Da notebook sezione 4.2.2.2
    """
    if df is None:
        return None
    
    # Calcola sopravvivenza per classe
    survival_by_class = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    survival_by_class.columns = ['Morti', 'Sopravvissuti']
    
    # Mappa le classi
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in survival_by_class.index]
    
    fig = go.Figure()
    
    # Aggiungi barre per morti e sopravvissuti
    fig.add_trace(go.Bar(
        name='Morti',
        x=class_labels,
        y=survival_by_class['Morti'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Sopravvissuti',
        x=class_labels,
        y=survival_by_class['Sopravvissuti'],
        marker_color=COLOR_PALETTES['survival'][1]
    ))
    
    fig.update_layout(
        title='Sopravvivenza per Classe',
        xaxis_title='Classe',
        yaxis_title='Numero Passeggeri',
        barmode='stack',
        height=400
    )
    
    return fig

# ----------------4. Distribuzione Eta (da notebook sezione 4.2.1 - Age Analysis)
def create_age_distribution_chart(df):
    """
    Istogramma distribuzione eta
    Da notebook sezione 4.2.1
    """
    if df is None:
        return None
    
    # Rimuovi valori mancanti per l'eta
    age_data = df['Age'].dropna()
    
    fig = px.histogram(
        x=age_data,
        nbins=20,
        title="Distribuzione Eta Passeggeri",
        labels={'x': 'Eta (anni)', 'y': 'Frequenza'},
        color_discrete_sequence=[COLOR_PALETTES['primary']]
    )
    
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------5. Sopravvivenza per Genere (da notebook sezione 4.2.2.3 - Survival by Gender)
def create_survival_by_gender_chart(df):
    """
    Grafico sopravvivenza per genere
    Da notebook sezione 4.2.2.3
    """
    if df is None:
        return None
    
    # Calcola percentuali sopravvivenza per genere
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    
    # Mappa i generi
    gender_labels = [VALUE_MAPPINGS['Sex'][idx] for idx in gender_survival.index]
    colors = COLOR_PALETTES['gender']
    
    fig = go.Figure(data=[go.Bar(
        x=gender_labels,
        y=gender_survival.values,
        marker=dict(color=colors),
        text=[f"{val:.1f}%" for val in gender_survival.values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Tasso di Sopravvivenza per Genere",
        xaxis_title="Genere",
        yaxis_title="Tasso Sopravvivenza (%)",
        height=400,
        margin=dict(t=50, b=0, l=0, r=0)
    )
    
    return fig

# ----------------6. Grafico Combinato Dashboard (sintesi per homepage)
def create_dashboard_summary_chart(df):
    """
    Grafico riassuntivo per dashboard principale
    Combina insights chiave da multiple sezioni notebook
    """
    if df is None:
        return None
    
    # Sottografici: Classe vs Genere vs Sopravvivenza
    survival_summary = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
    
    # Mappa etichette
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in survival_summary.index]
    
    fig = go.Figure()
    
    # Aggiungi tracce per ogni genere
    for i, gender in enumerate(survival_summary.columns):
        gender_label = VALUE_MAPPINGS['Sex'][gender]
        fig.add_trace(go.Bar(
            name=gender_label,
            x=class_labels,
            y=survival_summary[gender] * 100,
            marker_color=COLOR_PALETTES['gender'][i]
        ))
    
    fig.update_layout(
        title='Tasso Sopravvivenza per Classe e Genere',
        xaxis_title='Classe',
        yaxis_title='Tasso Sopravvivenza (%)',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------7. Heatmap Valori Mancanti (da notebook sezione 2.2)
def create_missing_values_heatmap(df):
    """
    Crea heatmap dei valori mancanti
    Da notebook sezione 2.2 - Missing values visualization
    """
    if df is None:
        return None
    
    # Calcola missing values per riga
    missing_data = df.isnull()
    
    # Se ci sono troppi dati, campiona le righe
    if len(df) > 100:
        missing_data = missing_data.sample(n=100, random_state=42)
    
    fig = go.Figure(data=go.Heatmap(
        z=missing_data.values.astype(int),
        x=missing_data.columns,
        y=list(range(len(missing_data))),
        colorscale=[[0, 'lightblue'], [1, 'red']],
        showscale=True,
        colorbar=dict(title="Missing Values", tickvals=[0, 1], ticktext=["Present", "Missing"])
    ))
    
    fig.update_layout(
        title="Heatmap Valori Mancanti (campione 100 righe)",
        xaxis_title="Colonne",
        yaxis_title="Righe (campione)",
        height=400
    )
    
    return fig

# ----------------8. Grafico Tipi di Dati (da notebook - data types analysis)
def create_data_types_chart(df):
    """
    Visualizza distribuzione tipi di dati
    """
    if df is None:
        return None
    
    data_types = df.dtypes.value_counts()
    
    # Converti in stringhe per evitare errori di serializzazione JSON
    values = data_types.values.astype(int)
    names = [str(name) for name in data_types.index]
    
    fig = px.pie(
        values=values,
        names=names,
        title="Distribuzione Tipi di Dati",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(height=400)
    return fig