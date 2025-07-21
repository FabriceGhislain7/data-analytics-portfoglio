"""
src/components/univariate_charts.py
Grafici specializzati per l'analisi univariata delle variabili
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import COLOR_PALETTES, COLUMN_LABELS, VALUE_MAPPINGS

# ----------------1. Distribuzione Età Dettagliata (da notebook sezione 4.2.1)
def create_age_distribution_detailed(df):
    """
    Crea distribuzione età dettagliata con istogramma e KDE
    Da notebook sezione 4.2.1 - Age distribution
    """
    if df is None or 'Age' not in df.columns:
        return None
    
    age_data = df['Age'].dropna()
    
    fig = px.histogram(
        x=age_data,
        nbins=30,
        title="Distribuzione Età Passeggeri",
        labels={'x': 'Età (anni)', 'y': 'Frequenza'},
        marginal="box",  # Aggiunge boxplot sopra
        color_discrete_sequence=[COLOR_PALETTES['primary']]
    )
    
    # Aggiungi linea della media
    mean_age = age_data.mean()
    fig.add_vline(
        x=mean_age, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Media: {mean_age:.1f} anni"
    )
    
    fig.update_layout(height=500)
    return fig

# ----------------2. Analisi Numerica Completa (da notebook sezione 4.2.1)
def create_numerical_analysis_charts(df, variable):
    """
    Crea set completo di grafici per variabile numerica
    Basato sull'analisi del notebook sezione 4.2.1
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    # Crea subplot con 4 grafici
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Istogramma - {COLUMN_LABELS.get(variable, variable)}',
            f'Boxplot - {COLUMN_LABELS.get(variable, variable)}',
            f'Q-Q Plot - {COLUMN_LABELS.get(variable, variable)}',
            f'Distribuzione Cumulativa - {COLUMN_LABELS.get(variable, variable)}'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Istogramma
    fig.add_trace(
        go.Histogram(x=data, nbinsx=20, name="Frequenza", marker_color=COLOR_PALETTES['primary']),
        row=1, col=1
    )
    
    # 2. Boxplot
    fig.add_trace(
        go.Box(y=data, name="Distribuzione", marker_color=COLOR_PALETTES['secondary']),
        row=1, col=2
    )
    
    # 3. Q-Q Plot (approssimato)
    sorted_data = np.sort(data)
    theoretical_quantiles = np.linspace(0, 1, len(sorted_data))
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles, 
            y=sorted_data, 
            mode='markers',
            name="Q-Q Plot",
            marker_color=COLOR_PALETTES['warning']
        ),
        row=2, col=1
    )
    
    # 4. Distribuzione Cumulativa
    sorted_vals = np.sort(data)
    cumulative_prob = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    fig.add_trace(
        go.Scatter(
            x=sorted_vals, 
            y=cumulative_prob, 
            mode='lines',
            name="CDF",
            line_color=COLOR_PALETTES['success']
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

# ----------------3. Analisi Categorica (da notebook sezione 4.2.2)
def create_categorical_analysis_chart(df, variable):
    """
    Crea grafico per analisi variabile categorica
    Da notebook sezione 4.2.2 - Categorical analysis
    """
    if df is None or variable not in df.columns:
        return None
    
    value_counts = df[variable].value_counts()
    
    # Mappa i valori se disponibile
    if variable in VALUE_MAPPINGS:
        labels = [VALUE_MAPPINGS[variable].get(val, str(val)) for val in value_counts.index]
    else:
        labels = [str(val) for val in value_counts.index]
    
    # Crea subplot con grafico a barre e a torta
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Conteggi', 'Percentuali'),
        specs=[[{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Grafico a barre
    fig.add_trace(
        go.Bar(x=labels, y=value_counts.values, name="Conteggi"),
        row=1, col=1
    )
    
    # Grafico a torta
    fig.add_trace(
        go.Pie(labels=labels, values=value_counts.values, name="Percentuali"),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# ----------------4. Analisi Età Completa (da notebook sezione 4.2.1 e 4.2.2.4)
def create_age_complete_analysis(df):
    """
    Analisi completa dell'età con multiple visualizzazioni
    Combina sezioni 4.2.1 e 4.2.2.4 del notebook
    """
    if df is None or 'Age' not in df.columns:
        return None
    
    age_data = df['Age'].dropna()
    
    # Crea subplot con 3 grafici
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribuzione Età con KDE',
            'Boxplot per Rilevamento Outliers',
            'Distribuzione per Decadi',
            'Età per Genere'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Istogramma con curva KDE simulata
    fig.add_trace(
        go.Histogram(
            x=age_data, 
            nbinsx=25, 
            name="Età", 
            opacity=0.7,
            marker_color=COLOR_PALETTES['primary']
        ),
        row=1, col=1
    )
    
    # 2. Boxplot
    fig.add_trace(
        go.Box(
            y=age_data, 
            name="Distribuzione Età",
            marker_color=COLOR_PALETTES['secondary']
        ),
        row=1, col=2
    )
    
    # 3. Distribuzione per decadi
    age_decades = pd.cut(age_data, bins=range(0, 90, 10), labels=[f"{i}-{i+9}" for i in range(0, 80, 10)])
    decade_counts = age_decades.value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=decade_counts.index.astype(str), 
            y=decade_counts.values,
            name="Per Decade",
            marker_color=COLOR_PALETTES['warning']
        ),
        row=2, col=1
    )
    
    # 4. Età per genere (se disponibile)
    if 'Sex' in df.columns:
        for i, sex in enumerate(df['Sex'].unique()):
            age_by_sex = df[df['Sex'] == sex]['Age'].dropna()
            sex_label = VALUE_MAPPINGS['Sex'].get(sex, sex)
            
            fig.add_trace(
                go.Box(
                    y=age_by_sex,
                    name=sex_label,
                    marker_color=COLOR_PALETTES['gender'][i]
                ),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

# ----------------5. Confronto Trattamento Outliers (da notebook sezione 4.2.1)
def create_outlier_comparison_chart(df_original, df_processed, variable):
    """
    Confronta distribuzione prima e dopo trattamento outliers
    Da notebook sezione 4.2.1 - Outlier management
    """
    if df_original is None or df_processed is None or variable not in df_original.columns:
        return None
    
    original_data = df_original[variable].dropna()
    processed_data = df_processed[variable].dropna()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Prima del Trattamento', 'Dopo il Trattamento')
    )
    
    # Distribuzione originale
    fig.add_trace(
        go.Histogram(
            x=original_data,
            nbinsx=20,
            name="Originale",
            opacity=0.7,
            marker_color=COLOR_PALETTES['danger']
        ),
        row=1, col=1
    )
    
    # Distribuzione processata
    fig.add_trace(
        go.Histogram(
            x=processed_data,
            nbinsx=20,
            name="Processato",
            opacity=0.7,
            marker_color=COLOR_PALETTES['success']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title=f"Confronto Distribuzione {COLUMN_LABELS.get(variable, variable)}"
    )
    
    return fig

# ----------------6. Grafico Percentili (da notebook sezione 4.1.1)
def create_percentiles_chart(df, variable):
    """
    Visualizza i percentili di una variabile
    Da notebook sezione 4.1.1 - Descriptive statistics
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    # Calcola percentili
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    percentile_values = [np.percentile(data, p) for p in percentiles]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=percentile_values,
        mode='lines+markers',
        name='Percentili',
        line=dict(color=COLOR_PALETTES['primary'], width=3),
        marker=dict(size=8)
    ))
    
    # Evidenzia quartili
    quartile_indices = [2, 3, 4]  # Q1, Q2, Q3
    quartile_values = [percentile_values[i] for i in quartile_indices]
    quartile_names = ['Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)']
    
    fig.add_trace(go.Scatter(
        x=[25, 50, 75],
        y=quartile_values,
        mode='markers',
        name='Quartili',
        marker=dict(size=12, color=COLOR_PALETTES['secondary'])
    ))
    
    fig.update_layout(
        title=f"Percentili - {COLUMN_LABELS.get(variable, variable)}",
        xaxis_title="Percentile",
        yaxis_title="Valore",
        height=400
    )
    
    return fig

# ----------------7. Distribuzione con Outliers Evidenziati (da notebook sezione 4.2.1)
def create_distribution_with_outliers(df, variable):
    """
    Mostra distribuzione con outliers evidenziati
    Da notebook sezione 4.2.1 - Outlier detection
    """
    if df is None or variable not in df.columns:
        return None
    
    from src.utils.data_processor import detect_outliers_iqr
    
    data = df[variable].dropna()
    outliers, lower_bound, upper_bound = detect_outliers_iqr(data)
    
    # Separa outliers e valori normali
    normal_values = data[(data >= lower_bound) & (data <= upper_bound)]
    outlier_values = data[(data < lower_bound) | (data > upper_bound)]
    
    fig = go.Figure()
    
    # Istogramma valori normali
    fig.add_trace(go.Histogram(
        x=normal_values,
        nbinsx=20,
        name="Valori Normali",
        opacity=0.7,
        marker_color=COLOR_PALETTES['success']
    ))
    
    # Punti outliers
    if len(outlier_values) > 0:
        fig.add_trace(go.Scatter(
            x=outlier_values,
            y=[1] * len(outlier_values),  # Altezza fissa per visibilità
            mode='markers',
            name="Outliers",
            marker=dict(
                size=10,
                color=COLOR_PALETTES['danger'],
                symbol='diamond'
            )
        ))
    
    # Linee per i limiti
    fig.add_vline(x=lower_bound, line_dash="dash", line_color="orange", 
                  annotation_text=f"Limite Inf: {lower_bound:.2f}")
    fig.add_vline(x=upper_bound, line_dash="dash", line_color="orange",
                  annotation_text=f"Limite Sup: {upper_bound:.2f}")
    
    fig.update_layout(
        title=f"Distribuzione con Outliers - {COLUMN_LABELS.get(variable, variable)}",
        xaxis_title=COLUMN_LABELS.get(variable, variable),
        yaxis_title="Frequenza",
        height=400
    )
    
    return fig

# ----------------8. Analisi Valore Mancante Pattern (da notebook sezione 2.2)
def create_missing_pattern_chart(df):
    """
    Analizza pattern dei valori mancanti
    Da notebook sezione 2.2 - Missing values analysis
    """
    if df is None:
        return None
    
    # Calcola percentuali valori mancanti
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]  # Solo colonne con missing values
    
    if len(missing_pct) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=[COLUMN_LABELS.get(col, col) for col in missing_pct.index],
        x=missing_pct.values,
        orientation='h',
        marker_color=COLOR_PALETTES['warning'],
        text=[f"{val:.1f}%" for val in missing_pct.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Percentuale Valori Mancanti per Variabile",
        xaxis_title="Percentuale Missing (%)",
        yaxis_title="Variabili",
        height=400
    )
    
    return fig

# ----------------9. Distribuzione Multipla Comparativa (per confronti)
def create_multiple_distribution_comparison(df, variables):
    """
    Confronta distribuzioni di multiple variabili numeriche
    """
    if df is None or not variables:
        return None
    
    fig = go.Figure()
    
    colors = COLOR_PALETTES['seaborn_palettes'][0:len(variables)]
    
    for i, var in enumerate(variables):
        if var in df.columns:
            data = df[var].dropna()
            
            fig.add_trace(go.Histogram(
                x=data,
                name=COLUMN_LABELS.get(var, var),
                opacity=0.6,
                nbinsx=20
            ))
    
    fig.update_layout(
        title="Confronto Distribuzioni Multiple",
        xaxis_title="Valori",
        yaxis_title="Frequenza",
        barmode='overlay',
        height=400
    )
    
    return fig

# ----------------10. Summary Statistics Visual (da notebook sezione 4.1.1)
def create_summary_statistics_visual(df, variable):
    """
    Visualizzazione grafica delle statistiche descrittive
    Da notebook sezione 4.1.1
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    stats = {
        'Media': data.mean(),
        'Mediana': data.median(),
        'Moda': data.mode().iloc[0] if len(data.mode()) > 0 else data.median(),
        'Dev. Standard': data.std(),
        'Minimo': data.min(),
        'Massimo': data.max()
    }
    
    fig = go.Figure()
    
    # Grafico a barre delle statistiche
    fig.add_trace(go.Bar(
        x=list(stats.keys()),
        y=list(stats.values()),
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.2f}" for val in stats.values()],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Statistiche Descrittive - {COLUMN_LABELS.get(variable, variable)}",
        xaxis_title="Statistiche",
        yaxis_title="Valori",
        height=400
    )
    
    return fig