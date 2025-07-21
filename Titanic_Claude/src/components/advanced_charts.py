"""
src/components/advanced_charts.py
Grafici specializzati per analisi avanzate e feature engineering
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import COLOR_PALETTES, COLUMN_LABELS, VALUE_MAPPINGS

# ----------------1. Matrice Correlazione Avanzata (da notebook sezione 4.1.2 estesa)
def create_correlation_matrix(df, method='pearson'):
    """
    Crea matrice di correlazione con metodo specificato
    Estende notebook sezione 4.1.2
    """
    if df is None:
        return None
    
    # Seleziona solo variabili numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'PassengerId']
    
    if len(numeric_cols) < 2:
        return None
    
    # Calcola correlazione
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Maschera triangolo superiore per evitare ridondanza
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_masked = corr_matrix.mask(mask)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=[COLUMN_LABELS.get(col, col) for col in corr_matrix.columns],
        y=[COLUMN_LABELS.get(col, col) for col in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        # Rimuovi i numeri dal grafico per migliore leggibilità
        text=None,
        hovertemplate='%{x} vs %{y}<br>Correlazione: %{z:.3f}<extra></extra>',
        colorbar=dict(title=f"Correlazione {method.title()}")
    ))
    
    fig.update_layout(
        title=f"Matrice Correlazione {method.title()}",
        height=500,
        # Migliora la leggibilità
        xaxis_tickangle=-45,
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig

# ----------------2. Correlazioni con Target
def create_target_correlation_chart(correlations_df):
    """
    Grafico correlazioni con variabile target
    """
    if correlations_df is None:
        return None
    
    # Prendi top 10 correlazioni (assolute)
    top_corr = correlations_df.head(10)
    
    fig = go.Figure(data=[go.Bar(
        y=top_corr.index,
        x=top_corr.values,
        orientation='h',
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.3f}" for val in top_corr.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Top Correlazioni con Sopravvivenza",
        xaxis_title="Correlazione",
        yaxis_title="Variabili",
        height=400
    )
    
    return fig

# ----------------3. Correlazioni per Categoria
def create_correlation_by_category(df, category_col):
    """
    Matrice correlazione separata per categorie
    """
    if df is None or category_col not in df.columns:
        return None
    
    categories = df[category_col].unique()
    
    # Crea subplot per ogni categoria
    fig = make_subplots(
        rows=1, cols=len(categories),
        subplot_titles=[VALUE_MAPPINGS.get(category_col, {}).get(cat, str(cat)) for cat in categories]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['PassengerId']]
    
    for i, category in enumerate(categories):
        subset = df[df[category_col] == category]
        if len(subset) > 5:  # Solo se abbiamo abbastanza dati
            corr_matrix = subset[numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=numeric_cols,
                    y=numeric_cols,
                    colorscale='RdBu',
                    zmid=0,
                    showscale=(i == len(categories)-1)  # Solo ultima scala
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title=f"Correlazioni per {COLUMN_LABELS.get(category_col, category_col)}",
        height=400
    )
    
    return fig

# ----------------4. Analisi Titolo (da feature engineering nome)
def create_title_survival_analysis(df):
    """
    Analisi sopravvivenza per titolo estratto dal nome
    """
    if df is None or 'Title' not in df.columns:
        return None
    
    title_survival = df.groupby('Title')['Survived'].mean() * 100
    title_counts = df['Title'].value_counts()
    
    # Filtra titoli con almeno 5 occorrenze
    common_titles = title_counts[title_counts >= 5].index
    title_survival_filtered = title_survival[common_titles]
    
    fig = go.Figure(data=[go.Bar(
        x=title_survival_filtered.index,
        y=title_survival_filtered.values,
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.1f}%" for val in title_survival_filtered.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Sopravvivenza per Titolo",
        xaxis_title="Titolo",
        yaxis_title="Tasso Sopravvivenza (%)",
        height=400
    )
    
    return fig

# ----------------5. Analisi Deck (da feature engineering cabina)
def create_deck_survival_analysis(df):
    """
    Analisi sopravvivenza per deck estratto dalla cabina
    """
    if df is None or 'Deck' not in df.columns:
        return None
    
    deck_survival = df.groupby('Deck')['Survived'].mean() * 100
    deck_counts = df['Deck'].value_counts()
    
    # Filtra deck con almeno 3 occorrenze
    common_decks = deck_counts[deck_counts >= 3].index
    deck_survival_filtered = deck_survival[common_decks]
    
    fig = go.Figure(data=[go.Bar(
        x=deck_survival_filtered.index,
        y=deck_survival_filtered.values,
        marker_color=COLOR_PALETTES['secondary'],
        text=[f"{val:.1f}%" for val in deck_survival_filtered.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Sopravvivenza per Deck",
        xaxis_title="Deck",
        yaxis_title="Tasso Sopravvivenza (%)",
        height=400
    )
    
    return fig

# ----------------6. Feature Importance Chart
def create_feature_importance_chart(importance_df):
    """
    Grafico importanza features
    """
    if importance_df is None:
        return None
    
    # Top 15 features
    top_features = importance_df.head(15)
    
    fig = go.Figure(data=[go.Bar(
        y=top_features['Feature'],
        x=top_features['Importance'],
        orientation='h',
        marker_color=COLOR_PALETTES['success'],
        text=[f"{val:.3f}" for val in top_features['Importance']],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Importanza Features (Proxy)",
        xaxis_title="Importanza",
        yaxis_title="Features",
        height=500
    )
    
    return fig

# ----------------7. Scatter Plot Outliers (da notebook esteso)
def create_outliers_scatter_plot(df, var1, var2):
    """
    Scatter plot con outliers evidenziati
    Estende analisi outliers notebook sezione 4.1.1
    """
    if df is None or var1 not in df.columns or var2 not in df.columns:
        return None
    
    from src.utils.data_processor import detect_outliers_iqr
    
    # Rileva outliers per entrambe le variabili
    outliers1, _, _ = detect_outliers_iqr(df[var1].dropna())
    outliers2, _, _ = detect_outliers_iqr(df[var2].dropna())
    
    # Combina indici outliers
    outlier_indices = set(outliers1.index) | set(outliers2.index)
    
    # Crea indicatore outlier
    df_plot = df.copy()
    df_plot['Is_Outlier'] = df_plot.index.isin(outlier_indices)
    
    fig = px.scatter(
        df_plot,
        x=var1,
        y=var2,
        color='Is_Outlier',
        color_discrete_map={False: COLOR_PALETTES['primary'], True: COLOR_PALETTES['danger']},
        title=f"Outliers: {COLUMN_LABELS.get(var1, var1)} vs {COLUMN_LABELS.get(var2, var2)}"
    )
    
    fig.update_layout(height=400)
    return fig

# ----------------8. Boxplot Comparison Outliers
def create_outliers_comparison_boxplot(df, variables):
    """
    Boxplot confronto multiple variabili per outliers
    """
    if df is None or not variables:
        return None
    
    fig = go.Figure()
    
    # Usa colori diretti
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98']
    
    for i, var in enumerate(variables):
        if var in df.columns:
            fig.add_trace(go.Box(
                y=df[var],
                name=COLUMN_LABELS.get(var, var),
                marker_color=default_colors[i % len(default_colors)]
            ))
    
    fig.update_layout(
        title="Confronto Distribuzioni per Outliers",
        yaxis_title="Valori",
        height=400
    )
    
    return fig

# ----------------9. Test Normalità Plots
def create_normality_test_plots(df, variable):
    """
    Grafici per test di normalità
    """
    if df is None or variable not in df.columns:
        return None
    
    data = df[variable].dropna()
    
    # Crea subplot con istogramma e Q-Q plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuzione', 'Q-Q Plot Normale')
    )
    
    # Istogramma
    fig.add_trace(
        go.Histogram(x=data, nbinsx=20, name="Distribuzione", opacity=0.7),
        row=1, col=1
    )
    
    # Q-Q plot approssimato
    sorted_data = np.sort(data)
    n = len(sorted_data)
    theoretical_quantiles = np.random.normal(data.mean(), data.std(), n)
    theoretical_quantiles.sort()
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles, 
            y=sorted_data, 
            mode='markers',
            name="Q-Q Plot",
            marker_color=COLOR_PALETTES['warning']
        ),
        row=1, col=2
    )
    
    # Linea di riferimento per Q-Q plot
    min_val = min(min(theoretical_quantiles), min(sorted_data))
    max_val = max(max(theoretical_quantiles), max(sorted_data))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name="Linea Normale",
            line=dict(dash='dash', color='red')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Test Normalità - {COLUMN_LABELS.get(variable, variable)}",
        height=400,
        showlegend=False
    )
    
    return fig

# ----------------10. Confronto Distribuzioni per Gruppo
def create_distribution_comparison_by_group(df, numeric_var, group_var):
    """
    Confronta distribuzioni di variabile numerica per gruppi
    """
    if df is None or numeric_var not in df.columns or group_var not in df.columns:
        return None
    
    fig = go.Figure()
    
    groups = df[group_var].unique()
    # Usa colori diretti invece di seaborn_palettes
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98FB98', '#87CEEB', '#DDA0DD']
    colors = default_colors[:len(groups)]
    
    for i, group in enumerate(groups):
        group_data = df[df[group_var] == group][numeric_var].dropna()
        group_label = VALUE_MAPPINGS.get(group_var, {}).get(group, str(group))
        
        fig.add_trace(go.Histogram(
            x=group_data,
            name=group_label,
            opacity=0.7,
            nbinsx=15,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title=f"Distribuzione {COLUMN_LABELS.get(numeric_var, numeric_var)} per {COLUMN_LABELS.get(group_var, group_var)}",
        xaxis_title=COLUMN_LABELS.get(numeric_var, numeric_var),
        yaxis_title="Frequenza",
        barmode='overlay',
        height=400
    )
    
    return fig

# ----------------11. Grafici per Segmentazione
def create_segments_survival_chart(df):
    """
    Grafico sopravvivenza per segmenti
    """
    if df is None or 'Segment' not in df.columns:
        return None
    
    segment_survival = df.groupby('Segment')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Bar(
        x=segment_survival.index,
        y=segment_survival.values,
        marker_color=COLOR_PALETTES['age_groups'],
        text=[f"{val:.1f}%" for val in segment_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Tasso Sopravvivenza per Segmento",
        xaxis_title="Segmento",
        yaxis_title="Tasso Sopravvivenza (%)",
        height=400
    )
    
    return fig

# ----------------12. Distribuzione Segmenti
def create_segments_distribution_chart(df):
    """
    Distribuzione dei segmenti nel dataset
    """
    if df is None or 'Segment' not in df.columns:
        return None
    
    segment_counts = df['Segment'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=segment_counts.index,
        values=segment_counts.values,
        hole=.3
    )])
    
    fig.update_layout(
        title="Distribuzione Segmenti Passeggeri",
        height=400
    )
    
    return fig

# ----------------13. Grafici Profili AFC (Age-Fare-Class)
def create_profiles_chart(profile_survival):
    """
    Grafico per profili Age-Fare-Class
    """
    if profile_survival is None:
        return None
    
    # Bubble chart: dimensione = conteggio, y = tasso sopravvivenza
    fig = go.Figure(data=go.Scatter(
        x=profile_survival.index,
        y=profile_survival['Tasso_Sopravvivenza'] * 100,
        mode='markers',
        marker=dict(
            size=profile_survival['Conteggio'],
            sizemode='diameter',
            sizeref=2.*max(profile_survival['Conteggio'])/(40.**2),
            sizemin=4,
            color=profile_survival['Tasso_Sopravvivenza'] * 100,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Tasso Sopravvivenza (%)")
        ),
        text=[f"Conteggio: {cnt}<br>Tasso: {rate*100:.1f}%" 
              for cnt, rate in zip(profile_survival['Conteggio'], profile_survival['Tasso_Sopravvivenza'])],
        hovertemplate='Profilo: %{x}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Profili Age-Fare-Class",
        xaxis_title="Profilo",
        yaxis_title="Tasso Sopravvivenza (%)",
        height=400
    )
    
    return fig

# ----------------14. Heatmap Pattern Sopravvivenza
def create_survival_patterns_heatmap(df, var1, var2):
    """
    Heatmap pattern sopravvivenza per 2 variabili
    """
    if df is None or var1 not in df.columns or var2 not in df.columns:
        return None
    
    # Calcola tasso sopravvivenza per combinazioni
    survival_matrix = df.groupby([var1, var2])['Survived'].mean() * 100
    survival_pivot = survival_matrix.unstack(fill_value=0)
    
    # Mappa etichette se disponibili
    x_labels = [VALUE_MAPPINGS.get(var2, {}).get(col, str(col)) for col in survival_pivot.columns]
    y_labels = [VALUE_MAPPINGS.get(var1, {}).get(idx, str(idx)) for idx in survival_pivot.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=survival_pivot.values,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        text=survival_pivot.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 10},
        colorbar=dict(title="Tasso Sopravvivenza (%)")
    ))
    
    fig.update_layout(
        title=f"Pattern Sopravvivenza: {COLUMN_LABELS.get(var1, var1)} vs {COLUMN_LABELS.get(var2, var2)}",
        xaxis_title=COLUMN_LABELS.get(var2, var2),
        yaxis_title=COLUMN_LABELS.get(var1, var1),
        height=400
    )
    
    return fig

# ----------------15. Radar Chart Confronto Segmenti
def create_segments_radar_chart(df, segments_col='Segment'):
    """
    Radar chart per confrontare caratteristiche dei segmenti
    """
    if df is None or segments_col not in df.columns:
        return None
    
    # Variabili per radar chart
    numeric_vars = ['Age', 'Fare', 'Family_Size', 'Survived']
    available_vars = [var for var in numeric_vars if var in df.columns]
    
    if len(available_vars) < 3:
        return None
    
    segments = df[segments_col].unique()
    
    fig = go.Figure()
    
    for segment in segments:
        segment_data = df[df[segments_col] == segment]
        
        # Calcola medie normalizzate (0-1)
        values = []
        for var in available_vars:
            if var == 'Survived':
                val = segment_data[var].mean()  # Già 0-1
            else:
                val = (segment_data[var].mean() - df[var].min()) / (df[var].max() - df[var].min())
            values.append(val)
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Chiudi il poligono
            theta=available_vars + [available_vars[0]],
            fill='toself',
            name=f'Segmento {segment}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Confronto Caratteristiche Segmenti",
        height=500
    )
    
    return fig