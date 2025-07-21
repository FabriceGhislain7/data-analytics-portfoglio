"""
src/components/bivariate_charts.py
Grafici specializzati per l'analisi bivariata dei fattori di sopravvivenza
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.config import COLOR_PALETTES, COLUMN_LABELS, VALUE_MAPPINGS

# ----------------1. Sopravvivenza per Classe Dettagliata (da notebook sezione 4.2.2.2)
def create_survival_by_class_detailed(df):
    """
    Grafico dettagliato sopravvivenza per classe
    Da notebook sezione 4.2.2.2
    """
    if df is None:
        return None
    
    # Calcola statistiche per classe
    class_stats = df.groupby('Pclass')['Survived'].agg(['sum', 'count', 'mean']).reset_index()
    class_stats.columns = ['Pclass', 'Sopravvissuti', 'Totale', 'Tasso']
    
    # Mappa le classi
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in class_stats['Pclass']]
    
    fig = go.Figure()
    
    # Barre sopravvissuti
    fig.add_trace(go.Bar(
        name='Sopravvissuti',
        x=class_labels,
        y=class_stats['Sopravvissuti'],
        marker_color=COLOR_PALETTES['survival'][1],
        text=class_stats['Sopravvissuti'],
        textposition='auto',
    ))
    
    # Barre totali (sfondo)
    fig.add_trace(go.Bar(
        name='Morti',
        x=class_labels,
        y=class_stats['Totale'] - class_stats['Sopravvissuti'],
        marker_color=COLOR_PALETTES['survival'][0],
        text=class_stats['Totale'] - class_stats['Sopravvissuti'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Sopravvivenza per Classe Passeggeri',
        xaxis_title='Classe',
        yaxis_title='Numero Passeggeri',
        barmode='stack',
        height=400
    )
    
    return fig

# ----------------2. Sopravvivenza per Genere Dettagliata (da notebook sezione 4.2.2.3)
def create_survival_by_gender_detailed(df):
    """
    Grafico dettagliato sopravvivenza per genere
    Da notebook sezione 4.2.2.3
    """
    if df is None:
        return None
    
    # Calcola percentuali sopravvivenza per genere
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    
    # Mappa i generi
    gender_labels = [VALUE_MAPPINGS['Sex'][sex] for sex in gender_survival.index]
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
        height=400
    )
    
    return fig

# ----------------3. Matrice Correlazione (da notebook sezione 4.1.2)
def create_correlation_heatmap(df):
    """
    Heatmap matrice di correlazione
    Da notebook sezione 4.1.2 - Spearman correlation
    """
    if df is None:
        return None
    
    # Seleziona solo variabili numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'PassengerId']
    
    if len(numeric_cols) < 2:
        return None
    
    # Calcola correlazione
    corr_matrix = df[numeric_cols].corr(method='spearman')
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[COLUMN_LABELS.get(col, col) for col in corr_matrix.columns],
        y=[COLUMN_LABELS.get(col, col) for col in corr_matrix.index],
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlazione")
    ))
    
    fig.update_layout(
        title="Matrice di Correlazione (Spearman)",
        height=500
    )
    
    return fig

# ----------------4. Distribuzione Classe (da notebook sezione 4.2.2.1)
def create_class_distribution_analysis(df):
    """
    Analisi distribuzione passeggeri per classe
    Da notebook sezione 4.2.2.1
    """
    if df is None:
        return None
    
    class_counts = df['Pclass'].value_counts().sort_index()
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in class_counts.index]
    
    fig = go.Figure(data=[go.Bar(
        x=class_labels,
        y=class_counts.values,
        marker=dict(color=COLOR_PALETTES['class']),
        text=class_counts.values,
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Distribuzione Passeggeri per Classe",
        xaxis_title="Classe",
        yaxis_title="Numero Passeggeri",
        height=400
    )
    
    return fig

# ----------------5. Tassi Sopravvivenza per Classe (da notebook sezione 4.2.2.2)
def create_survival_rates_by_class(df):
    """
    Tassi di sopravvivenza per classe
    Da notebook sezione 4.2.2.2
    """
    if df is None:
        return None
    
    survival_rates = df.groupby('Pclass')['Survived'].mean() * 100
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in survival_rates.index]
    
    fig = go.Figure(data=[go.Bar(
        x=class_labels,
        y=survival_rates.values,
        marker=dict(color=COLOR_PALETTES['class']),
        text=[f"{val:.1f}%" for val in survival_rates.values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title="Tasso di Sopravvivenza per Classe",
        xaxis_title="Classe",
        yaxis_title="Tasso Sopravvivenza (%)",
        height=400
    )
    
    return fig

# ----------------6. Analisi Classe Dettagliata (da notebook sezione 4.2.2.2)
def create_class_survival_detailed_analysis(df):
    """
    Analisi dettagliata classe con sopravvissuti e morti
    Da notebook sezione 4.2.2.2
    """
    if df is None:
        return None
    
    # Calcola statistiche dettagliate
    class_detail = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
    class_detail.columns = ['Morti', 'Sopravvissuti']
    class_detail['Totale'] = class_detail['Morti'] + class_detail['Sopravvissuti']
    class_detail['Tasso_Sopravvivenza'] = (class_detail['Sopravvissuti'] / class_detail['Totale']) * 100
    
    class_labels = [VALUE_MAPPINGS['Pclass'][idx] for idx in class_detail.index]
    
    # Crea subplot con barre e linea
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Barre sopravvissuti e morti
    fig.add_trace(
        go.Bar(name='Morti', x=class_labels, y=class_detail['Morti'], 
               marker_color=COLOR_PALETTES['survival'][0]),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(name='Sopravvissuti', x=class_labels, y=class_detail['Sopravvissuti'],
               marker_color=COLOR_PALETTES['survival'][1]),
        secondary_y=False,
    )
    
    # Linea tasso sopravvivenza
    fig.add_trace(
        go.Scatter(x=class_labels, y=class_detail['Tasso_Sopravvivenza'],
                   mode='lines+markers', name='Tasso Sopravvivenza (%)',
                   line=dict(color='red', width=3), marker=dict(size=8)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Classe")
    fig.update_yaxes(title_text="Numero Passeggeri", secondary_y=False)
    fig.update_yaxes(title_text="Tasso Sopravvivenza (%)", secondary_y=True)
    fig.update_layout(title="Analisi Dettagliata Sopravvivenza per Classe", height=400)
    
    return fig

# ----------------7. Confronto Sopravvivenza Genere (da notebook sezione 4.2.2.3)
def create_gender_survival_comparison(df):
    """
    Confronto sopravvivenza tra generi
    Da notebook sezione 4.2.2.3
    """
    if df is None:
        return None
    
    gender_survival = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
    gender_survival.columns = ['Morti', 'Sopravvissuti']
    
    gender_labels = [VALUE_MAPPINGS['Sex'][sex] for sex in gender_survival.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Morti',
        x=gender_labels,
        y=gender_survival['Morti'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Sopravvissuti',
        x=gender_labels,
        y=gender_survival['Sopravvissuti'],
        marker_color=COLOR_PALETTES['survival'][1]
    ))
    
    fig.update_layout(
        title='Confronto Sopravvivenza per Genere',
        xaxis_title='Genere',
        yaxis_title='Numero Passeggeri',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------8. Distribuzione Genere per Classe (da notebook sezione 4.2.2.3)
def create_gender_class_distribution(df):
    """
    Distribuzione genere per classe
    """
    if df is None:
        return None
    
    gender_class = df.groupby(['Pclass', 'Sex']).size().unstack(fill_value=0)
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in gender_class.index]
    
    fig = go.Figure()
    
    for i, sex in enumerate(gender_class.columns):
        sex_label = VALUE_MAPPINGS['Sex'][sex]
        fig.add_trace(go.Bar(
            name=sex_label,
            x=class_labels,
            y=gender_class[sex],
            marker_color=COLOR_PALETTES['gender'][i]
        ))
    
    fig.update_layout(
        title='Distribuzione Genere per Classe',
        xaxis_title='Classe',
        yaxis_title='Numero Passeggeri',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------9. Sopravvivenza Genere per Classe (da notebook sezione 4.2.2.3)
def create_gender_class_survival_analysis(df):
    """
    Analisi sopravvivenza per genere e classe
    Da notebook sezione 4.2.2.3
    """
    if df is None:
        return None
    
    # Calcola tassi sopravvivenza per genere e classe
    survival_by_gender_class = df.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100
    survival_pivot = survival_by_gender_class.unstack()
    
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in survival_pivot.index]
    
    fig = go.Figure()
    
    for i, sex in enumerate(survival_pivot.columns):
        sex_label = VALUE_MAPPINGS['Sex'][sex]
        fig.add_trace(go.Bar(
            name=sex_label,
            x=class_labels,
            y=survival_pivot[sex],
            marker_color=COLOR_PALETTES['gender'][i],
            text=[f"{val:.1f}%" for val in survival_pivot[sex]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Tasso Sopravvivenza per Genere e Classe',
        xaxis_title='Classe',
        yaxis_title='Tasso Sopravvivenza (%)',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------10. Distribuzione Età per Sopravvivenza (da notebook sezione 4.2.2.4)
def create_age_survival_distribution(df):
    """
    Distribuzione età per sopravvivenza
    Da notebook sezione 4.2.2.4
    """
    if df is None or 'Age' not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Istogramma per sopravvissuti e morti
    for survived, color, label in [(0, COLOR_PALETTES['survival'][0], 'Morti'), 
                                   (1, COLOR_PALETTES['survival'][1], 'Sopravvissuti')]:
        age_data = df[df['Survived'] == survived]['Age'].dropna()
        
        fig.add_trace(go.Histogram(
            x=age_data,
            name=label,
            opacity=0.7,
            nbinsx=20,
            marker_color=color
        ))
    
    fig.update_layout(
        title='Distribuzione Età per Sopravvivenza',
        xaxis_title='Età (anni)',
        yaxis_title='Frequenza',
        barmode='overlay',
        height=400
    )
    
    return fig

# ----------------11. Tassi Sopravvivenza per Gruppo Età (da notebook sezione 4.2.2.4)
def create_age_group_survival_rates(df):
    """
    Tassi sopravvivenza per gruppo di età
    Da notebook sezione 4.2.2.4
    """
    if df is None or 'Age_Group' not in df.columns:
        return None
    
    age_group_survival = df.groupby('Age_Group')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Bar(
        x=age_group_survival.index,
        y=age_group_survival.values,
        marker_color=COLOR_PALETTES['age_groups'],
        text=[f"{val:.1f}%" for val in age_group_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Tasso Sopravvivenza per Gruppo di Età',
        xaxis_title='Gruppo di Età',
        yaxis_title='Tasso Sopravvivenza (%)',
        height=400
    )
    
    return fig

# ----------------12. Sopravvivenza Età per Genere (da notebook sezione 4.2.2.4)
def create_age_gender_survival_analysis(df):
    """
    Analisi sopravvivenza età per genere
    Da notebook sezione 4.2.2.4
    """
    if df is None or 'Age_Group' not in df.columns:
        return None
    
    # Calcola tassi per età e genere
    age_gender_survival = df.groupby(['Age_Group', 'Sex'])['Survived'].mean() * 100
    survival_pivot = age_gender_survival.unstack()
    
    fig = go.Figure()
    
    for i, sex in enumerate(survival_pivot.columns):
        sex_label = VALUE_MAPPINGS['Sex'][sex]
        fig.add_trace(go.Bar(
            name=sex_label,
            x=survival_pivot.index,
            y=survival_pivot[sex],
            marker_color=COLOR_PALETTES['gender'][i]
        ))
    
    fig.update_layout(
        title='Tasso Sopravvivenza per Età e Genere',
        xaxis_title='Gruppo di Età',
        yaxis_title='Tasso Sopravvivenza (%)',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------13. Sopravvivenza per Categoria Prezzo (da notebook sezione 4.2.2.5)
def create_fare_category_survival(df):
    """
    Sopravvivenza per categoria prezzo
    Da notebook sezione 4.2.2.5
    """
    if df is None or 'Fare_Category' not in df.columns:
        return None
    
    fare_survival = df.groupby('Fare_Category')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Bar(
        x=fare_survival.index,
        y=fare_survival.values,
        marker_color=COLOR_PALETTES['primary'],
        text=[f"{val:.1f}%" for val in fare_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Tasso Sopravvivenza per Categoria Prezzo',
        xaxis_title='Categoria Prezzo',
        yaxis_title='Tasso Sopravvivenza (%)',
        height=400
    )
    
    return fig

# ----------------14. Distribuzione Prezzi per Sopravvivenza (da notebook sezione 4.2.2.5)
def create_fare_distribution_by_survival(df):
    """
    Distribuzione prezzi per sopravvivenza
    Da notebook sezione 4.2.2.5
    """
    if df is None:
        return None
    
    fig = go.Figure()
    
    for survived, color, label in [(0, COLOR_PALETTES['survival'][0], 'Morti'), 
                                   (1, COLOR_PALETTES['survival'][1], 'Sopravvissuti')]:
        fare_data = df[df['Survived'] == survived]['Fare']
        
        fig.add_trace(go.Box(
            y=fare_data,
            name=label,
            marker_color=color
        ))
    
    fig.update_layout(
        title='Distribuzione Prezzi per Sopravvivenza',
        yaxis_title='Prezzo Biglietto',
        height=400
    )
    
    return fig

# ----------------15. Analisi Prezzo-Classe-Sopravvivenza
def create_fare_class_survival_analysis(df):
    """
    Analisi combinata prezzo-classe-sopravvivenza
    """
    if df is None:
        return None
    
    # Media prezzi per classe e sopravvivenza
    fare_class_survival = df.groupby(['Pclass', 'Survived'])['Fare'].mean().unstack()
    fare_class_survival.columns = ['Morti', 'Sopravvissuti']
    
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in fare_class_survival.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Morti',
        x=class_labels,
        y=fare_class_survival['Morti'],
        marker_color=COLOR_PALETTES['survival'][0]
    ))
    
    fig.add_trace(go.Bar(
        name='Sopravvissuti',
        x=class_labels,
        y=fare_class_survival['Sopravvissuti'],
        marker_color=COLOR_PALETTES['survival'][1]
    ))
    
    fig.update_layout(
        title='Prezzo Medio per Classe e Sopravvivenza',
        xaxis_title='Classe',
        yaxis_title='Prezzo Medio',
        barmode='group',
        height=400
    )
    
    return fig

# ----------------16. Sopravvivenza per Dimensione Famiglia (da notebook sezione 4.2.2.6)
def create_family_size_survival(df):
    """
    Sopravvivenza per dimensione famiglia
    Da notebook sezione 4.2.2.6
    """
    if df is None or 'Family_Size' not in df.columns:
        return None
    
    family_survival = df.groupby('Family_Size')['Survived'].mean() * 100
    
    fig = go.Figure(data=[go.Scatter(
        x=family_survival.index,
        y=family_survival.values,
        mode='lines+markers',
        marker=dict(size=8, color=COLOR_PALETTES['primary']),
        line=dict(width=3)
    )])
    
    fig.update_layout(
        title='Tasso Sopravvivenza per Dimensione Famiglia',
        xaxis_title='Dimensione Famiglia',
        yaxis_title='Tasso Sopravvivenza (%)',
        height=400
    )
    
    return fig

# ----------------17. Solo vs Famiglia (da notebook sezione 4.2.2.6)
def create_alone_vs_family_analysis(df):
    """
    Confronto tra viaggiatori soli e con famiglia
    Da notebook sezione 4.2.2.6
    """
    if df is None or 'Is_Alone' not in df.columns:
        return None
    
    alone_survival = df.groupby('Is_Alone')['Survived'].mean() * 100
    labels = ['Con Famiglia', 'Solo']
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=alone_survival.values,
        marker_color=[COLOR_PALETTES['success'], COLOR_PALETTES['warning']],
        text=[f"{val:.1f}%" for val in alone_survival.values],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Sopravvivenza: Solo vs Con Famiglia',
        xaxis_title='Tipo Viaggiatore',
        yaxis_title='Tasso Sopravvivenza (%)',
        height=400
    )
    
    return fig

# ----------------18. Composizione Famiglia Dettagliata (da notebook sezione 4.2.2.6)
def create_family_composition_analysis(df):
    """
    Analisi dettagliata composizione famiglia
    Da notebook sezione 4.2.2.6
    """
    if df is None:
        return None
    
    # Analisi SibSp e Parch separatamente
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sopravvivenza per Fratelli/Coniugi', 'Sopravvivenza per Genitori/Figli')
    )
    
    # SibSp analysis
    sibsp_survival = df.groupby('SibSp')['Survived'].mean() * 100
    fig.add_trace(
        go.Bar(x=sibsp_survival.index, y=sibsp_survival.values, 
               name='SibSp', marker_color=COLOR_PALETTES['primary']),
        row=1, col=1
    )
    
    # Parch analysis
    parch_survival = df.groupby('Parch')['Survived'].mean() * 100
    fig.add_trace(
        go.Bar(x=parch_survival.index, y=parch_survival.values, 
               name='Parch', marker_color=COLOR_PALETTES['secondary']),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

# ----------------19. Analisi Multivariata (combinazione tutti i fattori)
def create_multivariate_survival_analysis(df):
    """
    Analisi multivariata di tutti i fattori
    """
    if df is None:
        return None
    
    # Heatmap sopravvivenza per classe e genere
    survival_matrix = df.groupby(['Pclass', 'Sex'])['Survived'].mean() * 100
    survival_pivot = survival_matrix.unstack()
    
    class_labels = [VALUE_MAPPINGS['Pclass'][pclass] for pclass in survival_pivot.index]
    gender_labels = [VALUE_MAPPINGS['Sex'][sex] for sex in survival_pivot.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=survival_pivot.values,
        x=gender_labels,
        y=class_labels,
        colorscale='RdYlGn',
        text=survival_pivot.round(1).values,
        texttemplate="%{text}%",
        textfont={"size": 12},
        colorbar=dict(title="Tasso Sopravvivenza (%)")
    ))
    
    fig.update_layout(
        title="Tasso Sopravvivenza per Classe e Genere",
        xaxis_title="Genere",
        yaxis_title="Classe",
        height=400
    )
    
    return fig

# ----------------20. Ranking Fattori Influenza
def calculate_survival_factors_ranking(df):
    """
    Calcola ranking dei fattori di influenza sulla sopravvivenza
    """
    if df is None:
        return None
    
    factors_impact = []
    
    # Calcola range di sopravvivenza per ogni fattore
    
    # Genere
    gender_range = df.groupby('Sex')['Survived'].mean().max() - df.groupby('Sex')['Survived'].mean().min()
    factors_impact.append({'Fattore': 'Genere', 'Range_Impatto': gender_range * 100, 'Importanza': 'Molto Alta'})
    
    # Classe
    class_range = df.groupby('Pclass')['Survived'].mean().max() - df.groupby('Pclass')['Survived'].mean().min()
    factors_impact.append({'Fattore': 'Classe', 'Range_Impatto': class_range * 100, 'Importanza': 'Alta'})
    
    # Età (se disponibile)
    if 'Age_Group' in df.columns:
        age_range = df.groupby('Age_Group')['Survived'].mean().max() - df.groupby('Age_Group')['Survived'].mean().min()
        factors_impact.append({'Fattore': 'Gruppo Età', 'Range_Impatto': age_range * 100, 'Importanza': 'Media'})
    
    # Famiglia
    if 'Family_Size' in df.columns:
        family_range = df.groupby('Family_Size')['Survived'].mean().max() - df.groupby('Family_Size')['Survived'].mean().min()
        factors_impact.append({'Fattore': 'Dimensione Famiglia', 'Range_Impatto': family_range * 100, 'Importanza': 'Media'})
    
    # Prezzo
    if 'Fare_Category' in df.columns:
        fare_range = df.groupby('Fare_Category')['Survived'].mean().max() - df.groupby('Fare_Category')['Survived'].mean().min()
        factors_impact.append({'Fattore': 'Categoria Prezzo', 'Range_Impatto': fare_range * 100, 'Importanza': 'Media'})
    
    factors_df = pd.DataFrame(factors_impact)
    factors_df = factors_df.sort_values('Range_Impatto', ascending=False)
    factors_df['Range_Impatto'] = factors_df['Range_Impatto'].round(1)
    
    return factors_df