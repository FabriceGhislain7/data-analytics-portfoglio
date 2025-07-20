import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import COLORS, PLOT_CONFIG

# Set style
plt.style.use('default')
sns.set_style(PLOT_CONFIG['style'])
sns.set_context("notebook", font_scale=PLOT_CONFIG['font_scale'])

# =============================================================================
# MISSING VALUES VISUALIZATIONS
# =============================================================================

def plot_missing_values_heatmap(df):
    """
    Create heatmap of missing values.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        matplotlib.figure.Figure: Heatmap figure
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
    
    # Create heatmap
    sns.heatmap(df.isnull(), 
                cbar=True, 
                cmap="viridis", 
                ax=ax,
                yticklabels=False)
    
    ax.set_title("Missing Values Heatmap", fontsize=PLOT_CONFIG['title_size'])
    ax.set_xlabel("Columns", fontsize=PLOT_CONFIG['label_size'])
    ax.set_ylabel("Rows", fontsize=PLOT_CONFIG['label_size'])
    
    plt.tight_layout()
    return fig

def plot_missing_values_bar(missing_percentage):
    """
    Create bar plot of missing values percentage.
    
    Args:
        missing_percentage (pd.Series): Missing values percentage
        
    Returns:
        matplotlib.figure.Figure: Bar plot figure
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
    
    # Filter only columns with missing values
    missing_data = missing_percentage[missing_percentage > 0]
    
    if len(missing_data) > 0:
        bars = ax.bar(missing_data.index, missing_data.values, 
                     color=COLORS['not_survived'], alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax.set_title("Percentage of Missing Values by Column", 
                    fontsize=PLOT_CONFIG['title_size'])
        ax.set_xlabel("Columns", fontsize=PLOT_CONFIG['label_size'])
        ax.set_ylabel("Missing Percentage (%)", fontsize=PLOT_CONFIG['label_size'])
        ax.set_ylim(0, missing_data.max() * 1.1)
        
        plt.xticks(rotation=45)
    else:
        ax.text(0.5, 0.5, 'No missing values found', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=14)
        ax.set_title("No Missing Values", fontsize=PLOT_CONFIG['title_size'])
    
    plt.tight_layout()
    return fig

# =============================================================================
# BASIC DISTRIBUTION PLOTS
# =============================================================================

def plot_data_types_distribution(df):
    """
    Plot distribution of data types.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        matplotlib.figure.Figure: Pie chart figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Count data types
    type_counts = df.dtypes.astype(str).value_counts()
    type_mapping = {
        'int64': 'Integer',
        'float64': 'Float', 
        'object': 'Object/String'
    }
    
    # Map to readable names
    labels = [type_mapping.get(t, t) for t in type_counts.index]
    
    # Create pie chart
    colors = COLORS['palette_main'][:len(type_counts)]
    wedges, texts, autotexts = ax.pie(type_counts.values, 
                                     labels=labels,
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90)
    
    ax.set_title("Data Types Distribution", fontsize=PLOT_CONFIG['title_size'])
    
    plt.tight_layout()
    return fig

# =============================================================================
# SURVIVAL ANALYSIS PLOTS
# =============================================================================

def plot_survival_overview(df):
    """
    Create overview of survival statistics.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        matplotlib.figure.Figure: Combined survival plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    survival_counts = df['Survived'].value_counts()
    bars1 = axes[0].bar(['Not Survived', 'Survived'], 
                       survival_counts.values,
                       color=[COLORS['not_survived'], COLORS['survived']])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom')
    
    axes[0].set_title("Survival Count", fontsize=PLOT_CONFIG['title_size'])
    axes[0].set_ylabel("Count", fontsize=PLOT_CONFIG['label_size'])
    
    # Percentage plot
    survival_pct = df['Survived'].value_counts(normalize=True) * 100
    bars2 = axes[1].bar(['Not Survived', 'Survived'], 
                       survival_pct.values,
                       color=[COLORS['not_survived'], COLORS['survived']])
    
    # Add percentage labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    axes[1].set_title("Survival Percentage", fontsize=PLOT_CONFIG['title_size'])
    axes[1].set_ylabel("Percentage (%)", fontsize=PLOT_CONFIG['label_size'])
    axes[1].set_ylim(0, 70)
    
    plt.tight_layout()
    return fig

def plot_survival_by_category(df, category_col, title_suffix=""):
    """
    Plot survival rates by categorical variable.
    
    Args:
        df (pd.DataFrame): Input dataset
        category_col (str): Column to analyze
        title_suffix (str): Additional title text
        
    Returns:
        matplotlib.figure.Figure: Survival by category plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    survival_data = df.groupby([category_col, 'Survived']).size().unstack(fill_value=0)
    
    survival_data.plot(kind='bar', 
                      ax=axes[0],
                      color=[COLORS['not_survived'], COLORS['survived']],
                      width=0.8)
    
    axes[0].set_title(f"Survival Count by {category_col} {title_suffix}".strip(), 
                     fontsize=PLOT_CONFIG['title_size'])
    axes[0].set_xlabel(category_col, fontsize=PLOT_CONFIG['label_size'])
    axes[0].set_ylabel("Count", fontsize=PLOT_CONFIG['label_size'])
    axes[0].legend(['Not Survived', 'Survived'])
    axes[0].tick_params(axis='x', rotation=45)
    
    # Percentage plot
    survival_rate = df.groupby(category_col)['Survived'].mean() * 100
    bars = axes[1].bar(survival_rate.index, survival_rate.values,
                      color=COLORS['survived'], alpha=0.7)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    axes[1].set_title(f"Survival Rate by {category_col} {title_suffix}".strip(),
                     fontsize=PLOT_CONFIG['title_size'])
    axes[1].set_xlabel(category_col, fontsize=PLOT_CONFIG['label_size'])
    axes[1].set_ylabel("Survival Rate (%)", fontsize=PLOT_CONFIG['label_size'])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    return fig

# =============================================================================
# AGE ANALYSIS PLOTS
# =============================================================================

def plot_age_distribution(df):
    """
    Plot age distribution with survival overlay.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        matplotlib.figure.Figure: Age distribution plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Age histogram
    axes[0].hist(df['Age'].dropna(), bins=20, alpha=0.7, 
                color=COLORS['palette_main'][0], edgecolor='black')
    axes[0].set_title("Age Distribution", fontsize=PLOT_CONFIG['title_size'])
    axes[0].set_xlabel("Age", fontsize=PLOT_CONFIG['label_size'])
    axes[0].set_ylabel("Count", fontsize=PLOT_CONFIG['label_size'])
    
    # Age distribution by survival
    survived = df[df['Survived'] == 1]['Age'].dropna()
    not_survived = df[df['Survived'] == 0]['Age'].dropna()
    
    axes[1].hist([not_survived, survived], bins=20, alpha=0.7,
                label=['Not Survived', 'Survived'],
                color=[COLORS['not_survived'], COLORS['survived']])
    
    axes[1].set_title("Age Distribution by Survival", fontsize=PLOT_CONFIG['title_size'])
    axes[1].set_xlabel("Age", fontsize=PLOT_CONFIG['label_size'])
    axes[1].set_ylabel("Count", fontsize=PLOT_CONFIG['label_size'])
    axes[1].legend()
    
    plt.tight_layout()
    return fig

def plot_age_boxplot_by_survival(df):
    """
    Create boxplot of age by survival status.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        matplotlib.figure.Figure: Age boxplot figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplot
    box_data = [df[df['Survived'] == 0]['Age'].dropna(),
                df[df['Survived'] == 1]['Age'].dropna()]
    
    bp = ax.boxplot(box_data, 
                   labels=['Not Survived', 'Survived'],
                   patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor(COLORS['not_survived'])
    bp['boxes'][1].set_facecolor(COLORS['survived'])
    
    ax.set_title("Age Distribution by Survival Status", 
                fontsize=PLOT_CONFIG['title_size'])
    ax.set_ylabel("Age", fontsize=PLOT_CONFIG['label_size'])
    
    plt.tight_layout()
    return fig

# =============================================================================
# CORRELATION AND RELATIONSHIP PLOTS
# =============================================================================

def plot_correlation_heatmap(df, method='spearman'):
    """
    Create correlation heatmap for numerical variables.
    
    Args:
        df (pd.DataFrame): Input dataset
        method (str): Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        matplotlib.figure.Figure: Correlation heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr(method=method)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                ax=ax)
    
    ax.set_title(f"{method.capitalize()} Correlation Matrix", 
                fontsize=PLOT_CONFIG['title_size'])
    
    plt.tight_layout()
    return fig

# =============================================================================
# OUTLIER DETECTION PLOTS
# =============================================================================

def plot_outlier_detection(df, column):
    """
    Create outlier detection plots for a numerical column.
    
    Args:
        df (pd.DataFrame): Input dataset
        column (str): Column to analyze for outliers
        
    Returns:
        matplotlib.figure.Figure: Outlier detection plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    axes[0].boxplot(df[column].dropna())
    axes[0].set_title(f"Boxplot - {column}", fontsize=PLOT_CONFIG['title_size'])
    axes[0].set_ylabel(column, fontsize=PLOT_CONFIG['label_size'])
    
    # Histogram with outlier bounds
    data = df[column].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    axes[1].hist(data, bins=20, alpha=0.7, color=COLORS['palette_main'][0])
    axes[1].axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound: {lower_bound:.2f}')
    axes[1].axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound: {upper_bound:.2f}')
    
    axes[1].set_title(f"Distribution with Outlier Bounds - {column}", 
                     fontsize=PLOT_CONFIG['title_size'])
    axes[1].set_xlabel(column, fontsize=PLOT_CONFIG['label_size'])
    axes[1].set_ylabel("Frequency", fontsize=PLOT_CONFIG['label_size'])
    axes[1].legend()
    
    plt.tight_layout()
    return fig

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_plot(fig, filename, dpi=300):
    """
    Save plot to file.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Output filename
        dpi (int): Resolution for saved figure
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')

def set_plot_style(style='whitegrid'):
    """
    Set global plot style.
    
    Args:
        style (str): Seaborn style to use
    """
    sns.set_style(style)
    plt.rcParams['figure.figsize'] = PLOT_CONFIG['figure_size']
    plt.rcParams['font.size'] = PLOT_CONFIG['label_size']