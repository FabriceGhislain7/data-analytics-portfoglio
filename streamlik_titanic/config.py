import os

# Configurazione path del progetto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Path specifici per i file
TITANIC_CSV_PATH = os.path.join(DATA_DIR, "data_titanic.csv")

# URL di backup (se file locale non disponibile)
TITANIC_URL = "https://raw.githubusercontent.com/FabriceGhislain7/data_analyst_scientist/main/titanic_project/data_titanic.csv"

# Configurazioni Streamlit
PAGE_CONFIG = {
    "page_title": "Titanic Analysis",
    "page_icon": "🚢",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Parametri per l'analisi
AGE_BINS = [0, 12, 25, 40, 80]
AGE_LABELS = ['Child (0-12)', 'Young (13-25)', 'Adult (26-40)', 'Senior (41+)']

FARE_QUANTILES = 4
FARE_LABELS = ['Low', 'Medium', 'High', 'Very High']

# Parametri per Family Size
FAMILY_SIZE_CATEGORIES = {
    'Solo': [1],
    'Small': [2, 3],
    'Medium': [4],
    'Large': [5, 6, 7, 8]
}

# Parametri per l'outlier detection
OUTLIER_CONFIG = {
    'method': 'IQR',
    'lower_q': 0.25,
    'upper_q': 0.75,
    'iqr_factor': 1.5
}

# Colori per i grafici
COLORS = {
    "survived": "#2E8B57",
    "not_survived": "#DC143C",
    "male": "#1f77b4",
    "female": "#ff7f0e",
    "class_1": "#2E8B57",
    "class_2": "#FFD700",
    "class_3": "#DC143C",
    "palette_main": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "palette_survival": ["#DC143C", "#2E8B57"],
    "palette_classes": ["#2E8B57", "#FFD700", "#DC143C"]
}

# Configurazioni per i grafici
PLOT_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'whitegrid',
    'font_scale': 1.1,
    'title_size': 14,
    'label_size': 12
}

# Project Information
PROJECT_INFO = {
    "title": "Titanic Dataset Analysis: Factors Influencing Passenger Survival",
    "description": """
    Analisi completa dei fattori che hanno influenzato la sopravvivenza dei passeggeri del Titanic.
    Include data cleaning, analisi esplorativa, e identificazione dei pattern di sopravvivenza.
    """,
    "key_findings": [
        "Women had a 75% survival rate vs 20% for men",
        "First-class passengers had 65% survival rate vs 24% for third-class",
        "Children (0-12) had the highest survival rate at 58%",
        "Medium families (3-4 people) had better survival odds than solo travelers",
        "Higher fare strongly correlated with survival (20% to 62%)"
    ],
    "techniques": [
        "Data Cleaning: Handled missing values, removed duplicates, and imputed data",
        "Descriptive Statistics: Summarized key metrics like mean, median, and correlation",
        "Visualizations: Used bar plots, histograms, box plots, and heatmaps",
        "Outlier Detection: Identified and treated outliers using IQR method",
        "Grouped Analysis: Compared survival rates across categories",
        "Feature Engineering: Created family size and age group variables"
    ],
    "dataset_stats": {
        "total_passengers": 891,
        "survivors": 342,
        "overall_survival_rate": 38.4,
        "missing_age": 177,
        "missing_cabin": 687,
        "missing_embarked": 2
    }
}

# Variable Definitions
VARIABLE_DEFINITIONS = {
    "PassengerId": "Unique identifier for each passenger",
    "Survived": "Survival (0 = No, 1 = Yes)",
    "Pclass": "Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)",
    "Name": "Passenger name",
    "Sex": "Sex (male/female)",
    "Age": "Age in years",
    "SibSp": "# of siblings / spouses aboard the Titanic",
    "Parch": "# of parents / children aboard the Titanic",
    "Ticket": "Ticket number",
    "Fare": "Passenger fare",
    "Cabin": "Cabin number",
    "Embarked": "Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)"
}

# Variable Notes
VARIABLE_NOTES = {
    "Pclass": """
    A proxy for socio-economic status (SES):
    • 1st = Upper class
    • 2nd = Middle class  
    • 3rd = Lower class
    """,
    "Age": "Age is fractional if less than 1. If estimated, in the form of xx.5",
    "SibSp": """
    Sibling = brother, sister, stepbrother, stepsister
    Spouse = husband, wife (mistresses and fiancés were ignored)
    """,
    "Parch": """
    Parent = mother, father
    Child = daughter, son, stepdaughter, stepson
    Some children travelled only with a nanny, therefore parch=0
    """,
    "Embarked": """
    C = Cherbourg, France
    Q = Queenstown, Ireland  
    S = Southampton, England
    """
}

# Cleaning Methods Information
CLEANING_METHODS = {
    "method_1": {
        "name": "Remove Duplicates",
        "description": "Removes duplicate rows (none found in this dataset)"
    },
    "method_2": {
        "name": "Remove Columns with >50% Missing",
        "description": "Removes columns where more than 50% of values are missing (removed Cabin column)"
    },
    "method_3": {
        "name": "Remove Rows with Missing Values",
        "description": "Removes rows with missing Age or Embarked values (712 rows remaining)"
    },
    "method_4": {
        "name": "Impute Missing Values",
        "description": "Replaces missing Age values with mean age (891 rows preserved)"
    }
}

# Sidebar Configuration
SIDEBAR_CONFIG = {
    "data_cleaning_methods": list(CLEANING_METHODS.keys()),
    "visualization_types": ["Count Plot", "Percentage Plot", "Pie Chart", "Box Plot", "Histogram"],
    "analysis_variables": ["Survived", "Pclass", "Sex", "Age", "Fare", "Family_Size"]
}

# Messages and Texts
MESSAGES = {
    "data_loaded": "Dataset caricato con successo",
    "data_cleaned": "Data cleaning completato",
    "analysis_complete": "Analisi completata",
    "model_trained": "Modello addestrato",
    "no_data": "Errore nel caricamento dei dati",
    "processing": "Elaborazione in corso..."
}

# Cache Configuration
CACHE_CONFIG = {
    "ttl": 3600,  # 1 hour
    "allow_output_mutation": True
}

# Statistical Methods Configuration
STATISTICAL_METHODS = {
    "correlation_methods": ["pearson", "spearman", "kendall"],
    "outlier_methods": ["IQR", "Z-score", "Modified Z-score"],
    "imputation_methods": ["mean", "median", "mode", "forward_fill", "backward_fill"]
}

# Machine Learning Configuration
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "scoring_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
}

# Documentation Links
DOCUMENTATION_LINKS = {
    "pandas": {
        "general": "https://pandas.pydata.org/docs/",
        "groupby": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
        "merge": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html",
        "fillna": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html",
        "describe": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html",
        "value_counts": "https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html",
        "qcut": "https://pandas.pydata.org/docs/reference/api/pandas.qcut.html"
    },
    "matplotlib": {
        "general": "https://matplotlib.org/stable/contents.html",
        "pyplot": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html",
        "subplots": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html",
        "bar": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html",
        "hist": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html"
    },
    "seaborn": {
        "general": "https://seaborn.pydata.org/",
        "countplot": "https://seaborn.pydata.org/generated/seaborn.countplot.html",
        "boxplot": "https://seaborn.pydata.org/generated/seaborn.boxplot.html",
        "heatmap": "https://seaborn.pydata.org/generated/seaborn.heatmap.html",
        "barplot": "https://seaborn.pydata.org/generated/seaborn.barplot.html",
        "histplot": "https://seaborn.pydata.org/generated/seaborn.histplot.html"
    },
    "streamlit": {
        "general": "https://docs.streamlit.io/",
        "api_reference": "https://docs.streamlit.io/library/api-reference",
        "charts": "https://docs.streamlit.io/library/api-reference/charts",
        "data_elements": "https://docs.streamlit.io/library/api-reference/data",
        "layout": "https://docs.streamlit.io/library/api-reference/layout"
    },
    "numpy": {
        "general": "https://numpy.org/doc/stable/",
        "statistical_functions": "https://numpy.org/doc/stable/reference/routines.statistics.html",
        "array_creation": "https://numpy.org/doc/stable/reference/routines.array-creation.html"
    },
    "scikit_learn": {
        "general": "https://scikit-learn.org/stable/",
        "preprocessing": "https://scikit-learn.org/stable/modules/preprocessing.html",
        "model_selection": "https://scikit-learn.org/stable/modules/model_selection.html",
        "metrics": "https://scikit-learn.org/stable/modules/model_evaluation.html",
        "supervised_learning": "https://scikit-learn.org/stable/supervised_learning.html"
    },
    "statistical_concepts": {
        "iqr_outliers": "https://en.wikipedia.org/wiki/Interquartile_range",
        "correlation": "https://en.wikipedia.org/wiki/Correlation_and_dependence",
        "chi_square": "https://en.wikipedia.org/wiki/Chi-squared_test",
        "cross_validation": "https://en.wikipedia.org/wiki/Cross-validation_(statistics)"
    },
    "data_science": {
        "eda_guide": "https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15",
        "feature_engineering": "https://towardsdatascience.com/feature-engineering-for-machine-learning-3a5e293a5114",
        "data_cleaning": "https://towardsdatascience.com/the-ultimate-guide-to-data-cleaning-3969843991d4"
    }
}