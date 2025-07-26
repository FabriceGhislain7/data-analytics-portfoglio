"""
Titanic Survival Analysis - Configuration File
==============================================

Configurazione centralizzata per l'applicazione Streamlit di analisi del Titanic.
Tutti i parametri, percorsi e impostazioni sono definiti qui per facilitare
la manutenzione e la personalizzazione.

Autore: Data Analyst
Data: 2025
"""

import os

def find_project_root():
    """
    Trova la cartella root del progetto cercando file marker caratteristici.
    Questo approccio funziona ovunque il progetto venga spostato.
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # File/cartelle che indicano la root del progetto
    project_markers = [
        'requirements.txt',
        'README.md', 
        '.git',
        'app.py',
        'Home.py',
        'src'
    ]
    
    # Risali nella gerarchia fino a trovare la root
    max_levels = 5  # Limite per evitare loop infiniti
    for _ in range(max_levels):
        # Controlla se esistono i marker nella cartella corrente
        for marker in project_markers:
            if os.path.exists(os.path.join(current_path, marker)):
                return current_path
        
        # Se siamo già alla root del filesystem, fermati
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            break
            
        current_path = parent_path
    
    # Se non trova la root, usa la cartella del file config
    return os.path.dirname(os.path.abspath(__file__))

# Trova automaticamente la root del progetto
PROJECT_ROOT = find_project_root()

# Directory struttura del progetto (percorsi relativi alla root)
BASE_DIR = PROJECT_ROOT
DATA_DIR = os.path.join(BASE_DIR, "src", "data")
MODELS_DIR = os.path.join(BASE_DIR, "src", "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLES_DIR = os.path.join(ASSETS_DIR, "styles")

# Crea le directory se non esistono
def ensure_directories():
    """Crea le directory necessarie se non esistono"""
    directories = [DATA_DIR, MODELS_DIR, ASSETS_DIR, STYLES_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# File di dati
DATA_FILE = os.path.join(DATA_DIR, "data_titanic.csv")
DATA_URL = "https://raw.githubusercontent.com/FabriceGhislain7/data_analyst_scientist/main/titanic_project/data_titanic.csv"

# Colonne del dataset
DATASET_COLUMNS = {
    'NUMERICAL': ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare'],
    'CATEGORICAL': ['Survived', 'Pclass', 'Sex', 'Embarked'],
    'TEXT': ['Name', 'Ticket', 'Cabin'],
    'TARGET': 'Survived'
}

# Mapping per etichette leggibili
COLUMN_LABELS = {
    'PassengerId': 'ID Passeggero',
    'Survived': 'Sopravvissuto',
    'Pclass': 'Classe',
    'Name': 'Nome',
    'Sex': 'Sesso',
    'Age': 'Eta',
    'SibSp': 'Fratelli/Coniugi',
    'Parch': 'Genitori/Figli',
    'Ticket': 'Biglietto',
    'Fare': 'Prezzo Biglietto',
    'Cabin': 'Cabina',
    'Embarked': 'Porto di Imbarco'
}

# Mapping per valori categorici
VALUE_MAPPINGS = {
    'Survived': {0: 'No', 1: 'Si'},
    'Pclass': {1: '1a Classe', 2: '2a Classe', 3: '3a Classe'},
    'Sex': {'male': 'Uomo', 'female': 'Donna'},
    'Embarked': {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
}

# Parametri per la gestione degli outlier
OUTLIER_CONFIG = {
    'method': 'IQR',
    'lower_quantile': 0.25,
    'upper_quantile': 0.75,
    'iqr_multiplier': 1.5,
    'replacement_methods': ['remove', 'mean', 'median', 'mode']
}

# Configurazione gruppi di eta
AGE_GROUPS = {
    'Children': {'min': 0, 'max': 12, 'label': 'Bambini (0-12 anni)'},
    'Young_Adults': {'min': 13, 'max': 25, 'label': 'Giovani Adulti (13-25 anni)'},
    'Middle_Adults': {'min': 26, 'max': 40, 'label': 'Adulti (26-40 anni)'},
    'Older_Adults': {'min': 41, 'max': 100, 'label': 'Anziani (41+ anni)'}
}

# Configurazione categorie prezzo
FARE_CATEGORIES = {
    'bins': 4,
    'labels': ['Basso', 'Medio', 'Alto', 'Molto Alto'],
    'method': 'qcut'
}

# Soglie per missing values
MISSING_VALUE_THRESHOLDS = {
    'drop_column_threshold': 0.5,
    'drop_row_threshold': 0.8
}

# Palette colori
COLOR_PALETTES = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'success': '#45B7D1',
    'warning': '#FFA07A',
    'danger': '#FF6B6B',
    'survival': ['#FF6B6B', '#4ECDC4'],
    'gender': ['#87CEEB', '#FFB6C1'],
    'class': ['#FFD700', '#87CEEB', '#DDA0DD'],
    'age_groups': ['#FFE4B5', '#98FB98', '#87CEEB', '#DDA0DD'],
    'seaborn_palettes': ['viridis', 'plasma', 'coolwarm', 'Set2', 'Dark2']
}

# Configurazioni grafici
CHART_CONFIG = {
    'figure_size': (12, 6),
    'dpi': 100,
    'style': 'whitegrid',
    'font_scale': 1.1,
    'title_fontsize': 14,
    'label_fontsize': 12,
    'legend_fontsize': 10
}

# Configurazione istogrammi
HISTOGRAM_CONFIG = {
    'bins': 20,
    'kde': True,
    'alpha': 0.7,
    'edgecolor': 'black',
    'linewidth': 0.5
}

# Configurazione pagina
PAGE_CONFIG = {
    'page_title': 'Titanic Survival Analysis',
    'page_icon': '🚢',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Testi e descrizioni
APP_TEXTS = {
    'main_title': 'Analisi di Sopravvivenza del Titanic',
    'subtitle': 'Analisi esplorativa e predittiva dei fattori che hanno influenzato la sopravvivenza',
    'description': '''
    Questa applicazione fornisce un'analisi completa del famoso dataset del Titanic,
    esplorando i fattori che hanno influenzato la sopravvivenza dei passeggeri durante
    il tragico naufragio del 1912.
    ''',
    'data_source': 'Fonte dati: Kaggle Titanic Competition',
    'footer': '''
    ---
    Sviluppato con: Python, Streamlit, Pandas, Scikit-learn
    
    Nota: Questa analisi è basata sui dati storici disponibili e serve per scopi educativi.
    '''
}

# Modelli ML da utilizzare
ML_MODELS = {
    'LogisticRegression': {
        'name': 'Regressione Logistica',
        'params': {'random_state': 42, 'max_iter': 1000}
    },
    'RandomForestClassifier': {
        'name': 'Random Forest',
        'params': {'n_estimators': 100, 'random_state': 42}
    },
    'GradientBoostingClassifier': {
        'name': 'Gradient Boosting',
        'params': {'random_state': 42}
    },
    'SVC': {
        'name': 'Support Vector Machine',
        'params': {'random_state': 42, 'probability': True}
    },
    'DecisionTreeClassifier': {
        'name': 'Albero di Decisione',
        'params': {'random_state': 42}
    }
}

# Feature engineering
FEATURE_ENGINEERING = {
    'create_family_size': True,
    'create_is_alone': True,
    'create_title_from_name': True,
    'create_fare_per_person': True,
    'create_age_groups': True,
    'create_deck_from_cabin': True
}

# Preprocessamento
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True,
    'scale_features': True,
    'handle_missing': 'median'
}

# Metriche di valutazione
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1',
    'roc_auc', 'confusion_matrix', 'classification_report'
]

# Impostazioni debug
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
SHOW_RAW_DATA = DEBUG_MODE
SHOW_PROCESSING_STEPS = DEBUG_MODE

# Cache settings
CACHE_CONFIG = {
    'ttl': 3600,
    'max_entries': 100,
    'allow_output_mutation': True
}

# Mapping delle sezioni del notebook alle pagine Streamlit
NOTEBOOK_SECTIONS = {
    'data_loading': {
        'page': '1_Data_Overview.py',
        'section': 'Dataset Loading and Initial Exploration'
    },
    'missing_values': {
        'page': '1_Data_Overview.py', 
        'section': 'Missing Values Analysis'
    },
    'duplicates': {
        'page': '1_Data_Overview.py',
        'section': 'Duplicate Detection'
    },
    'data_cleaning': {
        'page': '1_Data_Overview.py',
        'section': 'Data Cleaning Methods'
    },
    'descriptive_stats': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Descriptive Statistics Analysis'
    },
    'numerical_viz': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Numerical Variables Visualization'
    },
    'age_analysis': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Single Variable Analysis - Age'
    },
    'categorical_analysis': {
        'page': '2_Univariate_Analysis.py',
        'section': 'Categorical Variables - Survival Analysis'
    },
    'survival_by_class': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Class-based Survival Analysis'
    },
    'survival_by_gender': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Gender-based Survival Analysis'
    },
    'survival_by_age_groups': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Age Group Survival Analysis'
    },
    'survival_by_fare': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Fare-based Survival Analysis'
    },
    'survival_by_family': {
        'page': '3_Bivariate_Analysis.py',
        'section': 'Family Size Impact on Survival'
    },
    'correlation_analysis': {
        'page': '4_Advanced_Analytics.py',
        'section': 'Correlation Matrix and Heatmaps'
    },
    'outlier_detection': {
        'page': '4_Advanced_Analytics.py',
        'section': 'Advanced Outlier Detection'
    },
    'machine_learning': {
        'page': '5_ML_Predictions.py',
        'section': 'Machine Learning Models'
    }
}

# Funzioni utility aggiornate
def get_data_path():
    """Restituisce il percorso del file dati come stringa"""
    return DATA_FILE

def get_model_path(model_name):
    """Restituisce il percorso per salvare un modello"""
    ensure_directories()  # Assicura che la directory esista
    return os.path.join(MODELS_DIR, f"{model_name}.pkl")

def load_custom_css():
    """Carica CSS personalizzato se disponibile"""
    css_file = os.path.join(STYLES_DIR, "main.css")
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def get_color_palette(palette_name='survival'):
    """Restituisce una palette di colori specifica"""
    return COLOR_PALETTES.get(palette_name, COLOR_PALETTES['survival'])

def format_percentage(value, decimals=1):
    """Formatta un valore come percentuale"""
    return f"{value:.{decimals}f}%"

def format_number(value, decimals=2):
    """Formatta un numero con decimali specificati"""
    return f"{value:.{decimals}f}"

def get_project_info():
    """Restituisce informazioni sul progetto e i percorsi"""
    return {
        'project_root': PROJECT_ROOT,
        'config_location': os.path.abspath(__file__),
        'data_directory': DATA_DIR,
        'models_directory': MODELS_DIR,
        'data_file_exists': os.path.exists(DATA_FILE),
        'directories_created': all(os.path.exists(d) for d in [DATA_DIR, MODELS_DIR, ASSETS_DIR])
    }

# Inizializzazione automatica
if __name__ == "__main__":
    # Quando il file config viene eseguito direttamente, 
    # mostra informazioni sui percorsi
    ensure_directories()
    info = get_project_info()
    
    print("=== TITANIC PROJECT CONFIGURATION ===")
    print(f"Project Root: {info['project_root']}")
    print(f"Config File: {info['config_location']}")
    print(f"Data Directory: {info['data_directory']}")
    print(f"Models Directory: {info['models_directory']}")
    print(f"Data file exists: {info['data_file_exists']}")
    print(f"All directories created: {info['directories_created']}")
    
    if not info['data_file_exists']:
        print(f"\n⚠️  Data file not found at: {DATA_FILE}")
        print("Download it from:", DATA_URL)