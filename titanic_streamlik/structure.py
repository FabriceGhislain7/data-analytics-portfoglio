#!/usr/bin/env python3
"""
Script per creare la struttura del progetto Titanic Streamlit App
"""

import os
import shutil
from pathlib import Path

def create_project_structure():
    """Crea la struttura completa del progetto Streamlit"""
    
    # Directory base del progetto
    base_dir = Path(".")
    
    # Definizione della struttura delle cartelle
    folders = [
        "src",
        "src/components",
        "src/data",
        "src/models",
        "src/utils",
        "assets",
        "assets/images",
        "assets/styles",
        "pages",
        "tests",
        "docs",
        ".streamlit"
    ]
    
    # Creazione delle cartelle
    print("Creazione delle cartelle...")
    for folder in folders:
        folder_path = base_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Cartella creata: {folder}")
    
    # Definizione dei file da creare/spostare
    files_to_create = {
        # File principali dell'app
        "app.py": "# File principale dell'applicazione Streamlit\n",
        "Home.py": "# Pagina principale (per multipage app)\n",
        
        # File di configurazione
        "requirements.txt": "# Requirements già esistente\n",
        ".streamlit/config.toml": """[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501
""",
        ".streamlit/secrets.toml": "# File per secrets (non committare su git)\n",
        
        # Componenti Streamlit
        "src/components/__init__.py": "",
        "src/components/sidebar.py": "# Componenti della sidebar\n",
        "src/components/charts.py": "# Componenti per i grafici\n",
        "src/components/metrics.py": "# Componenti per le metriche\n",
        "src/components/tables.py": "# Componenti per le tabelle\n",
        
        # Utilities
        "src/utils/__init__.py": "",
        "src/utils/data_loader.py": "# Funzioni per caricare i dati\n",
        "src/utils/data_processor.py": "# Funzioni per processare i dati\n",
        "src/utils/helpers.py": "# Funzioni di utilità\n",
        
        # Modelli ML
        "src/models/__init__.py": "",
        "src/models/ml_models.py": "# Modelli di machine learning\n",
        "src/models/model_trainer.py": "# Training dei modelli\n",
        "src/models/model_evaluator.py": "# Valutazione dei modelli\n",
        
        # Pagine multiple
        "pages/1_Data_Analysis.py": "# Pagina di analisi dati\n",
        "pages/2_Machine_Learning.py": "# Pagina machine learning\n",
        "pages/3_Predictions.py": "# Pagina predizioni\n",
        "pages/4_About.py": "# Pagina informazioni\n",
        
        # Stili CSS
        "assets/styles/main.css": """/* Stili CSS personalizzati per l'app */
.main-header {
    text-align: center;
    color: #FF6B6B;
}

.metric-card {
    background-color: #F0F2F6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #FF6B6B;
}
""",
        
        # Test
        "tests/__init__.py": "",
        "tests/test_data_processing.py": "# Test per il processing dei dati\n",
        "tests/test_models.py": "# Test per i modelli\n",
        
        # Documentazione
        "docs/README.md": """# Titanic Survival Prediction App

Applicazione Streamlit per l'analisi e predizione della sopravvivenza sul Titanic.

## Struttura del Progetto

- `app.py`: File principale dell'applicazione
- `src/`: Codice sorgente modulare
- `pages/`: Pagine multiple dell'app
- `assets/`: Risorse statiche (CSS, immagini)
- `tests/`: Test unitari
""",
        
        # File di ambiente
        ".env.example": """# Esempio di file di environment
DEBUG=True
DATA_PATH=src/data/
MODEL_PATH=src/models/saved/
""",
        
        # Git
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Streamlit
.streamlit/secrets.toml

# Environment variables
.env

# Data files (se sono grandi)
*.csv
*.xlsx
*.json

# Jupyter Notebook
.ipynb_checkpoints

# Models
*.pkl
*.joblib
""",
        

    }
    
    # Creazione dei file
    print("\nCreazione dei file...")
    for file_path, content in files_to_create.items():
        full_path = base_dir / file_path
        
        # Se il file esiste già, non sovrascriverlo
        if full_path.exists():
            print(f"File già esistente: {file_path}")
            continue
            
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"File creato: {file_path}")
    
    # Spostamento dei file esistenti nelle posizioni corrette
    print("\nSpostamento dei file esistenti...")
    
    # Mappatura dei file da spostare
    files_to_move = {
        "main.ipynb": "docs/main.ipynb",  # Sposta il notebook nella docs come riferimento
        "data_titanic.csv": "src/data/data_titanic.csv",  # Sposta i dati nella cartella data
        "config.py": "src/config.py"  # Sposta la config nella src
    }
    
    for source, destination in files_to_move.items():
        source_path = base_dir / source
        dest_path = base_dir / destination
        
        if source_path.exists():
            # Se la destinazione esiste già, rinomina con suffisso
            if dest_path.exists():
                counter = 1
                while dest_path.exists():
                    name_parts = dest_path.name.split('.')
                    if len(name_parts) > 1:
                        new_name = f"{'.'.join(name_parts[:-1])}_{counter}.{name_parts[-1]}"
                    else:
                        new_name = f"{dest_path.name}_{counter}"
                    dest_path = dest_path.parent / new_name
                    counter += 1
            
            shutil.move(str(source_path), str(dest_path))
            print(f"File spostato: {source} → {destination}")
        else:
            print(f"File non trovato: {source}")
    
    # Creazione di un file README principale
    readme_content = """# Titanic Survival Prediction App

Una moderna applicazione Streamlit per l'analisi e la predizione della sopravvivenza sul Titanic.

## Quick Start

1. Attiva il virtual environment:
   ```bash
   .\\venv\\Scripts\\activate  # Windows
   source venv/bin/activate   # Linux/Mac
   ```

2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

3. Esegui l'applicazione:
   ```bash
   streamlit run app.py
   ```

## Struttura del Progetto

```
Titanic_Claude/
├── app.py                      # File principale dell'app
├── Home.py                     # Homepage (multipage app)
├── src/                        # Codice sorgente
│   ├── components/             # Componenti riusabili
│   ├── data/                   # Dati del progetto
│   ├── models/                 # Modelli ML
│   ├── utils/                  # Utilities
│   └── config.py              # Configurazioni
├── pages/                      # Pagine multiple
├── assets/                     # Risorse statiche
├── tests/                      # Test unitari
├── docs/                       # Documentazione
└── .streamlit/                 # Configurazione Streamlit
```

## Tecnologie Utilizzate

- **Streamlit**: Framework per l'app web
- **Pandas**: Manipolazione dati
- **Scikit-learn**: Machine Learning
- **Matplotlib/Seaborn**: Visualizzazioni
- **Plotly**: Grafici interattivi

## Funzionalità

- Analisi esplorativa dei dati
- Visualizzazioni interattive
- Modelli di machine learning
- Predizioni in tempo reale
- Dashboard con metriche
- Interfaccia utente moderna

## Sviluppo

Il progetto è stato migrato da un Jupyter Notebook (`docs/main.ipynb`) a un'applicazione Streamlit modulare e scalabile.
"""
    
    with open(base_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("README.md principale creato")
    
    print("\nStruttura del progetto creata con successo!")
    print("\nProssimi passi:")
    print("1. Esegui lo script: python structure.py")
    print("2. Rivedi il notebook in docs/main.ipynb")
    print("3. Inizia a sviluppare l'app in app.py")
    print("4. Testa l'app con: streamlit run app.py")

    return True

if __name__ == "__main__":
    create_project_structure()