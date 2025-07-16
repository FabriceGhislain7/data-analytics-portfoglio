"""
Script per creare struttura progetto Streamlit Titanic
Crea solo file e cartelle vuote
"""

import os

def create_folder(folder_name):
    """Crea una cartella se non esiste"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Cartella creata: {folder_name}")
    else:
        print(f"Cartella esistente: {folder_name}")

def create_empty_file(file_path):
    """Crea un file vuoto"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("")  # File vuoto
    print(f"File vuoto creato: {file_path}")

def main():
    print("Creazione struttura progetto Titanic")
    print("=" * 40)
    print()

    # CREA CARTELLE
    print("Creazione cartelle...")
    folders = [
        "data",
        "pages", 
        "utils",
        "assets",
        "models"
    ]

    for folder in folders:
        create_folder(folder)

    print()

    # CREA FILE VUOTI
    print("Creazione file vuoti...")
    files = [
        "main.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "config.py",
        "utils/__init__.py",
        "utils/data_loader.py",
        "utils/visualizations.py",
        "pages/01_EDA.py",
        "pages/02_ML_Models.py",
        "pages/03_Predictions.py",
        "models/.gitkeep",
        "data/.gitkeep",
        "assets/.gitkeep"
    ]

    for file_path in files:
        create_empty_file(file_path)

    print()
    print("STRUTTURA CREATA!")
    print("=" * 40)
    print()

    # Mostra struttura
    print("Struttura finale:")
    for root, dirs, files in os.walk('.'):
        # Ignora venv e __pycache__
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git']]

        level = root.replace('.', '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")

        subindent = '  ' * (level + 1)
        for file in files:
            if not file.startswith('.') and file != 'structure.py':
                print(f"{subindent}{file}")

    print()
    print("File principali da modificare:")
    print("   main.py - Dashboard principale")
    print("   requirements.txt - Dipendenze")
    print("   utils/data_loader.py - Caricamento dati")
    print("   pages/01_EDA.py - Analisi esplorativa")
    print()
    print("Ora puoi iniziare a scrivere il codice!")

if __name__ == "__main__":
    main()