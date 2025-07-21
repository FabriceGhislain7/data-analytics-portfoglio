# Titanic Survival Prediction App

Una moderna applicazione Streamlit per l'analisi e la predizione della sopravvivenza sul Titanic.

## Quick Start

1. Attiva il virtual environment:
   ```bash
   .\venv\Scripts\activate  # Windows
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
