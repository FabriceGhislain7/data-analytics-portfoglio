"""
src/models/ml_models.py
Definizioni e configurazioni dei modelli Machine Learning
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from src.config import ML_MODELS

# ----------------1. Base Model Class

class TitanicModel:
    """
    Classe base per tutti i modelli Titanic
    """
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.requires_scaling = False
        self.hyperparameters = kwargs
        
    def get_model_info(self):
        """Restituisce informazioni sul modello"""
        return {
            'type': self.model_type,
            'name': ML_MODELS.get(self.model_type, {}).get('name', self.model_type),
            'requires_scaling': self.requires_scaling,
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters
        }

# ----------------2. Model Factory

class ModelFactory:
    """
    Factory per creare istanze dei modelli ML
    """
    
    @staticmethod
    def create_model(model_type, custom_params=None):
        """
        Crea istanza del modello specificato
        
        Args:
            model_type (str): Tipo di modello da creare
            custom_params (dict): Parametri personalizzati
        
        Returns:
            Istanza del modello richiesto
        """
        # Combina parametri di default con quelli personalizzati
        default_params = ML_MODELS.get(model_type, {}).get('params', {})
        params = {**default_params, **(custom_params or {})}
        
        if model_type == 'LogisticRegression':
            return LogisticRegressionModel(**params)
        elif model_type == 'RandomForestClassifier':
            return RandomForestModel(**params)
        elif model_type == 'GradientBoostingClassifier':
            return GradientBoostingModel(**params)
        elif model_type == 'SVC':
            return SVMModel(**params)
        elif model_type == 'DecisionTreeClassifier':
            return DecisionTreeModel(**params)
        elif model_type == 'GaussianNB':
            return NaiveBayesModel(**params)
        elif model_type == 'KNeighborsClassifier':
            return KNNModel(**params)
        elif model_type == 'ExtraTreesClassifier':
            return ExtraTreesModel(**params)
        elif model_type == 'AdaBoostClassifier':
            return AdaBoostModel(**params)
        elif model_type == 'MLPClassifier':
            return MLPModel(**params)
        else:
            raise ValueError(f"Tipo di modello non supportato: {model_type}")
    
    @staticmethod
    def get_available_models():
        """Restituisce lista dei modelli disponibili"""
        return list(ML_MODELS.keys())
    
    @staticmethod
    def create_ensemble_models(model_types=None):
        """
        Crea un insieme di modelli per ensemble
        
        Args:
            model_types (list): Lista dei tipi di modello da creare
        
        Returns:
            dict: Dizionario con istanze dei modelli
        """
        if model_types is None:
            model_types = ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier']
        
        models = {}
        for model_type in model_types:
            models[model_type] = ModelFactory.create_model(model_type)
        
        return models

# ----------------3. Concrete Model Classes

class LogisticRegressionModel(TitanicModel):
    """Modello Logistic Regression"""
    
    def __init__(self, **kwargs):
        super().__init__('LogisticRegression', **kwargs)
        self.requires_scaling = True
        self.model = LogisticRegression(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce coefficienti come importanza features"""
        if not self.is_trained:
            return None
        
        # Per logistic regression, usiamo valore assoluto dei coefficienti
        importances = np.abs(self.model.coef_[0])
        return dict(zip(self.feature_names, importances))

class RandomForestModel(TitanicModel):
    """Modello Random Forest"""
    
    def __init__(self, **kwargs):
        super().__init__('RandomForestClassifier', **kwargs)
        self.requires_scaling = False
        self.model = RandomForestClassifier(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce feature importance da Random Forest"""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def get_tree_depth_stats(self):
        """Statistiche sulla profondità degli alberi"""
        if not self.is_trained:
            return None
        
        depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        return {
            'mean_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'std_depth': np.std(depths)
        }

class GradientBoostingModel(TitanicModel):
    """Modello Gradient Boosting"""
    
    def __init__(self, **kwargs):
        super().__init__('GradientBoostingClassifier', **kwargs)
        self.requires_scaling = False
        self.model = GradientBoostingClassifier(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce feature importance da Gradient Boosting"""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def get_training_loss(self):
        """Restituisce loss durante training"""
        if not self.is_trained:
            return None
        
        return self.model.train_score_

class SVMModel(TitanicModel):
    """Modello Support Vector Machine"""
    
    def __init__(self, **kwargs):
        super().__init__('SVC', **kwargs)
        self.requires_scaling = True
        # Assicurati che probability=True per ROC
        kwargs['probability'] = kwargs.get('probability', True)
        self.model = SVC(**kwargs)
    
    def get_support_vectors_info(self):
        """Informazioni sui support vectors"""
        if not self.is_trained:
            return None
        
        return {
            'n_support_vectors': self.model.n_support_,
            'support_vectors_': self.model.support_vectors_.shape,
            'dual_coef_shape': self.model.dual_coef_.shape
        }

class DecisionTreeModel(TitanicModel):
    """Modello Decision Tree"""
    
    def __init__(self, **kwargs):
        super().__init__('DecisionTreeClassifier', **kwargs)
        self.requires_scaling = False
        self.model = DecisionTreeClassifier(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce feature importance da Decision Tree"""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def get_tree_info(self):
        """Informazioni sull'albero"""
        if not self.is_trained:
            return None
        
        return {
            'max_depth': self.model.tree_.max_depth,
            'n_nodes': self.model.tree_.node_count,
            'n_leaves': self.model.tree_.n_leaves,
            'n_features': self.model.tree_.n_features
        }

class NaiveBayesModel(TitanicModel):
    """Modello Naive Bayes"""
    
    def __init__(self, **kwargs):
        super().__init__('GaussianNB', **kwargs)
        self.requires_scaling = False
        self.model = GaussianNB(**kwargs)

class KNNModel(TitanicModel):
    """Modello K-Nearest Neighbors"""
    
    def __init__(self, **kwargs):
        super().__init__('KNeighborsClassifier', **kwargs)
        self.requires_scaling = True
        # Default parameters
        kwargs.setdefault('n_neighbors', 5)
        kwargs.setdefault('weights', 'uniform')
        self.model = KNeighborsClassifier(**kwargs)

class ExtraTreesModel(TitanicModel):
    """Modello Extra Trees"""
    
    def __init__(self, **kwargs):
        super().__init__('ExtraTreesClassifier', **kwargs)
        self.requires_scaling = False
        kwargs.setdefault('n_estimators', 100)
        kwargs.setdefault('random_state', 42)
        self.model = ExtraTreesClassifier(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce feature importance da Extra Trees"""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

class AdaBoostModel(TitanicModel):
    """Modello AdaBoost"""
    
    def __init__(self, **kwargs):
        super().__init__('AdaBoostClassifier', **kwargs)
        self.requires_scaling = False
        kwargs.setdefault('n_estimators', 50)
        kwargs.setdefault('random_state', 42)
        self.model = AdaBoostClassifier(**kwargs)
    
    def get_feature_importance(self):
        """Restituisce feature importance da AdaBoost"""
        if not self.is_trained:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

class MLPModel(TitanicModel):
    """Modello Multi-Layer Perceptron"""
    
    def __init__(self, **kwargs):
        super().__init__('MLPClassifier', **kwargs)
        self.requires_scaling = True
        kwargs.setdefault('hidden_layer_sizes', (100,))
        kwargs.setdefault('max_iter', 1000)
        kwargs.setdefault('random_state', 42)
        self.model = MLPClassifier(**kwargs)

# ----------------4. Model Configurations

class ModelConfigurations:
    """
    Configurazioni predefinite per diversi scenari
    """
    
    FAST_TRAINING = {
        'LogisticRegression': {'max_iter': 100},
        'RandomForestClassifier': {'n_estimators': 50},
        'GradientBoostingClassifier': {'n_estimators': 50},
        'DecisionTreeClassifier': {'max_depth': 5},
        'SVC': {'kernel': 'linear', 'max_iter': 100}
    }
    
    HIGH_ACCURACY = {
        'LogisticRegression': {'max_iter': 2000, 'solver': 'lbfgs'},
        'RandomForestClassifier': {'n_estimators': 200, 'max_depth': None},
        'GradientBoostingClassifier': {'n_estimators': 200, 'learning_rate': 0.05},
        'SVC': {'kernel': 'rbf', 'gamma': 'scale', 'C': 10},
        'DecisionTreeClassifier': {'max_depth': None, 'min_samples_split': 2}
    }
    
    INTERPRETABLE = {
        'LogisticRegression': {'penalty': 'l1', 'solver': 'liblinear'},
        'DecisionTreeClassifier': {'max_depth': 5, 'min_samples_leaf': 20},
        'RandomForestClassifier': {'n_estimators': 100, 'max_depth': 10}
    }
    
    @classmethod
    def get_config(cls, config_name):
        """Restituisce configurazione specifica"""
        return getattr(cls, config_name.upper(), {})

# ----------------5. Hyperparameter Grids

class HyperparameterGrids:
    """
    Griglie di iperparametri per GridSearch
    """
    
    LOGISTIC_REGRESSION = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs']
    }
    
    RANDOM_FOREST = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    GRADIENT_BOOSTING = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    SVM = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    DECISION_TREE = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    @classmethod
    def get_grid(cls, model_type):
        """Restituisce griglia per tipo di modello"""
        grid_name = model_type.upper().replace('CLASSIFIER', '')
        return getattr(cls, grid_name, {})

# ----------------6. Model Utilities

def get_model_complexity_score(model):
    """
    Calcola un punteggio di complessità del modello
    """
    if hasattr(model, 'n_estimators'):
        return model.n_estimators / 100
    elif hasattr(model, 'C'):
        return 1 / model.C
    elif hasattr(model, 'max_depth') and model.max_depth:
        return model.max_depth / 10
    else:
        return 0.5  # Default per modelli senza parametri evidenti

def recommend_models_for_dataset_size(n_samples):
    """
    Raccomanda modelli basati sulla dimensione del dataset
    """
    if n_samples < 100:
        return ['LogisticRegression', 'DecisionTreeClassifier', 'GaussianNB']
    elif n_samples < 1000:
        return ['LogisticRegression', 'RandomForestClassifier', 'SVC']
    else:
        return ['RandomForestClassifier', 'GradientBoostingClassifier', 'SVC']

def get_model_training_time_estimate(model_type, n_samples, n_features):
    """
    Stima il tempo di training (relativo)
    """
    base_time = n_samples * n_features / 10000
    
    multipliers = {
        'LogisticRegression': 1.0,
        'DecisionTreeClassifier': 1.5,
        'RandomForestClassifier': 3.0,
        'GradientBoostingClassifier': 4.0,
        'SVC': 5.0,
        'MLPClassifier': 6.0
    }
    
    return base_time * multipliers.get(model_type, 2.0)

# ----------------7. Model Ensemble

class ModelEnsemble:
    """
    Classe per gestire ensemble di modelli
    """
    
    def __init__(self, models):
        self.models = models
        self.weights = None
        
    def set_weights(self, weights):
        """Imposta pesi per voting pesato"""
        self.weights = weights
    
    def predict_ensemble(self, X):
        """Predizione ensemble"""
        predictions = []
        
        for model_name, model in self.models.items():
            if model.is_trained:
                pred = model.model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            return None
        
        # Voting majority
        ensemble_pred = np.round(np.mean(predictions, axis=0))
        return ensemble_pred
    
    def predict_proba_ensemble(self, X):
        """Probabilità ensemble"""
        probabilities = []
        
        for model_name, model in self.models.items():
            if model.is_trained and hasattr(model.model, 'predict_proba'):
                proba = model.model.predict_proba(X)[:, 1]
                probabilities.append(proba)
        
        if not probabilities:
            return None
        
        if self.weights:
            weighted_proba = np.average(probabilities, axis=0, weights=self.weights)
        else:
            weighted_proba = np.mean(probabilities, axis=0)
        
        return weighted_proba