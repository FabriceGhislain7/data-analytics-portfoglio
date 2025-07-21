"""
src/models/model_evaluator.py
Valutazione e analisi performance modelli Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss,
    average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)
# Import sicuro per brier_score_loss
try:
    from sklearn.metrics import brier_score_loss
except ImportError:
    try:
        from sklearn.calibration import brier_score_loss
    except ImportError:
        # Fallback: implementazione custom se non disponibile
        def brier_score_loss(y_true, y_prob):
            return np.mean((y_prob - y_true) ** 2)

from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, permutation_test_score
import scipy.stats as stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.config import EVALUATION_METRICS, ML_MODELS

# ----------------1. Core Evaluation Class

class ModelEvaluator:
    """
    Classe principale per valutazione modelli
    """
    
    def __init__(self, models_dict, X_test, y_test):
        """
        Args:
            models_dict: Dizionario {nome_modello: modello_addestrato}
            X_test: Features test set
            y_test: Target test set
        """
        self.models = models_dict
        self.X_test = X_test
        self.y_test = y_test
        self.evaluation_results = {}
        self.predictions = {}
        self.probabilities = {}
        
    def evaluate_all_models(self, detailed=True):
        """
        Valuta tutti i modelli disponibili
        
        Args:
            detailed: Se includere metriche dettagliate
        
        Returns:
            Dizionario con risultati valutazione
        """
        for model_name, model_data in self.models.items():
            self.evaluation_results[model_name] = self.evaluate_single_model(
                model_name, model_data, detailed=detailed
            )
        
        return self.evaluation_results
    
    def evaluate_single_model(self, model_name, model_data, detailed=True):
        """
        Valuta singolo modello
        
        Args:
            model_name: Nome del modello
            model_data: Dati del modello (model + preprocessor)
            detailed: Se includere analisi dettagliate
        
        Returns:
            Risultati valutazione del modello
        """
        model = model_data['model']
        preprocessor = model_data.get('preprocessor')
        
        # Preprocessing se necessario
        if preprocessor:
            X_test_processed = preprocessor.transform(self.X_test)
        else:
            X_test_processed = self.X_test
        
        # Predizioni
        y_pred = model.predict(X_test_processed)
        self.predictions[model_name] = y_pred
        
        # Probabilità (se disponibili)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                self.probabilities[model_name] = y_pred_proba
            except:
                pass
        
        # Metriche base
        results = self._calculate_basic_metrics(y_pred, y_pred_proba)
        
        # Metriche dettagliate
        if detailed:
            results.update(self._calculate_detailed_metrics(y_pred, y_pred_proba))
            results.update(self._calculate_confusion_matrix_metrics(y_pred))
            
            if y_pred_proba is not None:
                results.update(self._calculate_probability_metrics(y_pred_proba))
        
        # Informazioni modello
        results['model_info'] = self._get_model_info(model_name, model)
        
        return results
    
    def _calculate_basic_metrics(self, y_pred, y_pred_proba):
        """Calcola metriche base"""
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(self.y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(self.y_test, y_pred_proba)
        
        return metrics
    
    def _calculate_detailed_metrics(self, y_pred, y_pred_proba):
        """Calcola metriche dettagliate"""
        metrics = {
            'matthews_corrcoef': matthews_corrcoef(self.y_test, y_pred),
            'cohen_kappa': cohen_kappa_score(self.y_test, y_pred),
            'specificity': self._calculate_specificity(y_pred),
            'sensitivity': recall_score(self.y_test, y_pred, zero_division=0),  # Alias per recall
            'npv': self._calculate_npv(y_pred),  # Negative Predictive Value
            'fallout': self._calculate_fallout(y_pred)  # False Positive Rate
        }
        
        if y_pred_proba is not None:
            metrics['log_loss'] = log_loss(self.y_test, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(self.y_test, y_pred_proba)
        
        return metrics
    
    def _calculate_confusion_matrix_metrics(self, y_pred):
        """Calcola metriche da confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_errors': int(fp + fn),
            'error_rate': (fp + fn) / len(self.y_test)
        }
    
    def _calculate_probability_metrics(self, y_pred_proba):
        """Calcola metriche basate su probabilità"""
        # Calibration
        fraction_positives, mean_predicted_value = calibration_curve(
            self.y_test, y_pred_proba, n_bins=10
        )
        
        # ECE (Expected Calibration Error)
        ece = np.mean(np.abs(fraction_positives - mean_predicted_value))
        
        return {
            'calibration_curve': {
                'fraction_positives': fraction_positives,
                'mean_predicted_value': mean_predicted_value
            },
            'expected_calibration_error': ece
        }
    
    def _calculate_specificity(self, y_pred):
        """Calcola specificity (True Negative Rate)"""
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_npv(self, y_pred):
        """Calcola Negative Predictive Value"""
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0
    
    def _calculate_fallout(self, y_pred):
        """Calcola False Positive Rate"""
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def _get_model_info(self, model_name, model):
        """Ottieni informazioni sul modello"""
        return {
            'name': ML_MODELS.get(model_name, {}).get('name', model_name),
            'type': model_name,
            'n_features': getattr(model, 'n_features_in_', 'Unknown'),
            'classes': getattr(model, 'classes_', ['Unknown'])
        }

# ----------------2. Model Comparison

class ModelComparison:
    """
    Classe per confronto tra modelli
    """
    
    def __init__(self, evaluation_results):
        self.results = evaluation_results
        
    def create_comparison_table(self, metrics=None):
        """
        Crea tabella di confronto
        
        Args:
            metrics: Lista metriche da includere
        
        Returns:
            DataFrame con confronto
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': ML_MODELS.get(model_name, {}).get('name', model_name)}
            
            for metric in metrics:
                value = results.get(metric, np.nan)
                if not np.isnan(value):
                    row[metric.title()] = round(value, 4)
                else:
                    row[metric.title()] = 'N/A'
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def rank_models(self, metric='f1', ascending=False):
        """
        Classifica modelli per metrica
        
        Args:
            metric: Metrica per ranking
            ascending: Ordine crescente
        
        Returns:
            Lista ordinata di modelli
        """
        rankings = []
        
        for model_name, results in self.results.items():
            score = results.get(metric, 0)
            rankings.append({
                'model': model_name,
                'name': ML_MODELS.get(model_name, {}).get('name', model_name),
                'score': score
            })
        
        return sorted(rankings, key=lambda x: x['score'], reverse=not ascending)
    
    def find_best_model(self, metric='f1'):
        """
        Trova miglior modello per metrica
        
        Args:
            metric: Metrica di riferimento
        
        Returns:
            Dizionario con info miglior modello
        """
        rankings = self.rank_models(metric)
        if rankings:
            best = rankings[0]
            return {
                'model_type': best['model'],
                'model_name': best['name'],
                'score': best['score'],
                'metric': metric
            }
        return None
    
    def calculate_model_consensus(self):
        """
        Calcola consensus tra modelli per ogni predizione
        
        Returns:
            Analisi consensus
        """
        if not hasattr(self, 'evaluator'):
            return None
        
        # Raccoglie tutte le predizioni
        all_predictions = []
        model_names = []
        
        for model_name in self.results.keys():
            if model_name in self.evaluator.predictions:
                all_predictions.append(self.evaluator.predictions[model_name])
                model_names.append(model_name)
        
        if not all_predictions:
            return None
        
        predictions_matrix = np.array(all_predictions).T
        
        # Calcola consensus per ogni sample
        consensus_scores = []
        for i in range(len(predictions_matrix)):
            votes = predictions_matrix[i]
            consensus = np.mean(votes)
            consensus_scores.append(consensus)
        
        return {
            'consensus_scores': np.array(consensus_scores),
            'high_consensus': np.sum((np.array(consensus_scores) >= 0.8) | (np.array(consensus_scores) <= 0.2)),
            'low_consensus': np.sum((np.array(consensus_scores) > 0.2) & (np.array(consensus_scores) < 0.8)),
            'model_names': model_names
        }

# ----------------3. Statistical Tests

class StatisticalTests:
    """
    Test statistici per confronto modelli
    """
    
    @staticmethod
    def mcnemar_test(y_true, y_pred1, y_pred2):
        """
        Test di McNemar per confronto di due modelli
        
        Args:
            y_true: Valori veri
            y_pred1: Predizioni modello 1
            y_pred2: Predizioni modello 2
        
        Returns:
            Risultati test McNemar
        """
        from statsmodels.stats.contingency_tables import mcnemar
        
        # Crea tabella di contingenza
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # Conta casi
        both_correct = np.sum(correct1 & correct2)
        model1_correct = np.sum(correct1 & ~correct2)
        model2_correct = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        # Matrice di contingenza
        contingency_table = np.array([[both_correct, model1_correct],
                                     [model2_correct, both_wrong]])
        
        # Test McNemar
        try:
            result = mcnemar(contingency_table, exact=True)
            return {
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'contingency_table': contingency_table,
                'significant': result.pvalue < 0.05
            }
        except:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'contingency_table': contingency_table,
                'significant': False
            }
    
    @staticmethod
    def paired_t_test(scores1, scores2):
        """
        T-test appaiato per confronto cross-validation scores
        
        Args:
            scores1: Scores modello 1
            scores2: Scores modello 2
        
        Returns:
            Risultati t-test
        """
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'mean_diff': np.mean(scores1 - scores2),
            'significant': p_value < 0.05,
            'better_model': 1 if np.mean(scores1) > np.mean(scores2) else 2
        }
    
    @staticmethod
    def wilcoxon_test(scores1, scores2):
        """
        Test di Wilcoxon per confronto non-parametrico
        
        Args:
            scores1: Scores modello 1
            scores2: Scores modello 2
        
        Returns:
            Risultati test Wilcoxon
        """
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'better_model': 1 if np.median(scores1) > np.median(scores2) else 2
        }

# ----------------4. Error Analysis

class ErrorAnalysis:
    """
    Analisi dettagliata degli errori
    """
    
    def __init__(self, X_test, y_test, predictions, feature_names=None):
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions
        self.feature_names = feature_names or X_test.columns.tolist()
        
    def analyze_prediction_errors(self, model_name):
        """
        Analizza errori di predizione per singolo modello
        
        Args:
            model_name: Nome del modello
        
        Returns:
            Analisi errori
        """
        if model_name not in self.predictions:
            return None
        
        y_pred = self.predictions[model_name]
        
        # Identifica errori
        errors = y_pred != self.y_test
        false_positives = (y_pred == 1) & (self.y_test == 0)
        false_negatives = (y_pred == 0) & (self.y_test == 1)
        
        # Analisi per feature
        error_analysis = {}
        
        for feature in self.feature_names:
            if feature in self.X_test.columns:
                feature_values = self.X_test[feature]
                
                error_analysis[feature] = {
                    'total_errors': np.sum(errors),
                    'fp_mean': np.mean(feature_values[false_positives]) if np.sum(false_positives) > 0 else np.nan,
                    'fn_mean': np.mean(feature_values[false_negatives]) if np.sum(false_negatives) > 0 else np.nan,
                    'correct_mean': np.mean(feature_values[~errors]) if np.sum(~errors) > 0 else np.nan,
                    'error_correlation': np.corrcoef(feature_values, errors.astype(int))[0,1] if len(np.unique(feature_values)) > 1 else 0
                }
        
        return {
            'total_errors': np.sum(errors),
            'false_positives': np.sum(false_positives),
            'false_negatives': np.sum(false_negatives),
            'error_rate': np.mean(errors),
            'feature_analysis': error_analysis,
            'error_indices': np.where(errors)[0].tolist()
        }
    
    def find_difficult_samples(self, threshold=0.5):
        """
        Trova campioni difficili da classificare
        
        Args:
            threshold: Soglia per definire campioni difficili
        
        Returns:
            Analisi campioni difficili
        """
        # Conta errori per campione
        error_counts = np.zeros(len(self.y_test))
        
        for model_name, y_pred in self.predictions.items():
            errors = (y_pred != self.y_test).astype(int)
            error_counts += errors
        
        # Campioni difficili
        n_models = len(self.predictions)
        difficult_mask = error_counts >= (n_models * threshold)
        
        difficult_indices = np.where(difficult_mask)[0]
        
        if len(difficult_indices) == 0:
            return {'difficult_samples': 0, 'indices': []}
        
        # Analisi campioni difficili
        difficult_samples = self.X_test.iloc[difficult_indices]
        difficult_targets = self.y_test.iloc[difficult_indices]
        
        return {
            'difficult_samples': len(difficult_indices),
            'indices': difficult_indices.tolist(),
            'percentage': len(difficult_indices) / len(self.y_test) * 100,
            'sample_data': difficult_samples,
            'target_distribution': difficult_targets.value_counts().to_dict(),
            'error_counts': error_counts[difficult_indices]
        }
    
    def analyze_misclassification_patterns(self):
        """
        Analizza pattern di misclassificazione
        
        Returns:
            Pattern comuni di errore
        """
        patterns = {}
        
        for model_name, y_pred in self.predictions.items():
            # Pattern per questo modello
            fp_indices = np.where((y_pred == 1) & (self.y_test == 0))[0]
            fn_indices = np.where((y_pred == 0) & (self.y_test == 1))[0]
            
            patterns[model_name] = {
                'false_positive_patterns': self._analyze_feature_patterns(fp_indices),
                'false_negative_patterns': self._analyze_feature_patterns(fn_indices)
            }
        
        return patterns
    
    def _analyze_feature_patterns(self, error_indices):
        """Analizza pattern nelle feature per errori specifici"""
        if len(error_indices) == 0:
            return {}
        
        error_samples = self.X_test.iloc[error_indices]
        patterns = {}
        
        for feature in self.feature_names:
            if feature in error_samples.columns:
                values = error_samples[feature]
                
                patterns[feature] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'most_common': values.mode().iloc[0] if not values.mode().empty else np.nan
                }
        
        return patterns

# ----------------5. Performance Monitoring

class PerformanceMonitor:
    """
    Monitoraggio performance nel tempo
    """
    
    def __init__(self):
        self.performance_history = []
        
    def log_performance(self, model_name, metrics, timestamp=None):
        """
        Registra performance per monitoraggio
        
        Args:
            model_name: Nome modello
            metrics: Dizionario metriche
            timestamp: Timestamp (default: ora corrente)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'metrics': metrics.copy()
        }
        
        self.performance_history.append(entry)
    
    def get_performance_trend(self, model_name, metric='accuracy', periods=10):
        """
        Ottieni trend performance
        
        Args:
            model_name: Nome modello
            metric: Metrica da analizzare
            periods: Numero di periodi
        
        Returns:
            Trend analysis
        """
        model_history = [
            entry for entry in self.performance_history[-periods:]
            if entry['model_name'] == model_name
        ]
        
        if len(model_history) < 2:
            return None
        
        timestamps = [entry['timestamp'] for entry in model_history]
        values = [entry['metrics'].get(metric, np.nan) for entry in model_history]
        
        # Calcola trend
        if len(values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(values)), values
            )
        else:
            slope = 0
        
        return {
            'timestamps': timestamps,
            'values': values,
            'trend_slope': slope,
            'is_improving': slope > 0,
            'latest_value': values[-1] if values else np.nan,
            'change_from_first': values[-1] - values[0] if len(values) > 1 else 0
        }
    
    def detect_performance_degradation(self, model_name, metric='accuracy', threshold=0.05):
        """
        Rileva degradazione performance
        
        Args:
            model_name: Nome modello
            metric: Metrica da monitorare
            threshold: Soglia di degradazione
        
        Returns:
            Alert se degradazione rilevata
        """
        trend = self.get_performance_trend(model_name, metric)
        
        if trend is None:
            return None
        
        degradation_detected = (
            trend['trend_slope'] < -threshold or
            trend['change_from_first'] < -threshold
        )
        
        return {
            'degradation_detected': degradation_detected,
            'trend_slope': trend['trend_slope'],
            'change_from_baseline': trend['change_from_first'],
            'latest_value': trend['latest_value'],
            'recommendation': 'Retraining suggested' if degradation_detected else 'Performance stable'
        }

# ----------------6. Model Interpretability Metrics

class InterpretabilityMetrics:
    """
    Metriche per interpretabilità modelli
    """
    
    @staticmethod
    def calculate_feature_stability(feature_importances_history):
        """
        Calcola stabilità feature importance nel tempo
        
        Args:
            feature_importances_history: Lista di dizionari feature importance
        
        Returns:
            Score di stabilità per feature
        """
        if len(feature_importances_history) < 2:
            return {}
        
        # Converti in DataFrame
        df = pd.DataFrame(feature_importances_history)
        
        stability_scores = {}
        for feature in df.columns:
            values = df[feature].values
            # Calcola coefficiente di variazione
            cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
            stability_scores[feature] = 1 / (1 + cv)  # Score tra 0 e 1
        
        return stability_scores
    
    @staticmethod
    def calculate_prediction_confidence_distribution(probabilities):
        """
        Analizza distribuzione confidence delle predizioni
        
        Args:
            probabilities: Array di probabilità
        
        Returns:
            Analisi distribuzione confidence
        """
        # Converte probabilità in confidence (distanza da 0.5)
        confidence = np.abs(probabilities - 0.5) * 2
        
        return {
            'mean_confidence': np.mean(confidence),
            'std_confidence': np.std(confidence),
            'low_confidence_ratio': np.sum(confidence < 0.6) / len(confidence),
            'high_confidence_ratio': np.sum(confidence > 0.8) / len(confidence),
            'confidence_percentiles': {
                '25th': np.percentile(confidence, 25),
                '50th': np.percentile(confidence, 50),
                '75th': np.percentile(confidence, 75),
                '90th': np.percentile(confidence, 90)
            }
        }

# ----------------7. Evaluation Report Generator

class EvaluationReportGenerator:
    """
    Generatore di report di valutazione
    """
    
    def __init__(self, evaluation_results):
        self.results = evaluation_results
        
    def generate_summary_report(self):
        """
        Genera report riassuntivo
        
        Returns:
            Report strutturato
        """
        comparison = ModelComparison(self.results)
        best_model = comparison.find_best_model('f1')
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': len(self.results),
            'best_model': best_model,
            'model_rankings': {
                'accuracy': comparison.rank_models('accuracy')[:3],
                'precision': comparison.rank_models('precision')[:3],
                'recall': comparison.rank_models('recall')[:3],
                'f1': comparison.rank_models('f1')[:3]
            },
            'performance_summary': self._create_performance_summary()
        }
        
        return report
    
    def _create_performance_summary(self):
        """Crea riassunto performance"""
        summary = {}
        
        for model_name, results in self.results.items():
            summary[model_name] = {
                'accuracy': results.get('accuracy', 0),
                'f1': results.get('f1', 0),
                'roc_auc': results.get('roc_auc', 0),
                'key_strengths': self._identify_model_strengths(results),
                'key_weaknesses': self._identify_model_weaknesses(results)
            }
        
        return summary
    
    def _identify_model_strengths(self, results):
        """Identifica punti di forza del modello"""
        strengths = []
        
        if results.get('precision', 0) > 0.8:
            strengths.append('High Precision')
        if results.get('recall', 0) > 0.8:
            strengths.append('High Recall')
        if results.get('roc_auc', 0) > 0.9:
            strengths.append('Excellent ROC-AUC')
        if results.get('balanced_accuracy', 0) > 0.8:
            strengths.append('Balanced Performance')
        
        return strengths
    
    def _identify_model_weaknesses(self, results):
        """Identifica punti deboli del modello"""
        weaknesses = []
        
        if results.get('precision', 1) < 0.7:
            weaknesses.append('Low Precision')
        if results.get('recall', 1) < 0.7:
            weaknesses.append('Low Recall')
        if results.get('roc_auc', 1) < 0.8:
            weaknesses.append('Poor ROC-AUC')
        if results.get('false_positives', 0) > results.get('true_positives', 1):
            weaknesses.append('High False Positive Rate')
        
        return weaknesses