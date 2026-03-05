# src/evaluation/walk_forward.py
# Validation walk-forward pour séries temporelles

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from ..evaluation.metrics import compute_metrics

class WalkForwardValidator:
    """Validation walk-forward pour séries temporelles"""
    
    def __init__(self, n_splits: int = 5, train_size: float = 0.7, 
                 gap: int = 0, forecast_horizon: int = 1):
        """
        Initialise le validateur walk-forward
        
        Parameters:
        -----------
        n_splits : int
            Nombre de fenêtres de validation
        train_size : float
            Proportion d'entraînement (0.0-1.0)
        gap : int
            Période entre fin train et début test
        forecast_horizon : int
            Horizon de prévision H
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.gap = gap
        self.forecast_horizon = forecast_horizon
        self.results = []
    
    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple]:
        """
        Génère les indices pour walk-forward
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        
        Returns:
        --------
        List[Tuple]
            Liste de (train_idx, test_idx)
        """
        n = len(X)
        indices = np.arange(n)
        
        # Taille de la fenêtre d'entraînement
        train_window = int(n * self.train_size)
        
        # Taille de la fenêtre de test
        test_window = (n - train_window) // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Indices d'entraînement
            train_end = train_window + i * test_window
            train_idx = indices[:train_end]
            
            # Indices de test (après le gap)
            test_start = train_end + self.gap
            test_end = min(test_start + test_window, n)
            test_idx = indices[test_start:test_end]
            
            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    def validate(self, model, X: pd.DataFrame, y: pd.Series,
                 feature_names: list = None) -> Dict[str, Any]:
        """
        Exécute la validation walk-forward
        
        Parameters:
        -----------
        model : BaseModel
            Modèle à évaluer
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        feature_names : list
            Noms des features pour importance
        
        Returns:
        --------
        Dict
            Résultats de la validation
        """
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD VALIDATION - {model.name}")
        print(f"Fenêtres: {self.n_splits} | Train: {self.train_size*100:.0f}%")
        print(f"{'='*60}")
        
        splits = self.split(X, y)
        
        all_predictions = []
        all_true = []
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n--- Fold {fold}/{len(splits)} ---")
            
            # Split
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            print(f"Train: {X_train.index[0].date()} -> {X_train.index[-1].date()} "
                  f"({len(X_train)} obs)")
            print(f"Test : {X_test.index[0].date()} -> {X_test.index[-1].date()} "
                  f"({len(X_test)} obs)")
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédiction
            y_pred = model.predict(X_test)
            
            # Métriques
            metrics = compute_metrics(y_test.values, y_pred)
            metrics['fold'] = fold
            metrics['train_start'] = X_train.index[0]
            metrics['train_end'] = X_train.index[-1]
            metrics['test_start'] = X_test.index[0]
            metrics['test_end'] = X_test.index[-1]
            
            fold_metrics.append(metrics)
            
            print(f"MAE: {metrics['MAE']:.6f} | RMSE: {metrics['RMSE']:.6f} | "
                  f"MAPE: {metrics['MAPE']:.2f}%")
            
            # Stocker pour courbe de prédiction
            all_predictions.extend(y_pred)
            all_true.extend(y_test.values)
        
        # Métriques globales
        global_metrics = compute_metrics(all_true, all_predictions)
        
        results = {
            'model_name': model.name,
            'fold_metrics': fold_metrics,
            'global_metrics': global_metrics,
            'predictions': np.array(all_predictions),
            'true_values': np.array(all_true),
            'n_splits': len(splits),
            'forecast_horizon': self.forecast_horizon
        }
        
        # Feature importance
        if hasattr(model, 'get_feature_importance') and feature_names is not None:
            try:
                importance = model.get_feature_importance(feature_names)
                results['feature_importance'] = importance
            except:
                pass
        
        self.results.append(results)
        
        print(f"\n{'='*60}")
        print(f"RÉSULTATS GLOBAUX - {model.name}")
        print(f"{'='*60}")
        print(f"MAE  : {global_metrics['MAE']:.6f}")
        print(f"RMSE : {global_metrics['RMSE']:.6f}")
        print(f"MAPE : {global_metrics['MAPE']:.2f}%")
        print(f"DIR  : {global_metrics['Direction_Accuracy']:.2f}%")
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """
        Résumé comparatif de tous les modèles validés
        
        Returns:
        --------
        pd.DataFrame
            Tableau comparatif
        """
        summary_data = []
        
        for result in self.results:
            row = {
                'Modèle': result['model_name'],
                'MAE': result['global_metrics']['MAE'],
                'RMSE': result['global_metrics']['RMSE'],
                'MAPE': result['global_metrics']['MAPE'],
                'Direction': result['global_metrics']['Direction_Accuracy'],
                'R²': result['global_metrics']['R2']
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.round({
            'MAE': 6,
            'RMSE': 6,
            'MAPE': 2,
            'Direction': 2,
            'R²': 4
        })
        
        return df.sort_values('RMSE')