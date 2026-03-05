# src/evaluation/metrics.py
# Métriques d'évaluation pour prévision

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcule les métriques d'évaluation
    
    Parameters:
    -----------
    y_true : np.ndarray
        Valeurs réelles
    y_pred : np.ndarray
        Valeurs prédites
    
    Returns:
    --------
    dict
        Dictionnaire des métriques
    """
    # Éviter division par zéro
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # MAE - Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # RMSE - Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE - Mean Absolute Percentage Error
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Direction accuracy (signe de la variation)
    y_true_diff = np.diff(y_true) if len(y_true) > 1 else np.array([0])
    y_pred_diff = np.diff(y_pred) if len(y_pred) > 1 else np.array([0])
    
    if len(y_true_diff) > 0:
        direction_accuracy = np.mean((y_true_diff * y_pred_diff) > 0) * 100
    else:
        direction_accuracy = np.nan
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy,
        'R2': r2,
        'n_obs': len(y_true)
    }
    
    return metrics

def print_metrics(metrics: dict, model_name: str = ""):
    """
    Affiche les métriques de façon formatée
    
    Parameters:
    -----------
    metrics : dict
        Dictionnaire des métriques
    model_name : str
        Nom du modèle
    """
    print("\n" + "="*60)
    print(f"MÉTRIQUES D'ÉVALUATION - {model_name}")
    print("="*60)
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"MAPE : {metrics['MAPE']:.2f}%")
    print(f"DIR  : {metrics['Direction_Accuracy']:.2f}%")
    print(f"R²   : {metrics['R2']:.4f}")
    print(f"N    : {metrics['n_obs']}")
    print("="*60)

def compare_models(all_metrics: dict) -> pd.DataFrame:
    """
    Compare les performances de plusieurs modèles
    
    Parameters:
    -----------
    all_metrics : dict
        Dictionnaire {nom_modele: metrics_dict}
    
    Returns:
    --------
    pd.DataFrame
        Tableau comparatif
    """
    comparison = []
    
    for model_name, metrics in all_metrics.items():
        row = {
            'Modèle': model_name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE (%)': metrics['MAPE'],
            'Direction (%)': metrics['Direction_Accuracy'],
            'R²': metrics['R2']
        }
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df = df.round({
        'MAE': 6,
        'RMSE': 6,
        'MAPE (%)': 2,
        'Direction (%)': 2,
        'R²': 4
    })
    
    # Trier par RMSE
    df = df.sort_values('RMSE')
    
    return df