# src/models/var.py
# Modèle VAR (Vector AutoRegression) multivarié

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

from .base import BaseModel

class VARModel(BaseModel):
    """Modèle VAR pour prévision multivariée"""
    
    def __init__(self, maxlags=10, ic='aic', name="VAR"):
        """
        Initialise le modèle VAR
        
        Parameters:
        -----------
        maxlags : int
            Nombre maximal de lags à considérer
        ic : str
            Critère d'information ('aic', 'bic', 'hqic')
        name : str
            Nom du modèle
        """
        super().__init__(name)
        self.maxlags = maxlags
        self.ic = ic
        self.params = {
            'maxlags': maxlags,
            'ic': ic
        }
        self.selected_lags = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entraîne le modèle VAR
        
        VAR utilise toutes les variables, on combine X_train et y_train
        """
        try:
            # Combiner features et cible pour VAR
            data = X_train.copy()
            data['target'] = y_train.values
            
            # Créer et entraîner le modèle VAR
            self.model = VAR(data)
            
            # Sélection automatique du nombre de lags
            self.lag_selection = self.model.select_order(maxlags=self.maxlags)
            self.selected_lags = getattr(self.lag_selection, self.ic)
            
            if self.selected_lags is None:
                self.selected_lags = 1  # fallback
            
            # Entraîner avec les lags sélectionnés
            self.fitted_model = self.model.fit(self.selected_lags)
            self.is_fitted = True
            
            print(f"✓ {self.name} entraîné avec {self.selected_lags} lags")
            
        except Exception as e:
            print(f"✗ Erreur entraînement {self.name}: {e}")
            self.is_fitted = False
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Prédit avec le modèle VAR
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        
        n_forecast = len(X_test)
        forecast = self.fitted_model.forecast(
            self.fitted_model.endog[-self.selected_lags:],
            steps=n_forecast
        )
        
        # Dernière colonne = target (BTC)
        return forecast[:, -1]
    
    def test_granger_causality(self, data: pd.DataFrame, target_col: str, 
                              variables: list, maxlag: int = 5) -> pd.DataFrame:
        """
        Test de causalité de Granger entre BTC et autres variables
        
        Parameters:
        -----------
        data : pd.DataFrame
            Données avec retours
        target_col : str
            Colonne cible (BTC)
        variables : list
            Liste des variables à tester
        maxlag : int
            Nombre maximal de lags pour le test
        
        Returns:
        --------
        pd.DataFrame
            Résultats des tests de Granger
        """
        results = []
        
        for var in variables:
            if var == target_col:
                continue
                
            test_data = data[[target_col, var]].dropna()
            
            try:
                gc_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                
                for lag in range(1, maxlag + 1):
                    p_value = gc_result[lag][0]['ssr_chi2test'][1]
                    results.append({
                        'cause': var,
                        'effect': target_col,
                        'lag': lag,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
            except Exception as e:
                print(f"Erreur test Granger {var} -> {target_col}: {e}")
        
        return pd.DataFrame(results)
    
    def summary(self):
        """Affiche le résumé du modèle"""
        if self.is_fitted:
            print(self.fitted_model.summary())
            print(f"\nLags sélectionnés ({self.ic}): {self.selected_lags}")
        else:
            print("Modèle non entraîné")