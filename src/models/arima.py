# src/models/arima.py
# Modèle ARIMA/SARIMA univarié pour prévision BTC

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

from .base import BaseModel

class ARIMAModel(BaseModel):
    """Modèle ARIMA pour prévision univariée"""
    
    def __init__(self, order=(1,0,0), seasonal_order=None, name="ARIMA"):
        """
        Initialise le modèle ARIMA
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) ordre ARIMA
        seasonal_order : tuple
            (P, D, Q, S) ordre saisonnier (SARIMA)
        name : str
            Nom du modèle
        """
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.params = {
            'order': order,
            'seasonal_order': seasonal_order
        }
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entraîne le modèle ARIMA sur la série cible
        
        Note: ARIMA est univarié, X_train n'est pas utilisé directement
        """
        try:
            if self.seasonal_order:
                self.model = SARIMAX(
                    y_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(
                    y_train,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            print(f"✓ {self.name} entraîné avec succès")
            
        except Exception as e:
            print(f"✗ Erreur entraînement {self.name}: {e}")
            self.is_fitted = False
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Prédit les valeurs futures
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Features de test (non utilisés pour ARIMA pur)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        
        n_forecast = len(X_test)
        forecast = self.fitted_model.forecast(steps=n_forecast)
        
        return forecast.values
    
    def summary(self):
        """Affiche le résumé du modèle"""
        if self.is_fitted:
            print(self.fitted_model.summary())
        else:
            print("Modèle non entraîné")