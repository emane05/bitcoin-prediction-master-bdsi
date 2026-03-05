# src/models/base.py
# Classe abstraite de base pour tous les modèles

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

class BaseModel(ABC):
    """Classe abstraite définissant l'interface commune à tous les modèles"""
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialise le modèle de base
        
        Parameters:
        -----------
        name : str
            Nom du modèle
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.params = {}
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entraîne le modèle
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Features d'entraînement
        y_train : pd.Series
            Cible d'entraînement
        """
        pass
    
    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Prédit avec le modèle entraîné
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Features de test
        
        Returns:
        --------
        np.ndarray
            Prédictions
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle"""
        return self.params
    
    def set_params(self, **params) -> None:
        """Définit les paramètres du modèle"""
        self.params.update(params)
    
    def __str__(self) -> str:
        status = "entraîné" if self.is_fitted else "non entraîné"
        return f"{self.name} ({status})"