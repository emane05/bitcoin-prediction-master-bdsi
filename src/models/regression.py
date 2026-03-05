# src/models/regression.py
# Modèles de régression supervisée sur features lags

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .base import BaseModel

class RegressionModel(BaseModel):
    """Modèle de régression (Linéaire, Ridge, Lasso)"""
    
    def __init__(self, model_type='linear', alpha=1.0, name=None):
        """
        Initialise le modèle de régression
        
        Parameters:
        -----------
        model_type : str
            Type de régression ('linear', 'ridge', 'lasso')
        alpha : float
            Paramètre de régularisation pour Ridge/Lasso
        name : str
            Nom du modèle
        """
        if name is None:
            name = f"{model_type.capitalize()}Regression"
        
        super().__init__(name)
        self.model_type = model_type
        self.alpha = alpha
        self.params = {
            'model_type': model_type,
            'alpha': alpha
        }
        self.scaler = StandardScaler()
        
        # Sélection du modèle
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entraîne le modèle de régression
        """
        try:
            # Standardisation des features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Entraînement
            self.model.fit(X_train_scaled, y_train)
            self.is_fitted = True
            
            print(f"✓ {self.name} entraîné avec succès")
            
            # Afficher les coefficients pour les modèles linéaires
            if hasattr(self.model, 'coef_'):
                n_features = len(X_train.columns)
                n_display = min(5, n_features)
                print(f"  Top {n_display} coefficients:")
                
                coefs = pd.DataFrame({
                    'feature': X_train.columns,
                    'coefficient': self.model.coef_
                }).sort_values('coefficient', key=abs, ascending=False)
                
                for i in range(n_display):
                    f = coefs.iloc[i]
                    print(f"    {f['feature']}: {f['coefficient']:.4f}")
                    
        except Exception as e:
            print(f"✗ Erreur entraînement {self.name}: {e}")
            self.is_fitted = False
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Prédit avec le modèle de régression
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        return predictions
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Retourne l'importance des features (coefficients)
        
        Parameters:
        -----------
        feature_names : list
            Noms des features
        
        Returns:
        --------
        pd.DataFrame
            Importance des features
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné")
        
        if not hasattr(self.model, 'coef_'):
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance