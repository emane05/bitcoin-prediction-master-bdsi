# src/models/random_forest.py
# Modèle Random Forest pour régression

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class RandomForestModel(BaseModel):
    """Modèle Random Forest pour prévision"""
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1,
                 random_state=42, name="RandomForest"):
        """
        Initialise le modèle Random Forest
        
        Parameters:
        -----------
        n_estimators : int
            Nombre d'arbres
        max_depth : int
            Profondeur maximale des arbres
        min_samples_split : int
            Nombre minimum d'échantillons pour split
        min_samples_leaf : int
            Nombre minimum d'échantillons par feuille
        random_state : int
            Graine aléatoire
        name : str
            Nom du modèle
        """
        super().__init__(name)
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }
        
        self.model = RandomForestRegressor(**self.params)
        self.scaler = StandardScaler()
        self.feature_importance_ = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entraîne le modèle Random Forest
        """
        try:
            # Random Forest n'a pas besoin de scaling mais on garde pour homogénéité
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Entraînement
            self.model.fit(X_train_scaled, y_train)
            self.is_fitted = True
            
            # Sauvegarder l'importance des features
            self.feature_importance_ = self.model.feature_importances_
            
            print(f"✓ {self.name} entraîné avec succès")
            print(f"  {self.params['n_estimators']} arbres")
            
            if hasattr(self.model, 'feature_importances_'):
                top_features = np.argsort(self.feature_importance_)[-5:][::-1]
                print("  Top 5 features importantes:")
                for idx in top_features:
                    if idx < len(X_train.columns):
                        print(f"    {X_train.columns[idx]}: {self.feature_importance_[idx]:.4f}")
            
        except Exception as e:
            print(f"✗ Erreur entraînement {self.name}: {e}")
            self.is_fitted = False
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Prédit avec le modèle Random Forest
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de prédire")
        
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        return predictions
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Retourne l'importance des features
        
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
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def optimize_params(self, X_train: pd.DataFrame, y_train: pd.Series,
                        param_grid: dict = None) -> dict:
        """
        Optimisation simple des hyperparamètres (grid search manuelle)
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Features d'entraînement
        y_train : pd.Series
            Cible d'entraînement
        param_grid : dict
            Grille de paramètres à tester
        
        Returns:
        --------
        dict
            Meilleurs paramètres
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        
        best_score = -np.inf
        best_params = self.params.copy()
        
        print("Optimisation des hyperparamètres...")
        
        # Grid search simplifié
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            # Entraîner avec ces paramètres
            model = RandomForestRegressor(random_state=42, **params)
            X_scaled = self.scaler.fit_transform(X_train)
            model.fit(X_scaled, y_train)
            
            # Score sur l'entraînement (validation croisée serait mieux)
            score = model.score(X_scaled, y_train)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        print(f"Meilleurs paramètres: {best_params}")
        print(f"Meilleur score: {best_score:.4f}")
        
        # Mettre à jour le modèle
        self.params.update(best_params)
        self.model = RandomForestRegressor(**self.params)
        
        return best_params