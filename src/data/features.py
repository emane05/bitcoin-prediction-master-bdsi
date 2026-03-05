# src/data/features.py
# Calcul des retours, transformations et lags

import pandas as pd
import numpy as np
import yaml

class FeatureEngineer:
    """Classe pour créer les features techniques"""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le feature engineer
        
        Parameters:
        -----------
        config_path : str
            Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.horizon = self.config.get('forecast_horizon', 1)
        self.max_lags = self.config.get('max_lags', 10)
    
    def compute_returns(self, df, price_col=None, log_returns=True):
        """
        Calcule les retours (simples ou logarithmiques)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame avec les prix
        price_col : str
            Colonne des prix (si None, prend toutes les colonnes Close)
        log_returns : bool
            True = log-retours, False = retours simples
        
        Returns:
        --------
        pd.DataFrame
            DataFrame avec les retours
        """
        df_returns = df.copy()
        
        # Identifier les colonnes de prix
        if price_col:
            price_cols = [price_col]
        else:
            price_cols = [col for col in df.columns if 'Close' in col]
        
        for col in price_cols:
            if log_returns:
                # Log-retour: ln(P_t) - ln(P_{t-1})
                df_returns[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
            else:
                # Retour simple: (P_t - P_{t-1}) / P_{t-1}
                df_returns[f'{col}_return'] = df[col].pct_change()
        
        return df_returns
    
    def make_stationary(self, df, diff_order=1):
        """
        Rend les séries stationnaires par différenciation
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame à transformer
        diff_order : int
            Ordre de différenciation
        
        Returns:
        --------
        pd.DataFrame
            Séries différenciées
        """
        df_stationary = df.copy()
        
        for col in df.columns:
            if 'Close' in col or 'Volume' in col:
                for i in range(diff_order):
                    df_stationary[col] = df_stationary[col].diff()
        
        # Supprimer les NaN générés par la différenciation
        df_stationary = df_stationary.dropna()
        
        return df_stationary
    
    def create_lags(self, df, target_col='BTC_Close', lags=None, features_cols=None):
        """
        Crée les variables retardées pour la modélisation supervisée
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame source
        target_col : str
            Colonne cible à prédire
        lags : list
            Liste des retards à créer (ex: [1,2,3])
        features_cols : list
            Colonnes à utiliser comme features
        
        Returns:
        --------
        pd.DataFrame, pd.Series
            X (features lags), y (cible)
        """
        if lags is None:
            lags = list(range(1, self.max_lags + 1))
        
        if features_cols is None:
            # Par défaut: toutes les colonnes sauf la cible
            features_cols = [col for col in df.columns if col != target_col]
        
        X_list = []
        
        # Création des lags pour chaque feature
        for col in features_cols:
            for lag in lags:
                X_list.append(df[col].shift(lag).rename(f'{col}_lag_{lag}'))
        
        X = pd.concat(X_list, axis=1)
        y = df[target_col].shift(-self.horizon)  # Prédiction à H horizons
        
        # Supprimer les lignes avec NaN
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        return X, y
    
    def split_train_test(self, X, y, test_size=0.2):
        """
        Split temporel (pas aléatoire!)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        test_size : float
            Proportion pour le test
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Split temporel: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test

# Exécution directe pour test
if __name__ == "__main__":
    # Charger les données processed
    df = pd.read_csv('data/processed/merged_dataset.csv', index_col=0, parse_dates=True)
    
    fe = FeatureEngineer()
    
    # Calcul des retours
    df_returns = fe.compute_returns(df, log_returns=True)
    print("Retours calculés")
    
    # Création des lags pour BTC
    X, y = fe.create_lags(df_returns, target_col="BTC_('Close', 'BTC-USD')", lags=[1,2,3])
    print(f"Features lags: {X.shape}")
    print(f"Cible: {y.shape}")
    
    # Split temporel
    X_train, X_test, y_train, y_test = fe.split_train_test(X, y)