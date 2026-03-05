# src/data/collector.py
# Téléchargement des données financières depuis Yahoo Finance

import yfinance as yf
import pandas as pd
from datetime import datetime
import yaml

class DataCollector:
    """Classe pour télécharger les données de crypto et indices"""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le collecteur avec la configuration
        
        Parameters:
        -----------
        config_path : str
            Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tickers = self.config['tickers']
        self.columns = self.config['columns']
        self.start = self.config['start_date']
        self.end = self.config['end_date']
    
    def download_ticker(self, ticker_name, ticker_symbol):
        """
        Télécharge les données pour un ticker spécifique
        
        Parameters:
        -----------
        ticker_name : str
            Nom de la variable (ex: 'BTC')
        ticker_symbol : str
            Symbole Yahoo Finance (ex: 'BTC-USD')
        
        Returns:
        --------
        pd.DataFrame
            Données téléchargées
        """
        print(f"Téléchargement de {ticker_name} ({ticker_symbol})...")
        
        try:
            # Téléchargement des données
            data = yf.download(
                ticker_symbol,
                start=self.start,
                end=self.end,
                progress=False
            )
            
            # Sélection des colonnes souhaitées
            available_cols = [col for col in self.columns if col in data.columns]
            data = data[available_cols].copy()
            
            # Renommage des colonnes avec préfixe
            data.columns = [f"{ticker_name}_{col}" for col in data.columns]
            
            print(f"✓ {len(data)} jours téléchargés pour {ticker_name}")
            return data
            
        except Exception as e:
            print(f"✗ Erreur pour {ticker_name}: {e}")
            return pd.DataFrame()
    
    def download_all(self):
        """
        Télécharge toutes les données configurées
        
        Returns:
        --------
        dict
            Dictionnaire des DataFrames par ticker
        """
        data_dict = {}
        
        for name, symbol in self.tickers.items():
            df = self.download_ticker(name, symbol)
            if not df.empty:
                data_dict[name] = df
        
        return data_dict
    
    def save_raw_data(self, data_dict, output_dir='data/raw/'):
        """
        Sauvegarde les données brutes en CSV
        
        Parameters:
        -----------
        data_dict : dict
            Dictionnaire des DataFrames
        output_dir : str
            Dossier de sortie
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data_dict.items():
            filepath = f"{output_dir}{name.lower()}.csv"
            df.to_csv(filepath)
            print(f"✓ Sauvegardé: {filepath}")

# Exécution directe pour test
if __name__ == "__main__":
    collector = DataCollector()
    data = collector.download_all()
    collector.save_raw_data(data)