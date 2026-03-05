# src/data/preprocessor.py
# Nettoyage, alignement et fusion des données

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import os

class DataPreprocessor:
    """Classe pour nettoyer et fusionner les données"""
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialise le préprocesseur avec la configuration
        
        Parameters:
        -----------
        config_path : str
            Chemin vers le fichier de configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fill_method = self.config.get('fill_method', 'ffill')
        self.resample_freq = self.config.get('resample_freq', '1D')
    
    def load_raw_data(self, raw_dir='data/raw/'):
        """
        Charge tous les CSV bruts
        
        Parameters:
        -----------
        raw_dir : str
            Dossier contenant les données brutes
        
        Returns:
        --------
        dict
            Dictionnaire des DataFrames chargés
        """
        data_dict = {}
        files = ['btc', 'eth', 'bnb', 'sp500', 'dxy']
        
        for file in files:
            filepath = f"{raw_dir}{file}.csv"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                data_dict[file.upper()] = df
                print(f"✓ Chargé: {filepath} ({len(df)} lignes)")
            else:
                print(f"⚠ Fichier manquant: {filepath}")
        
        return data_dict
    
    def clean_dataframe(self, df, name):
        """
        Nettoie un DataFrame individuel
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame à nettoyer
        name : str
            Nom de la variable pour logs
        
        Returns:
        --------
        pd.DataFrame
            DataFrame nettoyé
        """
        df_clean = df.copy()
        
        # Suppression des lignes totalement vides
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(how='all')
        
        if len(df_clean) < initial_len:
            print(f"  {name}: {initial_len - len(df_clean)} lignes vides supprimées")
        
        # Remplissage des NaN par forward fill puis backward fill
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Suppression des doublons d'index
        if df_clean.index.duplicated().any():
            df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
            print(f"  {name}: doublons d'index supprimés")
        
        return df_clean
    
    def align_datasets(self, data_dict):
        """
        Aligne tous les datasets sur les mêmes dates
        
        Parameters:
        -----------
        data_dict : dict
            Dictionnaire des DataFrames
        
        Returns:
        --------
        dict
            Dictionnaire des DataFrames alignés
        """
        # Trouver l'intersection de toutes les dates
        all_dates = None
        
        for name, df in data_dict.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
        
        all_dates = sorted(list(all_dates))
        print(f"Alignement sur {len(all_dates)} dates communes")
        
        # Filtrer chaque DataFrame
        aligned_dict = {}
        for name, df in data_dict.items():
            aligned_dict[name] = df.loc[all_dates].copy()
        
        return aligned_dict
    
    def merge_datasets(self, data_dict):
        """
        Fusionne tous les DataFrames en un seul
        
        Parameters:
        -----------
        data_dict : dict
            Dictionnaire des DataFrames alignés
        
        Returns:
        --------
        pd.DataFrame
            Dataset fusionné
        """
        dfs = []
        
        for name, df in data_dict.items():
            dfs.append(df)
        
        merged = pd.concat(dfs, axis=1)
        print(f"Fusion terminée: {merged.shape[0]} lignes, {merged.shape[1]} colonnes")
        
        return merged
    
    def save_processed_data(self, df, output_path='data/processed/merged_dataset.csv'):
        """
        Sauvegarde le dataset fusionné
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset à sauvegarder
        output_path : str
            Chemin de sortie
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"✓ Dataset sauvegardé: {output_path}")
    
    def run_pipeline(self):
        """
        Exécute le pipeline complet de preprocessing
        
        Returns:
        --------
        pd.DataFrame
            Dataset final prêt pour l'analyse
        """
        print("="*50)
        print("DÉBUT DU PIPELINE DE PREPROCESSING")
        print("="*50)
        
        # 1. Chargement
        print("\n1. Chargement des données brutes...")
        raw_data = self.load_raw_data()
        
        # 2. Nettoyage
        print("\n2. Nettoyage individuel...")
        cleaned_data = {}
        for name, df in raw_data.items():
            cleaned_data[name] = self.clean_dataframe(df, name)
        
        # 3. Alignement
        print("\n3. Alignement temporel...")
        aligned_data = self.align_datasets(cleaned_data)
        
        # 4. Fusion
        print("\n4. Fusion des datasets...")
        merged = self.merge_datasets(aligned_data)
        
        # 5. Sauvegarde
        print("\n5. Sauvegarde...")
        self.save_processed_data(merged)
        
        print("\n" + "="*50)
        print("✓ PIPELINE TERMINÉ AVEC SUCCÈS")
        print("="*50)
        
        return merged

# Exécution directe pour test
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.run_pipeline()
    print(f"\nAperçu du dataset final:")
    print(df.head())
    print(f"\nStatistiques descriptives:")
    print(df.describe())