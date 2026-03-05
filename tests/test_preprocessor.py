# tests/test_preprocessor.py
# Tests unitaires pour le module de preprocessing

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor
from src.data.collector import DataCollector

class TestPreprocessor(unittest.TestCase):
    """Tests pour la classe DataPreprocessor"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.preprocessor = DataPreprocessor()
        
        # Créer un petit DataFrame de test
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
        self.test_df = pd.DataFrame({
            'BTC_Close': [40000, 41000, None, 42000, 43000, None, 44000, 45000, 46000, 47000],
            'ETH_Close': [3000, 3100, 3200, None, 3300, 3400, 3500, None, 3600, 3700],
            'BTC_Volume': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        }, index=dates)
    
    def test_clean_dataframe_ffill(self):
        """Test du nettoyage avec forward fill"""
        cleaned = self.preprocessor.clean_dataframe(self.test_df, "Test")
        
        # Vérifier qu'il n'y a plus de NaN
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
        
        # Vérifier que les NaN ont été remplis
        self.assertEqual(cleaned.loc['2024-01-03', 'BTC_Close'], 41000)  # ffill
        self.assertEqual(cleaned.loc['2024-01-08', 'ETH_Close'], 3500)   # ffill
    
    def test_align_datasets(self):
        """Test de l'alignement temporel"""
        data_dict = {
            'BTC': self.test_df[['BTC_Close']],
            'ETH': self.test_df[['ETH_Close']].iloc[2:]  # Commence plus tard
        }
        
        aligned = self.preprocessor.align_datasets(data_dict)
        
        # Vérifier que les dates sont identiques
        btc_dates = set(aligned['BTC'].index)
        eth_dates = set(aligned['ETH'].index)
        self.assertEqual(btc_dates, eth_dates)
    
    def test_merge_datasets(self):
        """Test de la fusion"""
        data_dict = {
            'BTC': self.test_df[['BTC_Close']],
            'ETH': self.test_df[['ETH_Close']]
        }
        
        merged = self.preprocessor.merge_datasets(data_dict)
        
        # Vérifier la forme
        self.assertEqual(merged.shape, (10, 2))
        
        # Vérifier les noms de colonnes
        self.assertIn('BTC_Close', merged.columns)
        self.assertIn('ETH_Close', merged.columns)

class TestCollector(unittest.TestCase):
    """Tests pour la classe DataCollector"""
    
    def test_init(self):
        """Test de l'initialisation"""
        collector = DataCollector()
        self.assertIsNotNone(collector.tickers)
        self.assertIn('BTC', collector.tickers)
        self.assertIn('ETH', collector.tickers)

if __name__ == '__main__':
    unittest.main()