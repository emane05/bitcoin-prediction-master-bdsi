# tests/test_metrics.py
# Tests unitaires pour les métriques et l'évaluation

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from src.evaluation.metrics import compute_metrics, compare_models
from src.evaluation.walk_forward import WalkForwardValidator
from src.evaluation.backtest import StrategyBacktester

class TestMetrics(unittest.TestCase):
    """Tests pour les métriques d'évaluation"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    def test_compute_metrics(self):
        """Test du calcul des métriques"""
        metrics = compute_metrics(self.y_true, self.y_pred)
        
        # Vérifier la présence des métriques
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        self.assertIn('Direction_Accuracy', metrics)
        
        # Vérifier les valeurs (approximatives)
        self.assertAlmostEqual(metrics['MAE'], 0.14, places=2)
        self.assertAlmostEqual(metrics['RMSE'], 0.15, places=2)
    
    def test_compare_models(self):
        """Test de la comparaison de modèles"""
        metrics1 = {'MAE': 0.1, 'RMSE': 0.2, 'MAPE': 5.0, 
                    'Direction_Accuracy': 60.0, 'R2': 0.9, 'n_obs': 100}
        metrics2 = {'MAE': 0.2, 'RMSE': 0.3, 'MAPE': 8.0, 
                    'Direction_Accuracy': 55.0, 'R2': 0.8, 'n_obs': 100}
        
        all_metrics = {
            'Model1': metrics1,
            'Model2': metrics2
        }
        
        df = compare_models(all_metrics)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['Modèle'], 'Model1')  # Meilleur RMSE

class TestWalkForward(unittest.TestCase):
    """Tests pour la validation walk-forward"""
    
    def setUp(self):
        self.validator = WalkForwardValidator(n_splits=3, train_size=0.6)
        
        # Créer des données factices
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        n = len(dates)
        self.X = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n)
        }, index=dates)
        self.y = pd.Series(np.random.randn(n), index=dates)
    
    def test_split(self):
        """Test de la génération des splits"""
        splits = self.validator.split(self.X, self.y)
        
        self.assertEqual(len(splits), 3)
        
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            # Vérifier l'ordre temporel
            self.assertLess(train_idx[-1], test_idx[0])

class TestBacktest(unittest.TestCase):
    """Tests pour le backtest de stratégie"""
    
    def setUp(self):
        self.backtester = StrategyBacktester(initial_capital=10000)
        
        # Simuler des prédictions et retours
        self.predictions = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        self.true_returns = np.array([0.012, -0.004, 0.018, -0.012, 0.014])
        self.dates = pd.date_range('2024-01-01', periods=5, freq='D')
    
    def test_run_backtest(self):
        """Test de l'exécution du backtest"""
        results = self.backtester.run_backtest(
            self.predictions,
            self.true_returns,
            self.dates,
            model_name="Test"
        )
        
        self.assertIn('final_capital', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('total_return', results)
        self.assertIn('win_rate', results)
        
        # Vérifier que le capital final est calculé
        self.assertGreater(results['final_capital'], 0)

if __name__ == '__main__':
    unittest.main()