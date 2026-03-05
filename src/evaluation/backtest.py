# src/evaluation/backtest.py
# Simulation de stratégie de trading basée sur les prévisions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class StrategyBacktester:
    """Backtest d'une stratégie simple: long si prévision retour > 0, sinon flat"""
    
    def __init__(self, initial_capital: float = 10000.0, 
                 transaction_cost: float = 0.001):
        """
        Initialise le backtester
        
        Parameters:
        -----------
        initial_capital : float
            Capital initial
        transaction_cost : float
            Coûts de transaction (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = []
    
    def run_backtest(self, predictions: np.ndarray, true_returns: np.ndarray,
                    dates: pd.Index, model_name: str = "Strategy") -> dict:
        """
        Exécute le backtest de la stratégie
        
        Stratégie: 
        - Si prévision retour > 0 → Long (acheter)
        - Si prévision retour ≤ 0 → Flat (rester en cash)
        
        Parameters:
        -----------
        predictions : np.ndarray
            Prévisions des retours
        true_returns : np.ndarray
            Retours réels
        dates : pd.Index
            Dates correspondantes
        model_name : str
            Nom du modèle
        
        Returns:
        --------
        dict
            Résultats du backtest
        """
        # Aligner les longueurs
        n = min(len(predictions), len(true_returns), len(dates))
        pred = predictions[:n]
        ret = true_returns[:n]
        date_idx = dates[:n]
        
        # Signal: 1 si prévision > 0, 0 sinon
        signals = (pred > 0).astype(int)
        
        # Rendements de la stratégie
        strategy_returns = signals * ret
        
        # Coûts de transaction (quand le signal change)
        position_changes = np.diff(np.concatenate([[0], signals]))
        transaction_costs = np.abs(position_changes) * self.transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        
        # Capital cumulé
        capital = self.initial_capital * np.cumprod(1 + strategy_returns)
        
        # Benchmark: buy & hold
        benchmark_returns = ret  # Investi en permanence
        benchmark_capital = self.initial_capital * np.cumprod(1 + benchmark_returns)
        
        # Métriques de performance
        total_return = (capital[-1] / self.initial_capital - 1) * 100
        benchmark_return = (benchmark_capital[-1] / self.initial_capital - 1) * 100
        
        # Sharpe ratio (annualisé)
        risk_free_rate = 0.02  # 2%
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-6)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(capital)
        drawdown = (peak - capital) / peak
        max_drawdown = drawdown.max() * 100
        
        # Win rate
        winning_trades = (strategy_returns[signals == 1] > 0).sum()
        total_trades = (signals == 1).sum()
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        results = {
            'model_name': model_name,
            'initial_capital': self.initial_capital,
            'final_capital': capital[-1],
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': total_trades,
            'signals': signals,
            'strategy_returns': strategy_returns,
            'strategy_capital': capital,
            'benchmark_capital': benchmark_capital,
            'dates': date_idx
        }
        
        self.results.append(results)
        
        return results
    
    def plot_results(self, results: dict = None, save_path: str = None):
        """
        Affiche les résultats du backtest
        
        Parameters:
        -----------
        results : dict
            Résultats du backtest (sinon utilise le dernier)
        save_path : str
            Chemin pour sauvegarder le graphique
        """
        if results is None:
            if self.results:
                results = self.results[-1]
            else:
                print("Aucun résultat à afficher")
                return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Capital cumulé
        axes[0].plot(results['dates'], results['strategy_capital'], 
                    label=f"Stratégie ({results['model_name']})", linewidth=2)
        axes[0].plot(results['dates'], results['benchmark_capital'], 
                    label="Buy & Hold", linewidth=2, alpha=0.7)
        axes[0].axhline(y=results['initial_capital'], color='gray', 
                       linestyle='--', alpha=0.5)
        axes[0].set_title(f"Évolution du capital - {results['model_name']}")
        axes[0].set_ylabel("Capital ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Signaux
        axes[1].fill_between(results['dates'], 0, results['signals'], 
                            alpha=0.3, color='green', label='Long')
        axes[1].set_title("Signaux de trading (1 = Long, 0 = Flat)")
        axes[1].set_ylabel("Position")
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        peak = np.maximum.accumulate(results['strategy_capital'])
        drawdown = (peak - results['strategy_capital']) / peak * 100
        axes[2].fill_between(results['dates'], 0, drawdown, 
                            alpha=0.5, color='red')
        axes[2].set_title("Drawdown (%)")
        axes[2].set_ylabel("Drawdown %")
        axes[2].set_xlabel("Date")
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def print_report(self, results: dict = None):
        """
        Affiche un rapport détaillé du backtest
        
        Parameters:
        -----------
        results : dict
            Résultats du backtest
        """
        if results is None:
            if self.results:
                results = self.results[-1]
            else:
                print("Aucun résultat à afficher")
                return
        
        print("\n" + "="*60)
        print(f"RAPPORT DE BACKTEST - {results['model_name']}")
        print("="*60)
        print(f"Capital initial: ${results['initial_capital']:,.2f}")
        print(f"Capital final:   ${results['final_capital']:,.2f}")
        print("-"*60)
        print(f"Rendement total:      {results['total_return']:.2f}%")
        print(f"Rendement benchmark:  {results['benchmark_return']:.2f}%")
        print(f"Rendement excédentaire: {results['excess_return']:.2f}%")
        print("-"*60)
        print(f"Sharpe ratio:      {results['sharpe_ratio']:.3f}")
        print(f"Maximum drawdown:  {results['max_drawdown']:.2f}%")
        print(f"Win rate:          {results['win_rate']:.1f}%")
        print(f"Nombre de trades:  {results['n_trades']}")
        print("="*60)
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare les performances de toutes les stratégies backtestées
        
        Returns:
        --------
        pd.DataFrame
            Tableau comparatif
        """
        comparison = []
        
        for result in self.results:
            row = {
                'Modèle': result['model_name'],
                'Rendement %': result['total_return'],
                'Benchmark %': result['benchmark_return'],
                'Excès %': result['excess_return'],
                'Sharpe': result['sharpe_ratio'],
                'DD max %': result['max_drawdown'],
                'Win rate %': result['win_rate'],
                'Trades': result['n_trades']
            }
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.round({
            'Rendement %': 2,
            'Benchmark %': 2,
            'Excès %': 2,
            'Sharpe': 3,
            'DD max %': 2,
            'Win rate %': 1
        })
        
        return df.sort_values('Sharpe', ascending=False)