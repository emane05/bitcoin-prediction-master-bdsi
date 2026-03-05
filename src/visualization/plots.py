# src/visualization/plots.py
# Fonctions de visualisation pour l'analyse exploratoire

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import os

class Visualizer:
    """Classe pour générer toutes les visualisations"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialise le visualiseur
        
        Parameters:
        -----------
        style : str
            Style matplotlib
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.figsize = (12, 6)
    
    def plot_prices(self, df, cols=None, title="Évolution des prix", 
                   save_path=None):
        """
        Graphique des séries de prix
        
        Parameters:
        -----------
        df : pd.DataFrame
            Données
        cols : list
            Colonnes à afficher
        title : str
            Titre du graphique
        save_path : str
            Chemin de sauvegarde
        """
        if cols is None:
            cols = [col for col in df.columns if 'Close' in col]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for col in cols:
            # Normalisation pour comparaison (base 100)
            normalized = df[col] / df[col].iloc[0] * 100
            ax.plot(df.index, normalized, label=col, linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix normalisés (base 100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, df, cols=None, title="Distribution des retours",
                                 save_path=None):
        """
        Distribution des retours avec histogramme et boxplot
        
        Parameters:
        -----------
        df : pd.DataFrame
            Données avec retours
        cols : list
            Colonnes des retours
        title : str
            Titre
        save_path : str
            Chemin de sauvegarde
        """
        if cols is None:
            cols = [col for col in df.columns if 'return' in col]
        
        fig, axes = plt.subplots(len(cols), 2, figsize=(14, 4*len(cols)))
        
        if len(cols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(cols):
            # Histogramme
            axes[i, 0].hist(df[col].dropna(), bins=50, edgecolor='black', 
                           alpha=0.7, color='steelblue')
            axes[i, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)
            axes[i, 0].set_title(f'{col} - Distribution')
            axes[i, 0].set_xlabel('Retour')
            axes[i, 0].set_ylabel('Fréquence')
            
            # Boxplot
            axes[i, 1].boxplot(df[col].dropna(), vert=False)
            axes[i, 1].axvline(x=0, color='red', linestyle='--', linewidth=1)
            axes[i, 1].set_title(f'{col} - Boxplot')
            axes[i, 1].set_xlabel('Retour')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, df, title="Matrice de corrélation",
                               save_path=None):
        """
        Matrice de corrélation heatmap
        
        Parameters:
        -----------
        df : pd.DataFrame
            Données
        title : str
            Titre
        save_path : str
            Chemin de sauvegarde
        """
        # Sélectionner seulement les colonnes numériques
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculer la matrice de corrélation
        corr_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Retourner la matrice pour analyse
        return corr_matrix
    
    def plot_acf_pacf(self, series, lags=40, title="ACF et PACF",
                     save_path=None):
        """
        Fonction d'autocorrélation et autocorrélation partielle
        
        Parameters:
        -----------
        series : pd.Series
            Série temporelle
        lags : int
            Nombre de lags
        title : str
            Titre
        save_path : str
            Chemin de sauvegarde
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF
        plot_acf(series.dropna(), lags=lags, ax=ax1, alpha=0.05)
        ax1.set_title(f'{title} - ACF')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Autocorrélation')
        ax1.grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(series.dropna(), lags=lags, ax=ax2, alpha=0.05, method='ywm')
        ax2.set_title(f'{title} - PACF')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrélation partielle')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_rolling_stats(self, series, window=30, title="Statistiques glissantes",
                          save_path=None):
        """
        Moyenne mobile et écart-type mobile
        
        Parameters:
        -----------
        series : pd.Series
            Série temporelle
        window : int
            Taille de la fenêtre
        title : str
            Titre
        save_path : str
            Chemin de sauvegarde
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        # Série originale + moyenne mobile
        ax1.plot(series.index, series, label='Série originale', alpha=0.5)
        ax1.plot(series.index, rolling_mean, label=f'MM{window}', 
                color='red', linewidth=2)
        ax1.set_title(f'{title} - Moyenne mobile (window={window})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Valeur')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Écart-type mobile
        ax2.plot(series.index, rolling_std, label=f'Écart-type mobile', 
                color='green', linewidth=2)
        ax2.set_title(f'{title} - Écart-type mobile (window={window})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Écart-type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_pairplot(self, df, cols=None, title="Relations entre variables",
                     save_path=None):
        """
        Pairplot pour visualiser les relations entre variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Données
        cols : list
            Colonnes à inclure
        title : str
            Titre
        save_path : str
            Chemin de sauvegarde
        """
        if cols is None:
            # Limiter à 5 colonnes pour lisibilité
            cols = df.columns[:5]
        
        df_subset = df[cols].copy()
        
        # Créer le pairplot
        g = sns.pairplot(df_subset, diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        if save_path:
            g.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données
    df = pd.read_csv('data/processed/merged_dataset.csv', index_col=0, parse_dates=True)
    
    viz = Visualizer()
    
    # 1. Graphique des prix
    viz.plot_prices(df, save_path='reports/figures/prices.png')
    
    # 2. Matrice de corrélation
    corr = viz.plot_correlation_matrix(df, save_path='reports/figures/correlation.png')
    
    # 3. ACF/PACF pour BTC
    viz.plot_acf_pacf(df['BTC_Close'], lags=30, 
                     save_path='reports/figures/acf_pacf_btc.png')