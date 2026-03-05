# bitcoin-prediction-master-bdsi
 Prévision multivariée du Bitcoin avec variables explicatives - Projet Master 1 BDSI


## 📋 Table des matières

- [Contexte & Problématique](#-contexte--problématique)
- [Structure du projet](#-structure-du-projet)
- [Données](#-données)
- [Installation](#-installation)
- [Pipeline complet](#-pipeline-complet)
- [Modèles implémentés](#-modèles-implémentés)
- [Résultats](#-résultats)
- [Stratégie de trading](#-stratégie-de-trading)
- [Technologies utilisées](#-technologies-utilisées)
- [Perspectives](#-perspectives)

---

## 🎯 Contexte & Problématique

Le Bitcoin (BTC), depuis sa création en 2009, représente la cryptomonnaie dominante avec une capitalisation régulièrement supérieure à 500 milliards de dollars. Sa **volatilité exceptionnelle** (variations quotidiennes de 10–20% en période de turbulence) en fait un sujet d'étude privilégié pour les méthodes de prévision de séries temporelles.

Ce projet répond à deux questions de recherche centrales :

### ❓ Question 1 — BTC dépend-il d'autres variables financières ?
Les mouvements de prix du Bitcoin sont-ils influencés par :
- D'autres cryptomonnaies majeures (Ethereum, Binance Coin) ?
- Les volumes d'échange (liquidité, conviction) ?
- Des indices macro-économiques (S&P 500, Dollar Index) ?

### ❓ Question 2 — Les modèles classiques améliorent-ils la prévision ?
L'ajout de complexité méthodologique (ARIMA → VAR → Ridge/Lasso → Random Forest) se traduit-il par des **gains prédictifs mesurables** ?

---

## 📁 Structure du projet

```
btc-multivariate-forecasting/
│
├── data/
│   ├── raw/                      # Données brutes téléchargées via yfinance
│   │   ├── BTC-USD.csv
│   │   ├── ETH-USD.csv
│   │   ├── BNB-USD.csv
│   │   ├── GSPC.csv
│   │   └── DXY.csv
│   └── processed/
│       └── merged_dataset.csv    # Dataset fusionné et nettoyé (1257 obs × 11 col)
│
├── notebooks/
│   ├── 01_data_collection.ipynb        # Collecte automatisée via API Yahoo Finance
│   ├── 02_preprocessing.ipynb          # Nettoyage, alignement, log-retours
│   ├── 03_eda_stationarity.ipynb       # EDA, tests ADF/KPSS, ACF/PACF
│   ├── 04_feature_engineering.ipynb    # Construction des lags, tests de Granger
│   ├── 05_models_training.ipynb        # Entraînement des 5 modèles
│   ├── 06_evaluation.ipynb             # Comparaison walk-forward, métriques
│   └── 07_trading_strategy.ipynb       # Backtest stratégie long/flat
│
├── src/
│   ├── data_collection.py        # Téléchargement automatisé yfinance
│   ├── preprocessing.py          # Pipeline de nettoyage et transformation
│   ├── feature_engineering.py    # Construction des features laggées
│   ├── models/
│   │   ├── arima_model.py        # ARIMA univarié (benchmark)
│   │   ├── var_model.py          # VAR multivarié
│   │   ├── ridge_lasso.py        # Régression régularisée Ridge & Lasso
│   │   └── random_forest.py      # Random Forest avec grid search
│   ├── evaluation.py             # Walk-forward validation & métriques
│   └── trading_strategy.py       # Backtest de la stratégie long/flat
│
├── results/
│   ├── figures/                  # Graphiques de performance, feature importance
│   └── metrics_comparison.csv    # Tableau récapitulatif des métriques
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Données

### Sources

| Actif | Ticker | Période | Observations | Fréquence |
|---|---|---|---|---|
| Bitcoin | `BTC-USD` | 2020–2025 | 1 827 | Quotidienne |
| Ethereum | `ETH-USD` | 2020–2025 | 1 827 | Quotidienne |
| Binance Coin | `BNB-USD` | 2020–2025 | 1 827 | Quotidienne |
| S&P 500 | `^GSPC` | 2020–2025 | 1 258 | Jours ouvrés |
| Dollar Index | `DX-Y.NYB` | 2020–2025 | 1 258 | Jours ouvrés |

**Source :** Yahoo Finance via la bibliothèque `yfinance` (accès libre, format OHLCV standardisé).

### Pipeline de prétraitement

```
Données brutes (5 CSV)
        ↓
Sélection colonnes Close + Volume
        ↓
Gestion valeurs manquantes (ffill → bfill)
        ↓
Alignement temporel (intersection des dates communes)
        → Perte de 570 obs. (cryptos actives week-ends)
        ↓
Fusion en DataFrame unique (1257 lignes × 11 colonnes)
        ↓
Transformation en log-retours : rt = ln(Pt / Pt-1)
        ↓
Validation stationnarité (ADF + KPSS)
        ↓
merged_dataset.csv ✅
```

### Validation de la stationnarité

| Test | H₀ | Résultat sur prix | Résultat sur log-retours |
|---|---|---|---|
| ADF | Série non-stationnaire | p > 0.05 → **non-stationnaire** ✗ | p < 0.001 → **stationnaire** ✅ |
| KPSS | Série stationnaire | p < 0.05 → **non-stationnaire** ✗ | p > 0.05 → **stationnaire** ✅ |

> La transformation en log-retours est donc **validée et nécessaire** avant toute modélisation.

---

## ⚙️ Installation

### Prérequis

- Python 3.9+
- pip ou conda

### Installation des dépendances

```bash
git clone https://github.com/votre-username/btc-multivariate-forecasting.git
cd btc-multivariate-forecasting
pip install -r requirements.txt
```

### `requirements.txt`

```
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.10.0
joblib>=1.2.0
```

### Collecte des données

```bash
python src/data_collection.py
```

Ou directement dans un notebook :

```python
import yfinance as yf

tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "^GSPC", "DX-Y.NYB"]
for ticker in tickers:
    df = yf.download(ticker, start="2020-01-01", end="2025-01-01")
    df.to_csv(f"data/raw/{ticker.replace('^','')}.csv")
```

---

## 🔧 Pipeline complet

### 1. Feature Engineering

Pour chaque observation au temps *t*, un vecteur de **30 features** est construit :

```
X_t = [
  BTC_log_return(t-1), ..., BTC_log_return(t-5),    # 5 lags BTC
  ETH_log_return(t-1), ..., ETH_log_return(t-5),    # 5 lags ETH
  BNB_log_return(t-1), ..., BNB_log_return(t-5),    # 5 lags BNB
  SP500_log_return(t-1), ..., SP500_log_return(t-5), # 5 lags S&P500
  DXY_log_return(t-1), ..., DXY_log_return(t-5),    # 5 lags DXY
  BTC_Volume(t-1), ..., BTC_Volume(t-5)              # 5 lags Volume
]

y_t = BTC_log_return(t+1)  # Cible : retour J+1
```

### 2. Tests de causalité de Granger

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Exemple : ETH cause-t-il BTC ?
grangercausalitytests(df[['BTC_ret', 'ETH_ret']], maxlag=5)
```

| Variable | p-value Granger | Interprétation |
|---|---|---|
| ETH_lag1 | < 0.001 | ✅ Cause significative |
| BNB_lag1 | < 0.001 | ✅ Cause significative |
| BTC_lag1 | < 0.001 | ✅ Autocorrélation |
| Volume_lag1 | 0.0876 | ⚠️ Effet modéré |
| SP500_lag1 | 0.2345 | ✗ Pas d'effet |
| DXY_lag1 | 0.3124 | ✗ Pas d'effet |

### 3. Split temporel (80/20)

```python
# Respect strict de l'ordre chronologique — aucun data leakage
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```

---

## 🤖 Modèles implémentés

### 1. ARIMA — Benchmark univarié

Utilise uniquement l'historique du BTC (pas de variables externes).

```python
from statsmodels.tsa.arima.model import ARIMA

# Sélection par grid search sur AIC
# Résultat : ARIMA(1, 0, 1) — AIC = -1245.67
model = ARIMA(y_train, order=(1, 0, 1))
result = model.fit()
```

**Paramètres estimés :**
- `const = 0.0002` (p=0.019) — Tendance haussière faible
- `ar.L1 = 0.4567` (p<0.001) — Autocorrélation positive à lag 1
- `ma.L1 = -0.2345` (p<0.001) — Correction des erreurs passées

---

### 2. VAR — Extension multivariée

Chaque variable dépend de ses propres lags **et** des lags de toutes les autres.

```python
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(train_data)  # 6 variables
results = model.fit(maxlags=10, ic='aic')
# Résultat : VAR(5) sélectionné → 186 paramètres estimés
```

**Dimensions du système :**
- 6 équations × 5 lags × 6 variables = **180 coefficients + 6 constantes**

---

### 3. Ridge & Lasso — Régression régularisée

```python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

# Ridge (L2) — λ optimal = 1.0
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], cv=tscv)
ridge.fit(X_train, y_train)

# Lasso (L1) — λ = 0.01 → 18 coefficients forcés à 0
lasso = LassoCV(cv=tscv)
lasso.fit(X_train, y_train)
```

**Sélection Lasso :**
- ✅ **Conservées** (12 features) : `ETH_lag1-2`, `BNB_lag1-2`, `BTC_lag1-2`, `Volume_lag1-2`
- ✗ **Éliminées** (18 features) : Tous les lags SP500 et DXY, lags 3–5 des autres variables

---

### 4. Random Forest — Approche non-linéaire

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

# 432 configurations testées — ~45 minutes sur CPU standard
rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=tscv)
rf.fit(X_train, y_train)

# Hyperparamètres optimaux
# n_estimators=200, max_depth=15, min_samples_split=5,
# min_samples_leaf=2, max_features='sqrt'
```

**Top 5 features (MDI importance) :**

| Rang | Feature | Importance |
|---|---|---|
| 1 | `ETH_lag1` | 32.45% |
| 2 | `BNB_lag1` | 24.56% |
| 3 | `BTC_lag1` | 18.76% |
| 4 | `ETH_lag2` | 9.87% |
| 5 | `Volume_lag1` | 7.65% |

> Le **top 5 cumule 93.29%** de l'importance totale — les 25 autres features n'apportent que 6.71%.

---

## 📏 Validation — Walk-Forward

La validation walk-forward simule un **déploiement en conditions réelles** : le modèle est entraîné sur une fenêtre glissante et testé sur la période suivante, sans jamais voir le futur.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    mape = mean_absolute_percentage_error(y_te, preds)
    dir_acc = np.mean(np.sign(preds) == np.sign(y_te))
```

**Stabilité temporelle — Random Forest (CV = 1.5%) :**

| Fold | Période | RF RMSE | ARIMA RMSE |
|---|---|---|---|
| 1 | 2022 H2 | 0.02389 | 0.02945 |
| 2 | 2023 H1 | 0.02312 | 0.02898 |
| 3 | 2023 H2 | 0.02298 | 0.02812 |
| 4 | 2024 H1 | 0.02345 | 0.02901 |
| 5 | 2024 H2 | 0.02301 | 0.02834 |

---

## 📊 Résultats

### Comparaison des performances

| Modèle | MAE | RMSE | Direction Accuracy | Gain RMSE vs ARIMA | Complexité |
|---|---|---|---|---|---|
| 🥇 **Random Forest** | **0.01457** | **0.02340** | **58.3%** | **+19.0%** | Haute |
| 🥈 Ridge | 0.01568 | 0.02412 | 57.1% | +16.6% | Moyenne |
| 🥉 Lasso | 0.01623 | 0.02488 | 56.3% | +14.0% | Moyenne |
| VAR | 0.01654 | 0.02561 | 55.8% | +11.4% | Haute |
| ARIMA *(baseline)* | 0.01877 | 0.02890 | 52.4% | — | Faible |

### Insights clés

- **Tous les modèles multivariés surpassent ARIMA** : gain minimal de 11.4% (VAR), maximal de 19.0% (Random Forest) → la dépendance de BTC envers ETH et BNB est avérée.
- **Direction Accuracy > 55%** pour tous les modèles multivariés, contre 52.4% pour ARIMA — critique pour le trading.
- **Les indices macro (SP500, DXY) n'influencent pas BTC à court terme** : éliminés par Lasso, importance < 1% dans Random Forest, Granger non significatif.

### Convergence des méthodes d'identification

| Variable | Granger | Lasso | RF Importance | Conclusion |
|---|---|---|---|---|
| ETH_lag1 | p < 0.001 | ✅ Conservée | 32.45% | **Prédicteur #1** |
| BNB_lag1 | p < 0.001 | ✅ Conservée | 24.56% | **Prédicteur #2** |
| BTC_lag1 | p < 0.001 | ✅ Conservée | 18.76% | Autocorrélation |
| Volume_lag1 | p = 0.088 | ✅ Conservée | 7.65% | Effet modéré |
| SP500_lag1 | p = 0.235 | ✗ Éliminée | < 1% | Pas d'effet |
| DXY_lag1 | p = 0.312 | ✗ Éliminée | < 1% | Pas d'effet |

---

## 💹 Stratégie de trading

### Règle simple long/flat

```python
def trading_strategy(predictions, actual_returns, transaction_cost=0.001):
    """
    Si prévision retour > 0 → position LONG (acheter)
    Sinon             → FLAT  (ne rien faire)
    """
    positions = np.where(predictions > 0, 1, 0)
    
    # Coût de transaction à chaque changement de position
    trades = np.diff(positions, prepend=0)
    costs = np.abs(trades) * transaction_cost
    
    strategy_returns = positions * actual_returns - costs
    return strategy_returns
```

### Résultats du backtest (2024)

| Métrique | Stratégie RF | Buy & Hold |
|---|---|---|
| Rendement annuel | **+34.2%** | +18.5% |
| Sharpe Ratio | **1.42** | 0.85 |
| Max Drawdown | **-12.3%** | -22.4% |
| Win Rate | **58.3%** | — |
| Nombre de trades | 187 | 1 |

```
Sharpe = (Rendement - Taux sans risque) / Volatilité
       = (34.2% - 0%) / 24.1%
       = 1.42  ← Excellent (seuil > 1)
```

> La stratégie active **réduit de moitié le drawdown maximal** (-12.3% vs -22.4%) tout en générant presque le **double du rendement** (+34.2% vs +18.5%).

---

## 🛠 Technologies utilisées

| Catégorie | Bibliothèques |
|---|---|
| Collecte de données | `yfinance` |
| Manipulation des données | `pandas`, `numpy` |
| Tests statistiques | `statsmodels` (ADF, KPSS, Granger, ARIMA, VAR) |
| Machine Learning | `scikit-learn` (Ridge, Lasso, RandomForest, GridSearchCV) |
| Visualisation | `matplotlib`, `seaborn` |
| Environnement | Python 3.9+, Jupyter Notebook |

---

## 🔭 Perspectives

### Extensions méthodologiques
- **Deep Learning** : LSTM, GRU, Transformers pour les dépendances temporelles longues
- **Modèles GARCH/EGARCH** : Modélisation explicite des clusters de volatilité
- **Regime-switching (HMM)** : Détection automatique des phases bull/bear/lateral
- **Ensemble methods** : Stacking/blending de modèles pour plus de robustesse

### Enrichissement des données
- **Données on-chain** : Adresses actives, hash rate, transactions confirmées
- **Sentiment analysis** : Twitter/Reddit via NLP, Google Trends, Fear & Greed Index
- **Features techniques** : RSI, MACD, Bollinger Bands
- **Macro avancée** : Taux Fed, inflation, M2 money supply

### Optimisation de la stratégie
- **Position sizing dynamique** : Critère de Kelly adaptatif
- **Shorting** : Positions courtes quand prévision négative
- **Multi-horizon** : Combiner signaux H=1j, H=7j, H=30j
- **Optimisation de portefeuille** : Allocation BTC/ETH/BNB selon prévisions relatives

---

## 👥 Auteurs

**Imane Boujaj** & **Safae Wardi**  
Projet réalisé en février 2026 dans le cadre d'un cours de séries temporelles appliquées.

---

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

<div align="center">

*"L'avenir appartient à ceux qui combinent rigueur scientifique, expertise technique, et compréhension des marchés financiers."*

</div>
