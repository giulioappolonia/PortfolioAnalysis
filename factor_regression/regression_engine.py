# factor_regression/regression_engine.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.stattools import durbin_watson
from typing import Optional, Dict, Any

def prepare_regression_dataset(asset_returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Allinea temporalmente i rendimenti dell'asset con i fattori di French tramite un'inner join,
    e calcola il rendimento in eccesso (asset_excess_return = asset_return - RF).
    
    Parameters:
    -----------
    asset_returns : pd.DataFrame
        DataFrame con DatetimeIndex e colonna 'asset_return'.
    factors : pd.DataFrame
        DataFrame con DatetimeIndex e colonne ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
        
    Returns:
    --------
    pd.DataFrame
        DataFrame allineato e pulito contenente le colonne del contratto dati.
    """
    # Rimuoviamo fusi orari per sicurezza (devono essere tz-naive)
    if asset_returns.index.tz is not None:
        asset_returns.index = asset_returns.index.tz_localize(None)
    if factors.index.tz is not None:
        factors.index = factors.index.tz_localize(None)
        
    # Allineamento temporale (inner join)
    df = pd.merge(
        asset_returns[['asset_return']],
        factors,
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    if df.empty:
        raise ValueError("Nessuna sovrapposizione temporale trovata tra le date dell'asset e quelle dei fattori.")
        
    # Calcolo dell'excess return dell'asset rispetto al risk-free
    df['asset_excess_return'] = df['asset_return'] - df['RF']
    
    # Riordina colonne per conformità al data contract
    cols_order = [
        'asset_return', 'RF', 'asset_excess_return',
        'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'
    ]
    df = df[cols_order]
    
    return df


def run_static_regression(
    df: pd.DataFrame,
    cov_type: str = "HAC",
    maxlags: Optional[int] = None
) -> Dict[str, Any]:
    """
    Esegue la regressione OLS statica sull'intero periodo per stimare i beta ed alpha.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame conforme al data contract.
    cov_type : str
        Tipo di covarianza robusta ('nonrobust', 'HAC', 'HC0', 'HC1', 'HC2', 'HC3').
    maxlags : Optional[int]
        Numero massimo di lag per la covarianza HAC. Se non fornito, viene impostato
        automaticamente (1 per mensile, 5 per giornaliero).
        
    Returns:
    --------
    dict
        Contiene le stime dei parametri, t-stat, p-values, R-squared ed altre metriche.
    """
    # Variabile dipendente (Y)
    y = df['asset_excess_return']
    
    # Variabili indipendenti (X) con costante per calcolare l'intercetta (alpha)
    factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    X = sm.add_constant(df[factors_list])
    
    model = sm.OLS(y, X)
    
    # Adattamento del modello con la covarianza robusta selezionata
    if cov_type == "HAC":
        if maxlags is None:
            # Rilevamento empirico della frequenza
            diffs = df.index.to_series().diff().dropna()
            # Se la distanza mediana è inferiore a 5 giorni, assumiamo giornaliero
            is_daily = diffs.median() < pd.Timedelta(days=5)
            maxlags = 5 if is_daily else 1
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif cov_type in ["HC0", "HC1", "HC2", "HC3"]:
        results = model.fit(cov_type=cov_type)
    else:
        results = model.fit()
        
    # Calcolo Durbin-Watson statistic per l'autocorrelazione dei residui
    dw_stat = durbin_watson(results.resid)
    
    # Preparazione dell'output
    output = {
        'params': results.params,
        'bse': results.bse,
        'tvalues': results.tvalues,
        'pvalues': results.pvalues,
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'nobs': results.nobs,
        'durbin_watson': dw_stat,
        'fvalue': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'results_object': results,
        'covariance_type': cov_type,
        'maxlags': maxlags if cov_type == "HAC" else None
    }
    
    return output


def run_rolling_regression(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Esegue una regressione a finestra mobile (Rolling Regression) usando RollingOLS.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame conforme al data contract.
    window : int
        Ampiezza della finestra mobile in numero di osservazioni.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contenente le serie storiche dei parametri stimati per ogni finestra,
        con colonne: ['alpha', 'beta_Mkt-RF', 'beta_SMB', 'beta_HML', 'beta_RMW', 'beta_CMA', 'beta_Mom', 'r2']
    """
    if len(df) < window:
        raise ValueError(f"Numero di osservazioni ({len(df)}) inferiore alla finestra mobile richiesta ({window}).")
        
    y = df['asset_excess_return']
    factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    X = sm.add_constant(df[factors_list])
    
    # Esecuzione di RollingOLS
    model = RollingOLS(y, X, window=window)
    rolling_res = model.fit()
    
    # Estrazione dei parametri
    rolling_params = rolling_res.params.copy()
    
    # Ridenominazione e pulizia colonne
    rename_dict = {
        'const': 'beta_const',
        'Mkt-RF': 'beta_Mkt-RF',
        'SMB': 'beta_SMB',
        'HML': 'beta_HML',
        'RMW': 'beta_RMW',
        'CMA': 'beta_CMA',
        'Mom': 'beta_Mom'
    }
    rolling_params = rolling_params.rename(columns=rename_dict)
    rolling_params = rolling_params.rename(columns={'beta_const': 'alpha'})
    
    # Aggiunta dell'R-quadrato per ogni finestra
    rolling_params['r2'] = rolling_res.rsquared
    
    # Rimuoviamo i valori iniziali (che sono NaN a causa della finestra)
    rolling_params = rolling_params.dropna(subset=['alpha'])
    
    return rolling_params


def calculate_factor_contributions(
    regression_results: Dict[str, Any],
    df: pd.DataFrame,
    annualize: bool = True,
    frequency: str = "monthly"
) -> pd.DataFrame:
    """
    Scompone il rendimento medio in eccesso stimato (fitted) dell'asset nei contributi di ciascun fattore.
    
    Formula: Contributo = Beta_j * Media(Fattore_j)
    
    Parameters:
    -----------
    regression_results : dict
        Dizionario restituito da run_static_regression.
    df : pd.DataFrame
        DataFrame conforme al data contract.
    annualize : bool
        Se True, i rendimenti e contributi vengono annualizzati linearmente.
    frequency : str
        Frequenza dei dati ('monthly' o 'daily').
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contenente la scomposizione per ogni fattore più l'intercetta (alpha).
    """
    factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    params = regression_results['params']
    
    # Moltiplicatore di annualizzazione lineare
    mult = 1.0
    if annualize:
        mult = 12.0 if frequency == "monthly" else 252.0
        
    contributions = []
    
    # Contributo dei 6 fattori
    for factor in factors_list:
        beta = params[factor]
        factor_mean = df[factor].mean()
        contrib = beta * factor_mean
        
        contributions.append({
            'factor': factor,
            'beta': beta,
            'factor_mean': factor_mean * mult,
            'contribution': contrib * mult
        })
        
    # Contributo di Alpha (intercetta)
    alpha = params['const']
    # Nota: Alpha in frequenza mensile/giornaliera va moltiplicato per il rispettivo periodo per essere annualizzato
    alpha_contrib = alpha * mult
    
    contributions.append({
        'factor': 'Alpha (const)',
        'beta': alpha,
        'factor_mean': np.nan,
        'contribution': alpha_contrib
    })
    
    contrib_df = pd.DataFrame(contributions)
    return contrib_df
