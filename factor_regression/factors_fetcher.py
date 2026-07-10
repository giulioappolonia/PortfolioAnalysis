# factor_regression/factors_fetcher.py

import pandas as pd
import pandas_datareader.data as web
from typing import Optional
from factor_regression.config import REGION_DATASETS

def fetch_factors(
    region: str,
    frequency: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Scarica i 5 fattori Fama-French e il Momentum per la regione e frequenza specificate,
    pulisce le colonne, standardizza i nomi, converte i valori in decimali e unisce i dati.
    
    Parameters:
    -----------
    region : str
        La regione di riferimento (es. 'USA', 'Developed Markets', 'Europe', 'Emerging Markets', 'Global').
    frequency : str
        Frequenza temporale ('daily' o 'monthly').
    start_date : Optional[str]
        Data di inizio in formato 'YYYY-MM-DD'.
    end_date : Optional[str]
        Data di fine in formato 'YYYY-MM-DD'.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con DatetimeIndex (tz-naive) ordinato e le colonne standardizzate in decimali:
        ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']
    """
    if region not in REGION_DATASETS:
        raise ValueError(f"Regione '{region}' non supportata. Selezionare una tra: {list(REGION_DATASETS.keys())}")
    
    if frequency not in ["daily", "monthly"]:
        raise ValueError("La frequenza deve essere 'daily' o 'monthly'.")
        
    config = REGION_DATASETS[region][frequency]
    if config["factors"] is None or config["momentum"] is None:
        raise ValueError(f"La combinazione Regione '{region}' e Frequenza '{frequency}' non è disponibile.")
        
    try:
        # Download dei 5 fattori
        ff_data = web.DataReader(config["factors"], "famafrench", start=start_date, end=end_date)
        # Key 0 del dizionario restituito contiene il DataFrame della tabella principale
        df_ff = ff_data[0].copy()
        
        # Download del momentum
        mom_data = web.DataReader(config["momentum"], "famafrench", start=start_date, end=end_date)
        df_mom = mom_data[0].copy()
    except Exception as e:
        raise RuntimeError(f"Errore durante il download dei dati da Kenneth French per {region}/{frequency}: {e}")
        
    # Pulizia spazi nei nomi delle colonne
    df_ff.columns = df_ff.columns.str.strip()
    df_mom.columns = df_mom.columns.str.strip()
    
    # Conversione date dell'indice dei 5 fattori a DatetimeIndex tz-naive e normalizzazione a mezzanotte
    if isinstance(df_ff.index, pd.PeriodIndex):
        df_ff.index = df_ff.index.to_timestamp(how='end').normalize()
    else:
        df_ff.index = pd.to_datetime(df_ff.index).normalize()
        
    # Conversione date del Momentum a DatetimeIndex tz-naive e normalizzazione a mezzanotte
    if isinstance(df_mom.index, pd.PeriodIndex):
        df_mom.index = df_mom.index.to_timestamp(how='end').normalize()
    else:
        df_mom.index = pd.to_datetime(df_mom.index).normalize()
        
    # Normalizzazione tz-naive
    if df_ff.index.tz is not None:
        df_ff.index = df_ff.index.tz_localize(None)
    if df_mom.index.tz is not None:
        df_mom.index = df_mom.index.tz_localize(None)
        
    # Standardizzazione del nome della colonna Momentum
    mom_col_candidates = ["Mom", "WML"]
    mom_col = None
    for col in df_mom.columns:
        if col.upper() in [c.upper() for c in mom_col_candidates]:
            mom_col = col
            break
            
    if mom_col is None:
        raise ValueError(f"Colonna momentum non trovata. Colonne presenti: {df_mom.columns.tolist()}")
        
    # Rinomina la colonna momentum in 'Mom' ed estrai solo quella
    df_mom = df_mom[[mom_col]].rename(columns={mom_col: "Mom"})
    
    # Selezione delle colonne obbligatorie dei 5 fattori
    required_ff = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    missing_ff = [col for col in required_ff if col not in df_ff.columns]
    if missing_ff:
        raise ValueError(f"Colonne dei 5 fattori mancanti nel dataset scaricato: {missing_ff}")
        
    df_ff = df_ff[required_ff]
    
    # Unione dei DataFrame
    df_merged = pd.merge(df_ff, df_mom, left_index=True, right_index=True, how="inner")
    
    # Conversione dei rendimenti da percentuale (%) a formato decimale
    df_merged = df_merged / 100.0
    
    # Ordinamento temporale crescente
    df_merged = df_merged.sort_index()
    
    # Filtro finale delle date (se fornite)
    if start_date:
        df_merged = df_merged[df_merged.index >= pd.to_datetime(start_date)]
    if end_date:
        df_merged = df_merged[df_merged.index <= pd.to_datetime(end_date)]
        
    return df_merged
