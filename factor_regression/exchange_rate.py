# factor_regression/exchange_rate.py

import os
import urllib.request
import zipfile
import io
import pandas as pd

LOCAL_RATES_PATH = os.path.join(os.path.dirname(__file__), "data", "usd_eur_rates.csv")

def get_exchange_rates() -> tuple[pd.DataFrame, bool, str]:
    """
    Carica i tassi di cambio EUR/USD. Tenta di scaricare gli ultimi dati da BCE.
    Se fallisce, usa la copia locale e segnala l'offline con l'ultima data disponibile.
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame con DatetimeIndex e colonna 'EUR_USD' (valore di 1 EUR in USD).
    is_online : bool
        True se i dati sono stati aggiornati via internet, False altrimenti.
    last_date_str : str
        Data dell'ultimo tasso di cambio disponibile.
    """
    is_online = False
    df_local = None
    
    # 1. Carica il file locale esistente come base di fallback
    if os.path.exists(LOCAL_RATES_PATH):
        try:
            df_local = pd.read_csv(LOCAL_RATES_PATH, index_col='Date', parse_dates=True)
            df_local = df_local.sort_index()
        except Exception:
            pass

    # 2. Tenta di scaricare l'aggiornamento online da BCE
    try:
        url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        # Timeout breve di 5 secondi per rilevare l'assenza di internet o blocchi di rete
        with urllib.request.urlopen(req, timeout=5) as response:
            zip_data = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            with z.open("eurofxref-hist.csv") as f:
                df_online = pd.read_csv(f)
                
        df_online['Date'] = pd.to_datetime(df_online['Date'])
        df_online = df_online[['Date', 'USD']].rename(columns={'USD': 'EUR_USD'})
        df_online = df_online.set_index('Date').sort_index()
        df_online = df_online[pd.to_numeric(df_online['EUR_USD'], errors='coerce').notna()]
        df_online['EUR_USD'] = df_online['EUR_USD'].astype(float)
        
        # Unisci i dati online con quelli storici pre-1999 (Marco Tedesco) presenti nel file locale
        if df_local is not None:
            # Prendi la parte pre-1999 dal locale
            df_pre1999 = df_local[df_local.index < pd.to_datetime('1999-01-01')]
            df_merged = pd.concat([df_pre1999, df_online])
            # Rimuovi duplicati mantenendo il più recente ed ordina
            df_merged = df_merged[~df_merged.index.duplicated(keep='first')].sort_index()
        else:
            df_merged = df_online
            
        # Salva la versione aggiornata sul disco per utilizzi offline futuri
        os.makedirs(os.path.dirname(LOCAL_RATES_PATH), exist_ok=True)
        df_merged.to_csv(LOCAL_RATES_PATH)
        is_online = True
        df = df_merged
    except Exception:
        # Fallback offline
        if df_local is not None:
            df = df_local
        else:
            raise RuntimeError("Errore: nessun database tassi locale trovato e connessione internet assente.")
            
    last_date = df.index.max()
    last_date_str = last_date.strftime("%Y-%m-%d")
    
    return df, is_online, last_date_str
