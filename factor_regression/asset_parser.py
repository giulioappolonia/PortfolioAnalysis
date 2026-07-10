# factor_regression/asset_parser.py

import pandas as pd
import yfinance as yf
from typing import Union, List, Tuple
import io

def load_asset_from_yahoo(
    ticker: str,
    start_date: str,
    end_date: str,
    frequency: str
) -> pd.DataFrame:
    """
    Scarica i prezzi adjusted storici per un ticker da Yahoo Finance.
    
    Parameters:
    -----------
    ticker : str
        Il ticker dell'asset (es. 'SPY', 'AAPL').
    start_date : str
        Data di inizio in formato 'YYYY-MM-DD'.
    end_date : str
        Data di fine in formato 'YYYY-MM-DD'.
    frequency : str
        'daily' o 'monthly'.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con DatetimeIndex (tz-naive, ordinato) e colonna ['price']
    """
    # Scarichiamo con auto_adjust=True per avere i prezzi Close rettificati
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"Nessun dato trovato per il ticker '{ticker}' nelle date indicate.")
        
    # Appiattimento del MultiIndex se presente (es. yfinance scarica colonne multilivello)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if 'Close' not in df.columns:
        raise ValueError(f"Colonna 'Close' non trovata nei dati Yahoo Finance per il ticker '{ticker}'.")
        
    prices = df[['Close']].copy()
    prices.columns = ['price']
    
    # Rimozione fuso orario (tz-naive) e normalizzazione a mezzanotte
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    
    # Rimozione duplicati e ordinamento
    prices = prices[~prices.index.duplicated(keep='first')].sort_index()
    
    return prices


def load_asset_from_file(
    file,
    filename: str,
    date_col: str,
    price_col: str
) -> pd.DataFrame:
    """
    Legge un file CSV o Excel locale caricato dall'utente e restituisce i prezzi storici.
    Gestisce formati europei ed americani in modo deterministico.
    
    Parameters:
    -----------
    file : file-like object
        Il file caricato (Streamlit UploadedFile o file in memoria).
    filename : str
        Nome del file per determinare l'estensione (.csv, .xlsx, .xls).
    date_col : str
        Nome della colonna che contiene le date.
    price_col : str
        Nome della colonna che contiene i prezzi o rendimenti storici.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con DatetimeIndex (tz-naive, ordinato) e colonna ['price']
    """
    # Caricamento del file in base all'estensione
    if filename.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file)
    else:
        # Legge la prima riga come stringa per capire il delimitatore
        try:
            sample = file.read(2048)
            # Ripristina il puntatore
            file.seek(0)
            sample_str = sample.decode('utf-8', errors='ignore')
            if ';' in sample_str:
                df = pd.read_csv(file, sep=';', decimal=',', thousands='.')
            else:
                df = pd.read_csv(file, sep=',', decimal='.', thousands=',')
        except Exception:
            file.seek(0)
            df = pd.read_csv(file)
            
    # Validazione colonne
    if date_col not in df.columns:
        raise ValueError(f"Colonna date '{date_col}' non trovata nel file. Colonne disponibili: {df.columns.tolist()}")
    if price_col not in df.columns:
        raise ValueError(f"Colonna prezzi '{price_col}' non trovata nel file. Colonne disponibili: {df.columns.tolist()}")
        
    # Conversione date (con priorità giorno per formati europei)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    # Rimuove le righe con date non convertibili (NaT)
    df = df.dropna(subset=[date_col])
    
    # Conversione prezzi in numerico
    df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=[price_col])
    
    # Costruisci DataFrame pulito
    prices = df[[date_col, price_col]].copy()
    prices.columns = ['date', 'price']
    prices = prices.set_index('date')
    
    # Rimozione fuso orario e normalizzazione
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    
    # Ordinamento e rimozione duplicati
    prices = prices[~prices.index.duplicated(keep='first')].sort_index()
    
    return prices


def compute_returns(price_df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resampla i prezzi alla frequenza indicata e calcola i rendimenti percentuali decimali.
    Non applica forward-fill dei prezzi prima del calcolo per evitare distorsioni.
    
    Parameters:
    -----------
    price_df : pd.DataFrame
        DataFrame con DatetimeIndex e colonna 'price'.
    frequency : str
        'daily' o 'monthly'.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con colonne ['price', 'asset_return']
    """
    if frequency == 'monthly':
        # Pandas >= 2.2.0 supporta 'ME' per MonthEnd (in precedenza 'M')
        resampled_price = price_df['price'].resample('ME').last()
    else:
        resampled_price = price_df['price']
        
    # Calcolo dei rendimenti semplici (decimali)
    returns = resampled_price.pct_change()
    
    # Eliminiamo il primo valore NaN generato dal pct_change
    returns = returns.dropna()
    
    # Ricostruiamo il dataframe finale
    merged = pd.DataFrame(resampled_price).join(returns.rename('asset_return'), how='inner')
    return merged
