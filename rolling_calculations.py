import pandas as pd
import numpy as np

def calculate_rolling_returns(data, window_years=3):
    """Calcola i rendimenti rolling annualizzati."""
    window = window_years * 12
    # Aggiunge 1 per lavorare con i fattori di crescita, eleva alla potenza appropriata per annualizzare
    # Sottrarre 1 alla fine per tornare al rendimento percentuale annualizzato
    # .dropna() è importante qui per evitare NaN iniziali che falserebbero i calcoli successivi
    rolling_returns = (data.pct_change(periods=window) + 1) ** (12 / window) - 1
    # Rimuove le prime 'window' righe che saranno NaN
    return rolling_returns.dropna(how='all')

def calculate_min_median_by_window(data, max_window=20):
    """
    Calcola min, median e restituisce le distribuzioni rolling per diverse finestre.
    Restituisce: {asset: {'windows': [...], 'min_values': [...], 'median_values': [...], 'all_returns_by_window': {window_year: [list_of_returns], ...}}, ...}
    """
    results = {}
    for asset in data.columns:
        asset_data = data[asset].dropna()
        min_values = []
        median_values = []
        windows = []
        # Nuovo: Dizionario per memorizzare tutti i rendimenti per ogni finestra
        all_returns_by_window = {}

        # Per un corretto calcolo dei rolling return, servono almeno (window * 12 + 1) punti dati
        # Poiché pct_change(periods=N) richiede N+1 punti.
        # Se la finestra è in anni, i periodi sono window_years * 12.
        # Quindi, servono almeno (window_years * 12 + 1) punti dati.

        for window_years in range(1, max_window + 1):
            min_required_points = window_years * 12 + 1
            if len(asset_data) < min_required_points:
                # Se non ci sono abbastanza dati per questa finestra, salta
                continue

            # Calcola i rendimenti rolling per la finestra corrente
            rolling_returns = calculate_rolling_returns(asset_data.to_frame(), window_years)

            if not rolling_returns.empty:
                # Estrae la serie di rendimenti (sarà l'unica colonna)
                asset_rolling_series = rolling_returns.iloc[:, 0].dropna()

                if not asset_rolling_series.empty:
                    min_val = asset_rolling_series.min()
                    median_val = asset_rolling_series.median()

                    min_values.append(min_val)
                    median_values.append(median_val)
                    windows.append(window_years)
                    # Memorizza la lista completa dei rendimenti per questa finestra
                    all_returns_by_window[window_years] = asset_rolling_series.tolist()

        # Aggiunge la nuova chiave 'all_returns_by_window' al risultato
        results[asset] = {
            'windows': windows,
            'min_values': min_values,
            'median_values': median_values,
            'all_returns_by_window': all_returns_by_window # Aggiunto
        }
    return results


def calculate_risk_metrics(rolling_returns):
    """Calcola le metriche di rischio sui rendimenti rolling."""
    # Assicurati che rolling_returns non sia vuoto
    if rolling_returns is None or rolling_returns.empty:
        return pd.DataFrame()

    metrics = pd.DataFrame({
        'Min': rolling_returns.min(),
        'Max': rolling_returns.max(),
        'Media': rolling_returns.mean(),
        'Mediana': rolling_returns.median(),
        'Dev. Std': rolling_returns.std(),
        'Skewness': rolling_returns.skew(),
        'Kurtosis': rolling_returns.kurt(),
        '10° percentile': rolling_returns.quantile(0.1),
        '25° percentile': rolling_returns.quantile(0.25),
        '75° percentile': rolling_returns.quantile(0.75),
        '90° percentile': rolling_returns.quantile(0.9)
    })
    return metrics

def create_portfolio(data, weights):
    """Crea una serie storica del valore del portafoglio normalizzato."""
    if data is None or data.empty or not weights:
        return pd.DataFrame()

    # Seleziona solo le colonne presenti in data e con pesi validi
    valid_assets = [col for col in weights if col in data.columns and weights[col] > 0]

    if not valid_assets:
        return pd.DataFrame()

    portfolio_data = data[valid_assets].copy()

    # Normalizza le serie storiche degli asset validi
    # Usa .iloc[0] solo dopo aver eliminato i NaN iniziali per trovare il primo valore valido
    for col in valid_assets:
        first_valid_value = portfolio_data[col].dropna().iloc[0]
        portfolio_data[col] = portfolio_data[col] / first_valid_value

    # Applica i pesi e somma
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Portafoglio'] = 0.0 # Inizializza come float

    for col in valid_assets:
         # Usa .reindex(portfolio_data.index) per allineare prima della moltiplicazione
        portfolio['Portafoglio'] += portfolio_data[col].fillna(method='ffill').fillna(method='bfill') * weights[col] # Gestisce NaN con fillna

    # Rinomina la colonna come "Portafoglio" dopo la creazione
    portfolio = portfolio.rename(columns={portfolio.columns[0]: 'Portafoglio'})

    return portfolio

def create_short_names(full_names, max_length=15):
    """Crea nomi abbreviati eliminando vocali e spazi."""
    short_names = {}
    for name in full_names:
        if len(name) <= max_length:
            short_names[name] = name
        else:
            # Rimuove vocali (a, e, i, o, u) e spazi, poi prende i primi max_length caratteri
            abbreviated = ''.join([c for c in name if c.lower() not in 'aeiou ']).strip()
            if len(abbreviated) > max_length:
                 # Se ancora troppo lungo, taglia e aggiungi ...
                 abbreviated = abbreviated[:max_length-3] + '...'
            elif len(abbreviated) == 0: # Evita nomi vuoti
                 abbreviated = name[:max_length] + '...'
            short_names[name] = abbreviated if abbreviated else name # Fallback se abbreviato è vuoto
    return short_names
