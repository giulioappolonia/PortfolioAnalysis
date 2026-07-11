import os
import json
import numpy as np
import pandas as pd

def load_default_chart():
    csv_path = "chart_default.csv"
    backup_path = "chart_default_backup.csv"
    
    # Se esiste il backup pulito, carichiamo sempre dal backup per evitare di accumulare modifiche o date duplicate
    path_to_load = backup_path if os.path.exists(backup_path) else csv_path
    if not os.path.exists(path_to_load):
        raise FileNotFoundError(f"Impossibile trovare il file {path_to_load} nel percorso corrente.")
    
    print(f"[+] Caricamento dati da: {path_to_load}")
    df = pd.read_csv(path_to_load)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def save_default_chart(df):
    csv_path = "chart_default.csv"
    # Crea una copia di backup prima di sovrascrivere
    backup_path = "chart_default_backup.csv"
    if not os.path.exists(backup_path):
        import shutil
        shutil.copyfile(csv_path, backup_path)
        print(f"Creato backup di sicurezza del file originale in: {backup_path}")
    
    # Formatta l'indice come MM/YYYY per mantenere la compatibilità
    df_to_save = df.copy()
    df_to_save.index = df_to_save.index.strftime('%m/%Y')
    df_to_save.index.name = 'Date'
    df_to_save.to_csv(csv_path)
    print(f"Salvate modifiche con successo in: {csv_path}")

def resample_series_monthly(s):
    # Prova a usare 'ME' (Month End) richiesto da pandas recente, fallback su 'M' per versioni precedenti
    try:
        s_resampled = s.resample('ME').last()
    except ValueError:
        s_resampled = s.resample('M').last()
    
    # Forza la data al primo giorno del mese per allinearsi al formato %m/%Y di chart_default
    s_resampled.index = s_resampled.index.map(lambda x: x.replace(day=1))
    return s_resampled

def integrate_testfolio():
    json_path = "dbmfsim_raw.json"
    if not os.path.exists(json_path):
        print("[-] File 'dbmfsim_raw.json' non trovato. Salto l'integrazione di Testfolio.")
        return None
    
    print("[+] Trovato 'dbmfsim_raw.json'. Elaborazione in corso...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Estrazione della serie storica
    try:
        history = raw_data["charts"]["history"]
        timestamps, values = history
    except KeyError:
        # Se la risposta è leggermente diversa o contiene una lista di backtest
        try:
            backtests = raw_data.get("backtests", [])
            if backtests:
                history = raw_data["charts"]["history"]
                timestamps, values = history
            else:
                raise KeyError()
        except Exception:
            print("[-] Formato JSON di Testfolio non riconosciuto. Assicurati di aver copiato l'intera 'response' di rete.")
            return None
            
    dates = pd.to_datetime(timestamps, unit="s")
    s_daily_usd = pd.Series(values, index=dates)
    
    # CONVERSIONE DA USD A EUR
    try:
        print("[+] Caricamento tassi di cambio EUR/USD per conversione valuta...")
        from factor_regression.exchange_rate import get_exchange_rates
        rates_df, _, last_date = get_exchange_rates()
        rates_aligned = rates_df['EUR_USD'].reindex(s_daily_usd.index).ffill().bfill()
        
        # Converte da USD a EUR: Valore_EUR = Valore_USD / Tasso_EUR_USD
        s_daily = s_daily_usd / rates_aligned
        print("[+] Conversione da USD a EUR completata.")
    except Exception as e:
        print(f"[-] Impossibile convertire in EUR (errore tassi di cambio): {e}. Procedo con i valori originali in USD.")
        s_daily = s_daily_usd
    
    # Resample a mensile (valore di fine mese)
    s_monthly = resample_series_monthly(s_daily)
    
    # Normalizziamo in base al valore iniziale (portiamo a 10000 all'inizio della serie)
    first_valid_val = s_monthly.iloc[0]
    s_normalized = (s_monthly / first_valid_val) * 10000
    
    s_normalized.name = "DBMF_Testfolio"
    print(f"[+] Dati DBMFSIM di Testfolio elaborati. Periodo: {s_normalized.index.min().strftime('%m/%Y')} - {s_normalized.index.max().strftime('%m/%Y')}")
    return s_normalized

def integrate_socgen():
    # Cerca sia csv che xlsx
    csv_path = "sg_trend_raw.csv"
    xlsx_path = "sg_trend_raw.xlsx"
    
    df_sg = None
    if os.path.exists(xlsx_path):
        print("[+] Trovato 'sg_trend_raw.xlsx'. Lettura in corso...")
        df_sg = pd.read_excel(xlsx_path)
    elif os.path.exists(csv_path):
        print("[+] Trovato 'sg_trend_raw.csv'. Lettura in corso...")
        df_sg = pd.read_csv(csv_path)
    else:
        print("[-] Nessun file 'sg_trend_raw.csv' o 'sg_trend_raw.xlsx' trovato. Salto l'integrazione di Société Générale.")
        return None
    
    # Stampa le colonne per aiutare il debug
    print(f"[i] Colonne trovate nel file Société Générale: {list(df_sg.columns)}")
    
    # Trova la colonna Date e la colonna dell'indice
    date_col = None
    index_col = None
    
    for col in df_sg.columns:
        col_lower = str(col).lower()
        if 'date' in col_lower or 'data' in col_lower:
            date_col = col
        elif 'index' in col_lower or 'trend' in col_lower or 'cta' in col_lower or 'val' in col_lower:
            index_col = col
            
    if not date_col:
        # Assumi che la prima sia la data se non trovata per nome
        date_col = df_sg.columns[0]
    if not index_col:
        # Assumi che la seconda sia l'indice
        index_col = df_sg.columns[1]
        
    print(f"[+] Utilizzo colonna data: '{date_col}' e colonna valore: '{index_col}'")
    
    df_sg[date_col] = pd.to_datetime(df_sg[date_col], errors='coerce')
    df_sg.dropna(subset=[date_col, index_col], inplace=True)
    df_sg.set_index(date_col, inplace=True)
    df_sg.sort_index(inplace=True)
    
    # Serie storica originaria
    s_raw = df_sg[index_col].astype(float)
    
    # Calcola i rendimenti giornalieri
    daily_returns = s_raw.pct_change().dropna()
    
    # Modello di adeguamento commissioni (Fee DBMF + Fee Alpha)
    # Commissione annua ETF DBMF: 0.85% -> 0.0085 / 252 al giorno
    fee_daily = 0.0085 / 252
    
    # Fee Alpha (poiché DBMF replica a lordo delle fee degli hedge fund e tende a sovraperformare l'indice netto)
    # Stima consigliata: +0.30% mensile -> circa +3.6% annuo -> 0.036 / 252 al giorno
    alpha_daily = 0.036 / 252
    
    adjusted_returns = daily_returns - fee_daily + alpha_daily
    
    # Reimporta i rendimenti in un indice cumulativo a partire da 10000
    cum_values = [10000.0]
    for r in adjusted_returns:
        cum_values.append(cum_values[-1] * (1 + r))
        
    # Crea la serie storica finale rettificata
    s_adjusted_daily = pd.Series(cum_values, index=[s_raw.index[0]] + list(adjusted_returns.index))
    
    # Resample a mensile
    s_monthly = resample_series_monthly(s_adjusted_daily)
    s_monthly.name = "DBMF_SG_Trend"
    
    print(f"[+] Dati Société Générale elaborati e rettificati. Periodo: {s_monthly.index.min().strftime('%m/%Y')} - {s_monthly.index.max().strftime('%m/%Y')}")
    return s_monthly

def integrate_file(file_name, col_names):
    if not os.path.exists(file_name):
        print(f"[-] File '{file_name}' non trovato. Salto l'integrazione.")
        return None
    
    print(f"[+] Trovato '{file_name}'. Elaborazione in corso...")
    try:
        df = pd.read_csv(file_name)
        # Assicurati che Date sia convertito correttamente
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        res = {}
        for col in col_names:
            if col in df.columns:
                res[col] = df[col].astype(float)
            else:
                print(f"[-] Colonna '{col}' non trovata nel file '{file_name}'.")
        return res
    except Exception as e:
        print(f"[-] Errore nell'elaborazione del file '{file_name}': {e}")
        return None

def main():
    try:
        df_chart = load_default_chart()
        print(f"[+] File default caricato. Righe: {len(df_chart)}, Indici correnti: {list(df_chart.columns)}")
    except Exception as e:
        print(f"[-] Errore critico nel caricamento di chart_default.csv: {e}")
        return
    
    updated = False
    
    # Integra i file estesi caricati (chart (7) - chart (11))
    extended_files = {
        "chart (7).csv": ["SCV"],
        "chart (8).csv": ["MVOL"],
        "chart (9).csv": ["XDEM"],
        "chart (10).csv": ["EIMI"],
        "chart (11).csv": ["XDEV"]
    }
    
    for file_name, cols in extended_files.items():
        data = integrate_file(file_name, cols)
        if data:
            for name, s in data.items():
                if name in df_chart.columns:
                    df_chart.drop(columns=[name], inplace=True)
                # Re-normalizza al primo valore disponibile della serie
                first_val = s.dropna().iloc[0]
                s_normalized = (s / first_val) * 10000
                s_normalized = s_normalized.ffill()
                df_chart = df_chart.join(s_normalized, how='outer')
                print(f"[+] Integrato asset esteso {name} (normalizzato a 10000 in {s.dropna().index[0].strftime('%m/%Y')})")
                updated = True
                
    # Integra Testfolio
    s_testfolio = integrate_testfolio()
    if s_testfolio is not None:
        # Rimuove la colonna se già esistente per aggiornarla
        if s_testfolio.name in df_chart.columns:
            df_chart.drop(columns=[s_testfolio.name], inplace=True)
        
        # Merge
        df_chart = df_chart.join(s_testfolio, how='outer')
        print(f"[+] Integrato {s_testfolio.name} nel DataFrame.")
        updated = True
        
    # Integra Société Générale
    s_socgen = integrate_socgen()
    if s_socgen is not None:
        # Rimuove la colonna se già esistente
        if s_socgen.name in df_chart.columns:
            df_chart.drop(columns=[s_socgen.name], inplace=True)
            
        # Merge
        df_chart = df_chart.join(s_socgen, how='outer')
        print(f"[+] Integrato {s_socgen.name} nel DataFrame.")
        updated = True

    # Integra chart (5).csv
    c5_data = integrate_file("chart (5).csv", ["SCV+MOM"])
    if c5_data:
        for name, s in c5_data.items():
            if name in df_chart.columns:
                df_chart.drop(columns=[name], inplace=True)
            # Re-normalizza al primo valore disponibile della serie
            first_val = s.dropna().iloc[0]
            s_normalized = (s / first_val) * 10000
            s_normalized = s_normalized.ffill()
            df_chart = df_chart.join(s_normalized, how='outer')
            print(f"[+] Integrato {name} (normalizzato a 10000 in {s.dropna().index[0].strftime('%m/%Y')})")
            updated = True

    # Integra chart (6).csv
    c6_data = integrate_file("chart (6).csv", ["Gov global EUR", "SGLD"])
    if c6_data:
        for name, s in c6_data.items():
            if name in df_chart.columns:
                df_chart.drop(columns=[name], inplace=True)
            # Re-normalizza al primo valore disponibile della serie
            first_val = s.dropna().iloc[0]
            s_normalized = (s / first_val) * 10000
            s_normalized = s_normalized.ffill()
            df_chart = df_chart.join(s_normalized, how='outer')
            print(f"[+] Integrato {name} (normalizzato a 10000 in {s.dropna().index[0].strftime('%m/%Y')})")
            updated = True
        
    # Salva il file se almeno una colonna è stata aggiunta/aggiornata
    if updated:
        save_default_chart(df_chart)
    else:
        print("[-] Nessuna modifica effettuata (nessun file sorgente trovato).")

if __name__ == "__main__":
    main()
