import os
import sys
import json
import re
import requests
import pandas as pd

def extract_share_code(url_or_code):
    """Extract the share code from a full Testfolio URL or return it if it's already a code."""
    url_or_code = url_or_code.strip()
    match = re.search(r"[?&]s=([a-zA-Z0-9_-]+)", url_or_code)
    if match:
        return match.group(1)
    # Check for direct URL segment
    match_segment = re.search(r"testfol\.io/api/link/([a-zA-Z0-9_-]+)", url_or_code)
    if match_segment:
        return match_segment.group(1)
    return url_or_code

def download_testfolio_data(share_code, output_csv="testfolio_data.csv", frequency="monthly"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    config_url = f"https://testfol.io/api/link/{share_code}"
    backtest_url = "https://testfol.io/api/backtest"
    
    print(f"[+] Recupero della configurazione del link condiviso: {share_code}...")
    res_config = requests.get(config_url, headers=headers)
    if res_config.status_code != 200:
        print(f"[-] Errore nel recupero della configurazione (Stato: {res_config.status_code})")
        print("Verifica che il codice o l'URL siano corretti.")
        return False
        
    link_data = res_config.json()
    
    # Costruiamo il payload per l'API backtest
    payload = {}
    
    # 1. Copiamo i parametri globali direttamente alla radice
    if "globalParams" in link_data:
        for k, v in link_data["globalParams"].items():
            payload[k] = v
            
    payload["start_date"] = payload.get("start_date") or "1800-01-01"
    payload["end_date"] = payload.get("end_date") or "2100-01-01"
    
    # 2. Mappiamo i portafogli a 'backtests'
    payload["backtests"] = []
    portfolio_names = []
    
    portfolios = link_data.get("portfolios", [])
    if not portfolios:
        print("[-] Nessun portafoglio trovato nella configurazione.")
        return False
        
    for i, p in enumerate(portfolios):
        name = p.get("name") or f"Portfolio_{i+1}"
        portfolio_names.append(name)
        
        backtest_item = {
            "invest_dividends": p.get("investDividends", True),
            "rebalance_freq": p.get("rebalanceFreq", "Yearly"),
            "rebalance_offset": p.get("rebalanceOffset", 0),
            "drag": p.get("drag", 0),
            "allocation": {t["ticker"].upper(): t["percent"] for t in p.get("tickers", [])},
            "absolute_dev": p.get("absoluteDev", 0),
            "relative_dev": p.get("relativeDev", 0)
        }
        
        # Gestione bande asimmetriche
        if p.get("rebalanceBandMode") != "SYMMETRIC":
            backtest_item["absolute_dev_below"] = p.get("absoluteDevBelow", 0)
            backtest_item["absolute_dev_above"] = p.get("absoluteDevAbove", 0)
            backtest_item["relative_dev_below"] = p.get("relativeDevBelow", 0)
            backtest_item["relative_dev_above"] = p.get("relativeDevAbove", 0)
            
        payload["backtests"].append(backtest_item)
        
    print(f"[+] Configurazione completata. Portafogli rilevati: {', '.join(portfolio_names)}")
    print("[+] Invio richiesta di backtest a Testfolio...")
    
    res_backtest = requests.post(backtest_url, json=payload, headers=headers)
    if res_backtest.status_code != 200:
        print(f"[-] Errore nell'esecuzione del backtest (Stato: {res_backtest.status_code})")
        try:
            err_data = res_backtest.json()
            print("Dettaglio errori:", json.dumps(err_data.get("errors", err_data), indent=2))
        except Exception:
            print(res_backtest.text[:500])
        return False
        
    backtest_data = res_backtest.json()
    
    # 3. Estrazione della serie storica
    try:
        history = backtest_data["charts"]["history"]
        if not history or len(history) < 2:
            print("[-] Dati storici non presenti nella risposta del backtest.")
            return False
            
        timestamps = history[0]
        portfolio_series = history[1:]
        
        # Creiamo un DataFrame giornaliero
        dates = pd.to_datetime(timestamps, unit="s")
        df = pd.DataFrame(index=dates)
        df.index.name = "Date"
        
        for name, series_data in zip(portfolio_names, portfolio_series):
            df[name] = series_data
            
        # Applicazione della frequenza desiderata
        if frequency == "monthly":
            print("[+] Ricampionamento dei dati a frequenza mensile (Month-End)...")
            try:
                df = df.resample('ME').last()
            except ValueError:
                df = df.resample('M').last()
            
            # Imposta la data al primo giorno del mese per allinearsi a chart_default.csv
            df.index = df.index.map(lambda x: x.replace(day=1))
            # Formatta le date nel formato MM/YYYY
            df.index = df.index.strftime('%m/%Y')
            print("[+] Formattazione date applicata: MM/YYYY")
            
        # Salvataggio
        df.to_csv(output_csv)
        print(f"[+] Dati salvati con successo!")
        if frequency == "monthly":
            print(f"    - Periodo: {df.index[0]} a {df.index[-1]}")
        else:
            print(f"    - Periodo: {df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')}")
        print(f"    - Righe caricate: {len(df)}")
        print(f"    - File salvato in: {os.path.abspath(output_csv)}")
        return True
    except Exception as e:
        print(f"[-] Errore nell'elaborazione dei dati della risposta: {e}")
        return False

if __name__ == "__main__":
    url_input = "https://testfol.io/?s=0o9uikSA3ft"
    csv_output = "DatiInput/testfolio_backtest_data.csv"
    frequency = "monthly"
    
    # Parsing degli argomenti riga di comando
    args = sys.argv[1:]
    
    # Se viene specificato --daily, usiamo la frequenza giornaliera
    if "--daily" in args:
        frequency = "daily"
        args.remove("--daily")
        
    if len(args) > 0:
        url_input = args[0]
    if len(args) > 1:
        csv_output = args[1]
        
    code = extract_share_code(url_input)
    print(f"[*] Elaborazione URL/Codice: {url_input} (Codice estratto: {code})")
    print(f"[*] Frequenza impostata: {frequency}")
    
    # Assicuriamoci che la cartella del file di output esista
    out_dir = os.path.dirname(csv_output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    success = download_testfolio_data(code, csv_output, frequency)
    if not success:
        sys.exit(1)
