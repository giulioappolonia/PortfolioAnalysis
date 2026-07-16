# 📊 Guida all'Uso del Tool di Download Dati da Testfolio

Questa guida descrive come utilizzare lo script `download_testfolio.py` per scaricare automaticamente le serie storiche dei portafogli generati su **[Testfolio](https://testfol.io/)** partendo da un link di condivisione (es. `https://testfol.io/?s=0o9uikSA3ft`).

A differenza del metodo precedente che richiedeva di copiare manualmente i log di rete del browser, questo script automatizza interamente il processo di interrogazione e conversione dei dati.

---

## ⚙️ Requisiti di Sistema

Assicurati di avere installate le librerie Python per la gestione dei dati e delle richieste di rete:

```bash
pip install requests pandas
```

---

## 🚀 Come Utilizzare lo Script

Puoi avviare lo script direttamente da terminale posizionandoti nella cartella del progetto.

### Sintassi del Comando
```bash
python download_testfolio.py [URL_O_CODICE_BACKTEST] [PERCORSO_FILE_OUTPUT_CSV]
```

### Esempi Pratici

1. **Esecuzione predefinita (Scarica il backtest a 4 portafogli e salva in DatiInput):**
   ```bash
   python download_testfolio.py
   ```
   *Questo scaricherà i dati dal link preimpostato `https://testfol.io/?s=0o9uikSA3ft` e salverà i dati in `DatiInput/testfolio_backtest_data.csv`.*

2. **Scaricare un backtest specifico su un file CSV a scelta:**
   ```bash
   python download_testfolio.py "https://testfol.io/?s=0o9uikSA3ft" "DatiInput/analisi_4_portafogli.csv"
   ```

3. **Fornire solo il codice di condivisione:**
   ```bash
   python download_testfolio.py "0o9uikSA3ft" "DatiInput/output.csv"
   ```

---

## 🛠️ Come Funziona (Dettaglio Tecnico)

Lo script esegue i seguenti passaggi in background:
1. **Risoluzione del link:** Estrae l'identificativo univoco (es. `0o9uikSA3ft`) dall'URL.
2. **Download della Configurazione:** Interroga l'endpoint `https://testfol.io/api/link/{share_code}` per recuperare i parametri globali (es. data di inizio, valuta) e la composizione dei portafogli.
3. **Conversione dei Dati (Data Translation):** 
   - Converte i campi in formato *camelCase* nei rispettivi *snake_case* richiesti dall'API di elaborazione.
   - Trasforma la lista di asset `tickers` in un dizionario strutturato `allocation` (es. `{"VTSIM": 100}`).
4. **Esecuzione Backtest:** Invia un payload POST a `https://testfol.io/api/backtest`.
5. **Generazione CSV:** Parsa la serie storica della risposta (array dei timestamp in secondi UNIX e i valori storici di ciascun portafoglio) e crea un DataFrame Pandas allineando le date, salvando infine il file CSV strutturato.
