# Guida all'Integrazione dei Dati Storici DBMF (Pre-Quotazione)

Questa guida ti spiega come utilizzare lo script `integrate_dbmf.py` per inserire le due simulazioni storiche di DBMF pre-inception (da Testfolio e da Société Générale) nel file di dati predefinito `chart_default.csv`.

---

## Metodo 1: Integrazione dati da Testfolio (`DBMFSIM`)

1. Apri il browser e vai su **[Testfolio](https://testfol.io/)**.
2. Configura un portafoglio con il **100%** sull'asset `DBMFSIM`.
3. Imposta la data di inizio al **2000** (es. `2000-01-01`) e clicca su **Run Backtest**.
4. Premi **F12** sulla tastiera (o fai clic destro sulla pagina -> *Ispeziona*) per aprire gli Strumenti di Sviluppo del Browser.
5. Seleziona la scheda **Network** (o *Rete*) nella parte superiore del pannello.
6. Filtra i risultati selezionando la categoria **Fetch/XHR**.
7. Ricarica la pagina del browser (**F5**) e premi nuovamente su **Run Backtest** su Testfolio.
8. Nella scheda Network comparirà una chiamata di tipo `POST` denominata `backtest` (l'indirizzo completo della richiesta è `https://testfol.io/api/backtest`). Cliccaci sopra con il tasto destro.
9. Seleziona **Copy** -> **Copy response** (o *Copia risposta*).
10. Crea un nuovo file vuoto chiamato **`dbmfsim_raw.json`** all'interno della cartella principale del progetto (`PortfolioAnalysis`) e incolla al suo interno il testo che hai copiato (sarà un grosso blocco di testo in formato JSON). Salva il file.

---

## Metodo 2: Integrazione dati da Société Générale (`SG Trend Index`)

1. Vai sul portale ufficiale per gli indici: **[Societe Generale Prime Services Indices](https://cib.societegenerale.com/en/sg-prime-services-indices/)**.
2. Trova la riga corrispondente al **SG Trend Index** (o *SG CTA Index*).
3. Scarica la serie storica giornaliera in formato Excel (`.xlsx`) o CSV.
4. Rinomina il file scaricato in **`sg_trend_raw.xlsx`** (o **`sg_trend_raw.csv`** se in formato CSV) e salvalo all'interno della cartella principale del progetto (`PortfolioAnalysis`).

*Nota:* Lo script parserà automaticamente il file cercando la colonna delle date e dei valori, calcolando i rendimenti giornalieri, sottraendo la fee annua dell'ETF dello **0.85%** e aggiungendo un premio di efficienza (Fee Alpha) stimato al **+0.30% mensile** (circa +3.6% annuo), ricostruendo poi la serie a partire da una base di 10000.

---

## Come Avviare l'Integrazione

Una volta salvati uno o entrambi i file nella cartella principale del progetto, apri una finestra di terminale in questa cartella ed esegui:

```bash
python integrate_dbmf.py
```

### Cosa farà lo script:
1. Caricherà il file `chart_default.csv` esistente.
2. Se trova `dbmfsim_raw.json`, estrarrà i dati, li ricampionerà su base mensile e li normalizzerà su base 10000 inserendo la colonna **`DBMF_Testfolio`**.
3. Se trova `sg_trend_raw.xlsx`/`csv`, estrarrà i prezzi giornalieri, applicherà le rettifiche di commissioni ed efficienza, ricampionerà su base mensile e creerà la colonna **`DBMF_SG_Trend`**.
4. Inserirà i dati nel DataFrame riempiendo con `10000` i mesi antecedenti all'inizio della serie per evitare che le righe precedenti vengano eliminate a causa dell'allineamento.
5. Creerà un file di backup (`chart_default_backup.csv`) e aggiornerà `chart_default.csv`.

Una volta completato il comando, gli indici di esempio visualizzati all'avvio del tool comprenderanno automaticamente i nuovi dati storici e potrai analizzarli in tutte le tab del software!
