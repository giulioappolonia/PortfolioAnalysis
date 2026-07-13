import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile

# Aggiunge la cartella src al percorso di ricerca dei moduli
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data # Assumi che esista data_loader.py con la funzione load_data
from rolling_calculations import (calculate_rolling_returns, calculate_min_median_by_window, calculate_risk_metrics, create_portfolio) # Assumi esista rolling_calculations.py
from plots import (plot_rolling_returns, plot_boxplot, plot_violinplot, plot_min_vs_window, plot_median_vs_window, plot_combined_min_median, plot_detailed_window_analysis, plot_overlaid_histogram, plot_single_histogram_with_normal) # Assumi esista plots.py

from risk_metrics import PortfolioRiskMetrics # Importa la classe delle metriche di rischio
import math # Needed for sqrt(12)

# Dizionario contenente le spiegazioni dettagliate per ciascuna metrica.
METRIC_EXPLANATIONS = {
    # --- 1. Fondamenta (Metriche di Base) ---
    "Total Return (%)": {
        "Cos'è": "Guadagno complessivo cumulato dall'inizio alla fine.",
        "Come viene calcolato": "(Valore Finale - Iniziale) / Iniziale.",
        "Cosa indica": "Il risultato finale assoluto dell'investimento.",
        "Meglio": "**Alto**.",
    },
    "Annualized Return (%)": {
        "Cos'è": "Guadagno medio annuo composto dell'investimento.",
        "Come viene calcolato": "Rendimento totale proiettato su un anno standardizzato.",
        "Cosa indica": "La velocità media di crescita del capitale.",
        "Meglio": "**Alto**. Più è alto, maggiore è il profitto.",
    },
    "Annualized Volatility (%)": {
        "Cos'è": "Misura di quanto il prezzo oscilla attorno alla media.",
        "Come viene calcolato": "Deviazione standard dei rendimenti annualizzata.",
        "Cosa indica": "Incertezza e instabilità. Alta volatilità significa forti sbalzi di prezzo.",
        "Meglio": "**Basso**. Indica un andamento più stabile e prevedibile.",
    },
    "Max Drawdown (%)": {
        "Cos'è": "La peggiore perdita percentuale registrata dai massimi storici.",
        "Come viene calcolato": "Massimo calo dal picco precedente al minimo successivo.",
        "Cosa indica": "Il rischio massimo storico. Quanto avresti perso nel momento peggiore.",
        "Meglio": "**Basso (vicino a 0)**. Indica perdite massime contenute.",
    },
    "Downside Risk (%)": {
        "Cos'è": "Volatilità considerata solo quando i prezzi scendono.",
        "Come viene calcolato": "Deviazione standard dei soli rendimenti negativi.",
        "Cosa indica": "Il vero rischio di perdere denaro, ignorando la volatilità 'positiva' (rialzi).",
        "Meglio": "**Basso**.",
    },
    "VaR_Returns(95%) (%)": { 
        "Cos'è": "Value at Risk. Massima perdita periodica attesa nel 95% dei casi.",
        "Come viene calcolato": "Quantile 5% dei rendimenti periodici.",
        "Cosa indica": "Il rischio 'normale' di mercato su base periodica (es. mensile).",
        "Meglio": "**Basso (vicino a 0)**.",
    },
    # --- 2. Efficienza Classica e Code ---
    "Sharpe Ratio": {
        "Cos'è": "Rendimento per unità di rischio totale (volatilità).",
        "Come viene calcolato": "Rendimento / Volatilità.",
        "Cosa indica": "Efficienza classica. Quanto 'paga' assumersi dei rischi standard.",
        "Meglio": "**Alto**.",
    },
    "Sortino Ratio": {
        "Cos'è": "Rendimento per unità di rischio 'cattivo' (perdite).",
        "Come viene calcolato": "Rendimento / Downside Risk.",
        "Cosa indica": "Efficienza nel generare profitti minimizzando solo le perdite (non le oscillazioni).",
        "Meglio": "**Alto**.",
    },
    "Calmar Ratio": {
        "Cos'è": "Rendimento rispetto al peggior crollo storico.",
        "Come viene calcolato": "Rendimento / |Max Drawdown|.",
        "Cosa indica": "Resilienza: capacità di recuperare e guadagnare dopo il peggior disastro.",
        "Meglio": "**Alto**.",
    },
    "CVaR_Returns(95%) (%)": { 
        "Cos'è": "Conditional VaR. Perdita media periodica negli scenari peggiori.",
        "Come viene calcolato": "Media delle perdite che superano il VaR.",
        "Cosa indica": "Quanto ci si aspetta di perdere quando le cose vanno molto male.",
        "Meglio": "**Basso (vicino a 0)**.",
    },
    # --- 3. Analisi Avanzata dello Stress ---
    "Ulcer Index": {
        "Cos'è": "Misura lo stress dell'investitore combinando profondità e durata dei cali.",
        "Come viene calcolato": "Media quadratica di tutti i drawdown.",
        "Cosa indica": "La 'quantità di dolore' sofferta. Penalizza periodi lunghi e profondi di perdita.",
        "Meglio": "**Basso**. Indica cali brevi e poco profondi.",
    },
    "Ulcer Performance Index": {
        "Cos'è": "Rendimento ottenuto per ogni unità di 'stress' (Ulcer Index).",
        "Come viene calcolato": "Rendimento Annualizzato / Ulcer Index.",
        "Cosa indica": "Efficienza nel generare profitti minimizzando la sofferenza dei cali.",
        "Meglio": "**Alto**. Indica ottimi ritorni con poco stress.",
    },
    "DaR(95%) (%)": { 
        "Cos'è": "Drawdown at Risk. La perdita che non dovrebbe essere superata nel 95% dei casi peggiori.",
        "Come viene calcolato": "Soglia del 5% peggiore dei drawdown storici.",
        "Cosa indica": "Quanto potresti perdere in una situazione di mercato negativa ma non estrema.",
        "Meglio": "**Basso (vicino a 0)**.",
    },
    "CDaR(95%) (%)": { 
        "Cos'è": "Conditional DaR. La perdita media che si verifica negli scenari estremi (oltre il DaR).",
        "Come viene calcolato": "Media dei drawdown peggiori del 5%.",
        "Cosa indica": "Il danno atteso durante un crollo di mercato grave ('Cigno Nero').",
        "Meglio": "**Basso (vicino a 0)**.",
    },
    # --- 4. Alternative Portfolio Theory ---
    "Pitfall Indicator": {
        "Cos'è": "Indica quanto i crolli sono 'sorprendenti' rispetto alla normale volatilità.",
        "Come viene calcolato": "|CDaR| / Volatilità Annualizzata.",
        "Cosa indica": "Rischio nascosto. Se alto, l'asset sembra tranquillo ma ha crolli improvvisi e violenti.",
        "Meglio": "**Basso**. Indica che i crolli sono proporzionati alla volatilità.",
    },
    "Penalized Risk (%)": {
        "Cos'è": "Rischio totale che pesa sia la durata dei cali (Ulcer) che la loro violenza estrema (Pitfall).",
        "Come viene calcolato": "Ulcer Index * Pitfall Indicator.",
        "Cosa indica": "Una visione completa del rischio: quanto a lungo perdi e quanto violentemente.",
        "Meglio": "**Basso**.",
    },
    "Serenity Ratio": {
        "Cos'è": "La metrica definitiva di efficienza nella teoria alternativa.",
        "Come viene calcolato": "Rendimento Annualizzato / Penalized Risk.",
        "Cosa indica": "Generazione di rendimento con il minimo rischio totale (stress + eventi estremi).",
        "Meglio": "**Alto**.",
    },
}

st.set_page_config(page_title="Analizzatore Rolling Returns & Rischio", page_icon="📊", layout="wide")
st.title("📊 Analizzatore Rolling Returns & Rischio")
st.markdown("""
Questa applicazione permette di analizzare i rendimenti rolling annualizzati e le metriche di rischio di diversi indici finanziari.
Seleziona le opzioni dalla barra laterale e carica uno o più file **CSV o Excel** con i tuoi dati.
""")

with st.sidebar:
    st.header("Opzioni")
    uploaded_files = st.file_uploader("Carica uno o più file CSV o Excel", type=["csv", "xls", "xlsx"], accept_multiple_files=True)
    
    keep_default = False
    if uploaded_files:
        keep_default = st.checkbox("Mantieni dati di esempio", value=False, help="Spunta per mantenere gli indici di esempio precaricati insieme ai file caricati.")
    
    st.markdown("---")
    st.markdown("### Configurazione Valute")
    reg_display_currency = st.selectbox(
        "Valuta di Visualizzazione",
        options=["EUR", "USD"],
        index=0,
        key="reg_display_currency",
        help="Seleziona la valuta in cui visualizzare i rendimenti effettivi, stimati e la scomposizione della performance."
    )
    
    # Valuta di input per ciascun file caricato
    input_currencies = {}
    if uploaded_files:
        st.markdown("**Valuta dei File Caricati:**")
        for uploaded_file in uploaded_files:
            input_currencies[uploaded_file.name] = st.selectbox(
                f"Valuta per {uploaded_file.name}",
                options=["EUR", "USD"],
                index=0,
                key=f"input_currency_{uploaded_file.name}",
                help=f"Seleziona la valuta nativa dei dati in {uploaded_file.name}."
            )
    st.markdown("---")
    
    analysis_mode = st.radio("Modalità di analisi", ["Confronto Completo", "Indice Singolo", "Confronto Indici", "Portafoglio"], index=0)

    # Opzioni per la modalità Confronto Completo
    max_indices = None # Inizializza a None
    if analysis_mode == "Confronto Completo":
        limit_indices = st.checkbox("Limita numero di indici", value=False)
        if limit_indices:
            max_indices = st.slider("Numero massimo di indici", min_value=2, max_value=20, value=10, step=1)

    # Tasso privo di rischio per metriche (periodico - mensile) - Default 0.0
    # Si potrebbe rendere questo un input utente, ma per semplicità è fisso qui.
    # Se i dati non fossero mensili, questo valore e il fattore di annualizzazione andrebbero adeguati.
    # annual_risk_free_rate = st.slider("Tasso Privo di Rischio Annuale (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f") / 100.0
    # period_risk_free_rate = (1 + annual_risk_free_rate)**(1/12) - 1 # Converti annuale in mensile
    period_risk_free_rate = 0.0 # Mantenuto fisso a 0.0 come da ipotesi

def main():
    input_currencies = {}
    combined_data = None
    loaded_dfs = {} # Dizionario per conservare i DataFrames caricati da ciascun file

    if uploaded_files:
        with st.expander("Log Elaborazione", expanded=False):
            st.info(f"Caricati {len(uploaded_files)} file. Elaborazione...")
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    try:
                        temp_input_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_input_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                        data = None
                        # file_identifier non sembra essere usato, lo rimuovo o lo lascio se serve debug
                        
                        if file_extension in [".csv", ".xls", ".xlsx"]:
                            st.write(f"Processando file {file_extension}: {uploaded_file.name}")
                            data = load_data(temp_input_path)
                        else:
                            st.error(f"Tipo di file non supportato: {uploaded_file.name} ({file_extension}). Salto.")
                            data = None

                        if data is not None and not data.empty:
                            # Le colonne nel DataFrame 'data' mantengono i loro nomi originali dal file
                            loaded_dfs[uploaded_file.name] = data
                            st.success(f"File {uploaded_file.name} caricato e processato con successo. Colonne caricate: {list(data.columns)}")
                        elif data is not None and data.empty:
                            st.warning(f"Il file {uploaded_file.name} non contiene dati validi dopo l'elaborazione.")

                    except Exception as e:
                        st.error(f"Errore generico durante l'elaborazione del file {uploaded_file.name}: {e}")
        
        # Carica il file di default se richiesto dall'utente tramite checkbox
        if keep_default:
            default_file = os.path.join("data", "chart_default.csv")
            if os.path.exists(default_file):
                try:
                    data = load_data(default_file)
                    if data is not None and not data.empty:
                        loaded_dfs[default_file] = data
                    else:
                        st.warning("Il file di esempio non contiene dati validi.")
                except Exception as e:
                    st.error(f"Errore nel caricamento del file di esempio: {e}")
    else:
        # Logica per caricare il file di default se nessun file è stato caricato
        default_file = os.path.join("data", "chart_default.csv")
        if os.path.exists(default_file):
            st.info("Nessun file caricato. Caricamento dati di esempio...")
            try:
                data = load_data(default_file)
                if data is not None and not data.empty:
                    loaded_dfs[default_file] = data
                    # st.success(f"Dati di esempio caricati con successo.") 
                else:
                    st.warning("Il file di esempio non contiene dati validi.")
            except Exception as e:
                st.error(f"Errore nel caricamento del file di esempio: {e}")

    # Dopo aver processato tutti i file (caricati o default), combina i DataFrames
    if loaded_dfs:
        # Conversione valute allineando alla valuta di visualizzazione reg_display_currency
        rates_df = None
        rates_m = None
        reg_display_currency = st.session_state.get("reg_display_currency", "EUR")
        
        for file_name, df in list(loaded_dfs.items()):
            input_curr = "EUR" if os.path.basename(file_name) == "chart_default.csv" else input_currencies.get(file_name, "EUR")
            
            if input_curr != reg_display_currency:
                try:
                    if rates_df is None:
                        from factor_regression.exchange_rate import get_exchange_rates
                        rates_df, is_online, _ = get_exchange_rates()
                        try:
                            rates_m = rates_df['EUR_USD'].resample('ME').last()
                        except ValueError:
                            rates_m = rates_df['EUR_USD'].resample('M').last()
                        rates_m.index = rates_m.index.map(lambda x: x.replace(day=1))
                    
                    rates_aligned = rates_m.reindex(df.index).ffill().bfill()
                    
                    for col in df.columns:
                        if input_curr == "USD" and reg_display_currency == "EUR":
                            df[col] = df[col] / rates_aligned
                        elif input_curr == "EUR" and reg_display_currency == "USD":
                            df[col] = df[col] * rates_aligned
                        
                        first_valid_idx = df[col].first_valid_index()
                        if first_valid_idx is not None:
                            first_val = df.loc[first_valid_idx, col]
                            if first_val != 0:
                                df[col] = (df[col] / first_val) * 10000
                    
                    loaded_dfs[file_name] = df
                except Exception as currency_err:
                    st.error(f"Errore nella conversione valuta per il file {file_name}: {currency_err}")
        # Se siamo nel caso di default (nessun upload), non mostriamo "Combinazione dati..." se c'è un solo file
        if len(loaded_dfs) > 1 or uploaded_files:
             st.info("Combinazione dei dati caricati...")
        
        try:
            # Utilizza pd.concat e gestisci i nomi duplicati
            combined_data = pd.concat(loaded_dfs.values(), axis=1, join='outer')

            # Gestione nomi colonne duplicati dopo pd.concat
            cols = pd.Series(combined_data.columns)
            for dupl in combined_data.columns[combined_data.columns.duplicated(keep='first')].unique():
                dupl_cols_indices = [i for i, col in enumerate(combined_data.columns) if col == dupl]
                new_names = [f"{dupl}_{j+1}" for j in range(len(dupl_cols_indices))]
                for i, col_idx in enumerate(dupl_cols_indices):
                    combined_data.columns.values[col_idx] = new_names[i]


            # Assicurati che l'indice sia datetime
            if not pd.api.types.is_datetime64_any_dtype(combined_data.index):
                # Questo blocco potrebbe non essere necessario se load_data ha successo, ma serve come fallback
                # Cerca di convertire l'indice, gestendo potenziali errori
                try:
                    combined_data.index = pd.to_datetime(combined_data.index)
                except Exception as e:
                    st.error(f"Errore nella conversione dell'indice a datetime: {e}")
                    combined_data = None # Imposta a None se la conversione fallisce


            if combined_data is not None:
                # Ordina per indice temporale (utile dopo join='outer')
                combined_data.sort_index(inplace=True)
                if uploaded_files: # Mostra info solo se utente ha caricato file, per non intasare la vista default
                    st.success("Dati combinati con successo.")
                    st.write("Anteprima dei dati combinati:")
                    st.dataframe(combined_data.head())

        except Exception as e:
            st.error(f"Errore durante la combinazione o la pulizia dei dati: {e}")
            combined_data = None
    else:
        # st.warning("Nessun file valido è stato caricato o processato con successo.")
        # Non serve warning qui se gestiamo il caso "nessun file" nell'else finale di main
        combined_data = None

    # Il resto della logica di app.py procede solo se combined_data è stato creato con successo e non è vuoto
    if combined_data is not None and not combined_data.empty:
        # Filtra le colonne per rimuovere quelle non numeriche prima di analizzarle come indici
        available_indices = combined_data.select_dtypes(include=np.number).columns.tolist()

        # --- Logica per la selezione degli indici basata su analysis_mode ---
        selected_indices = []
        if analysis_mode == "Indice Singolo":
            if available_indices:
                selected_index = st.sidebar.selectbox("Seleziona un indice", available_indices)
                selected_indices = [selected_index] if selected_index else []
            else:
                st.warning("Nessun indice numerico disponibile nei dati caricati per l'analisi singola.")
                selected_indices = []
        elif analysis_mode == "Confronto Indici":
            if available_indices:
                default_selection = available_indices[:min(3, len(available_indices))]
                selected_indices = st.sidebar.multiselect("Seleziona gli indici da confrontare", available_indices, default=default_selection)
            else:
                st.warning("Nessun indice numerico disponibile nei dati caricati per il confronto.")
                selected_indices = []
        elif analysis_mode == "Portafoglio":
            if available_indices:
                default_selection = available_indices[:min(3, len(available_indices))]
                selected_portfolio_assets = st.sidebar.multiselect("Seleziona gli indici per il portafoglio", available_indices, default=default_selection)
                st.sidebar.markdown("### Pesi del portafoglio")
                weights = {}
                if selected_portfolio_assets:
                    initial_weight = 1.0/len(selected_portfolio_assets) if selected_portfolio_assets else 0.0
                    for idx in selected_portfolio_assets:
                        weights[idx] = st.sidebar.slider(f"Peso di {idx}", min_value=0.0, max_value=1.0, value=initial_weight, step=0.01, format="%.2f", key=f"weight_{idx}")

                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v/total_weight for k, v in weights.items()}
                        st.sidebar.info(f"Totale pesi normalizzati: {sum(weights.values()):.2f}")

                    if selected_portfolio_assets and all(asset in combined_data.columns for asset in selected_portfolio_assets):
                        st.info("Creazione portafoglio...")
                        try:
                            portfolio_data = create_portfolio(combined_data[selected_portfolio_assets].dropna(), weights)
                            if portfolio_data is not None and not portfolio_data.empty:
                                # Rimuovi la colonna 'Portafoglio' se già esiste per evitare duplicati
                                if 'Portafoglio' in combined_data.columns:
                                    combined_data = combined_data.drop(columns=['Portafoglio'])

                                # Assicurati che l'indice del portafoglio sia datetime
                                if not pd.api.types.is_datetime64_any_dtype(portfolio_data.index):
                                     try:
                                         portfolio_data.index = pd.to_datetime(portfolio_data.index)
                                     except Exception as e:
                                         st.error(f"Errore nella conversione dell'indice del portafoglio a datetime: {e}")
                                         portfolio_data = None # Imposta a None se la conversione fallisce

                                if portfolio_data is not None:
                                    combined_data = pd.concat([combined_data, portfolio_data], axis=1, join='outer')
                                    combined_data.sort_index(inplace=True)
                                    selected_indices = ["Portafoglio"] # Seleziona solo il portafoglio per l'analisi successiva
                                    available_indices = combined_data.select_dtypes(include=np.number).columns.tolist() # Aggiorna la lista degli indici disponibili
                                    st.success("Portafoglio creato con successo.")
                                else:
                                    st.warning("La creazione del portafoglio non ha prodotto dati validi.")
                                    selected_indices = []
                                    #combined_data = None # Potrebbe essere meglio non azzerare combined_data qui? Dipende dal flusso desiderato.
                            else:
                                st.warning("La creazione del portafoglio non ha prodotto dati validi.")
                                selected_indices = []
                                #combined_data = None
                        except Exception as e:
                            st.error(f"Errore nella creazione del portafoglio: {e}")
                            selected_indices = []
                            #combined_data = None
                    elif selected_portfolio_assets:
                        st.warning("Impossibile creare il portafoglio. Verifica la selezione degli indici.")
                        selected_indices = []
                    else:
                        st.info("Seleziona gli indici per creare un portafoglio.")
                        selected_indices = []
                else: # Nessun asset selezionato per il portafoglio
                    selected_indices = []
            else:
                st.warning("Nessun indice numerico disponibile nei dati caricati per la creazione del portafoglio.")
                selected_indices = []
        else: # Confronto Completo
            if available_indices:
                if 'limit_indices' in locals() and limit_indices and max_indices is not None:
                     if len(available_indices) > max_indices:
                         selected_indices = available_indices[:max_indices]
                         st.sidebar.warning(f"Analizzando solo i primi {max_indices} indici: {', '.join(selected_indices)}")
                     else:
                         selected_indices = available_indices
                else:
                     selected_indices = available_indices
            else:
                st.warning("Nessun indice numerico disponibile nei dati caricati per il confronto completo.")
                selected_indices = []

        # --- Fine logica per la selezione degli indici ---

        # Procedi con i calcoli e i grafici solo se ci sono indici selezionati E combined_data è valido
        if len(selected_indices) > 0 and combined_data is not None and not combined_data.empty:

            # Seleziona solo i dati per l'analisi dagli indici selezionati
            # Assicurati che le colonne selezionate esistano nel DataFrame dopo la pulizia
            analysis_data = combined_data[[idx for idx in selected_indices if idx in combined_data.columns]].dropna()
            actual_analysis_indices = analysis_data.columns.tolist() # Usa actual columns after filtering

            if not analysis_data.empty:
                with st.expander("Dettagli Calcoli e Metriche", expanded=False):
                    # Calcola rendimenti periodici (mensili) dai dati di prezzo/NAV per le metriche di rischio
                    st.info("Calcolo dei rendimenti periodici per metriche di rischio...")
                    returns_data = analysis_data.pct_change().dropna(how='all') # Rimuove righe completamente NaN

                    # Calcola le metriche di rischio per ogni asset sull'intero periodo
                    risk_metrics_results = {}
                    # Assumendo dati mensili, il fattore di annualizzazione per std dev è sqrt(12)
                    annualization_factor = math.sqrt(12)

                    st.info("Calcolo delle metriche di rischio sull'intero periodo...")
                    if not returns_data.empty:
                        for asset in actual_analysis_indices:
                            # Assicurati che l'asset esista sia nei dati di prezzo che nei dati di rendimento
                            if asset in analysis_data.columns and asset in returns_data.columns:
                                asset_nav_series = analysis_data[asset].dropna() # Rimuove NaN iniziali per il calcolo del total return
                                asset_returns_series = returns_data[asset].dropna() # Rimuove NaN iniziali per il calcolo delle std dev etc.

                                # Allinea le serie sull'indice comune dopo la rimozione dei NaN
                                common_index_metrics = asset_nav_series.index.intersection(asset_returns_series.index)
                                aligned_nav_series = asset_nav_series.loc[common_index_metrics]
                                aligned_returns_series = asset_returns_series.loc[common_index_metrics]


                                if not aligned_returns_series.empty and not aligned_nav_series.empty and len(aligned_nav_series) > 1:
                                    try:
                                        metrics_calculator = PortfolioRiskMetrics(
                                            nav_series=aligned_nav_series,
                                            returns_series=aligned_returns_series,
                                            annualization_factor=annualization_factor,
                                            risk_free_rate=period_risk_free_rate # Usa il tasso periodico
                                        )
                                        metrics = metrics_calculator.get_all_metrics()
                                        for metric_name, value in metrics.items():
                                            if metric_name not in risk_metrics_results:
                                                risk_metrics_results[metric_name] = {}
                                            risk_metrics_results[metric_name][asset] = value

                                    except ValueError as e:
                                        st.warning(f"Impossibile calcolare metriche di rischio per {asset}: {e}")
                                        # Inizializza con NaN se il calcolo fallisce
                                        # Lista delle metriche attese da get_all_metrics per inizializzazione con NaN
                                        expected_metrics_keys = list(METRIC_EXPLANATIONS.keys()) # Usa le chiavi del dizionario spiegazioni
                                        for metric_name_key in expected_metrics_keys:
                                             if metric_name_key not in risk_metrics_results:
                                                 risk_metrics_results[metric_name_key] = {}
                                             risk_metrics_results[metric_name_key][asset] = np.nan

                                    except Exception as e:
                                        st.error(f"Errore durante il calcolo delle metriche per {asset}: {e}")
                                        # Inizializza con NaN se il calcolo fallisce
                                        expected_metrics_keys = list(METRIC_EXPLANATIONS.keys())
                                        for metric_name_key in expected_metrics_keys:
                                             if metric_name_key not in risk_metrics_results:
                                                 risk_metrics_results[metric_name_key] = {}
                                             risk_metrics_results[metric_name_key][asset] = np.nan
                                else:
                                    st.warning(f"Serie di prezzi o rendimenti insufficiente per {asset} dopo la pulizia dei NaN ({len(aligned_nav_series)} punti dati validi). Impossibile calcolare metriche di rischio.")
                                    # Inizializza con NaN se i dati sono insufficienti
                                    expected_metrics_keys = list(METRIC_EXPLANATIONS.keys())
                                    for metric_name_key in expected_metrics_keys:
                                         if metric_name_key not in risk_metrics_results:
                                             risk_metrics_results[metric_name_key] = {}
                                         risk_metrics_results[metric_name_key][asset] = np.nan

                            else:
                                st.warning(f"Dati di prezzo o rendimenti mancanti per l'asset {asset}. Impossibile calcolare metriche di rischio.")
                                # Inizializza con NaN se i dati sono mancanti
                                expected_metrics_keys = list(METRIC_EXPLANATIONS.keys())
                                for metric_name_key in expected_metrics_keys:
                                    if metric_name_key not in risk_metrics_results:
                                        risk_metrics_results[metric_name_key] = {}
                                    risk_metrics_results[metric_name_key][asset] = np.nan


                        # Convert the results dictionary into a DataFrame
                        if risk_metrics_results:
                            risk_metrics_df = pd.DataFrame(risk_metrics_results).T # Transpose to have Metrics as rows, Assets as columns
                            # Optional: Reorder rows based on METRIC_EXPLANATIONS keys ordered_metric_names = list(METRIC_EXPLANATIONS.keys())
                            ordered_metric_names = list(METRIC_EXPLANATIONS.keys())
                            risk_metrics_df = risk_metrics_df.reindex(ordered_metric_names)
                        else:
                            risk_metrics_df = pd.DataFrame() # Empty DataFrame if no metrics were calculated

                    else:
                        st.warning("Impossibile calcolare i rendimenti periodici per gli indici selezionati.")
                        risk_metrics_df = pd.DataFrame() # Empty DataFrame if no returns data

                    # Calcola i dati per l'analisi per finestre
                    st.info("Preparazione dati per Analisi per Finestre...")
                    min_median_data = {} # Initialize as empty
                    try:
                        min_median_data = calculate_min_median_by_window(analysis_data)
                    except Exception as e:
                        st.error(f"Errore durante il calcolo min/median per finestre: {e}")
                        min_median_data = {} # Set to empty on error

                # --- Aggiunge le schede di primo livello ---
                tab_rolling, tab_windows, tab_risk_metrics, tab_factors = st.tabs([
                    "Rendimenti Rolling & Distribuzioni",
                    "Andamento per Finestre",
                    "Metriche di Rischio",
                    "Regressione Fattoriale"
                ])

                # --- Contenuto per la scheda "Rendimenti Rolling & Distribuzioni" ---
                with tab_rolling:
                    # Slider inserito direttamente all'interno della tab
                    rolling_years = st.slider("Periodo Rolling (anni)", min_value=1, max_value=20, value=3, step=1, key="rolling_years_slider")

                    # Calcola i rendimenti rolling per il periodo specificato
                    min_valid_data_points_rolling = rolling_years * 12 + 1 # Assuming monthly data
                    has_enough_data_rolling = any(analysis_data[col].dropna().shape[0] >= min_valid_data_points_rolling for col in analysis_data.columns)

                    rolling_returns = pd.DataFrame() # Initialize as empty
                    if has_enough_data_rolling:
                         try:
                             rolling_returns = calculate_rolling_returns(analysis_data, rolling_years)
                         except Exception as e:
                             st.error(f"Errore durante il calcolo dei rendimenti rolling: {e}")
                             rolling_returns = pd.DataFrame() # Set to empty on error
                    else:
                        st.warning(f"Non abbastanza dati per calcolare i rendimenti rolling di {rolling_years} anni per nessuno degli asset selezionati.")

                    st.subheader(f"Analisi per il periodo rolling di {rolling_years} anni")
                    if rolling_returns is not None and not rolling_returns.empty:
                        # Visualizza Statistiche sui rendimenti rolling
                        st.subheader("Statistiche sui Rendimenti Rolling")
                        # calculate_risk_metrics qui calcola statistiche DESCRITTIVE sui *rendimenti rolling*, non le metriche di rischio della classe
                        metrics_rolling = calculate_risk_metrics(rolling_returns)
                        if not metrics_rolling.empty:
                           st.dataframe(metrics_rolling.style.format("{:.2%}"))
                        else:
                            st.warning("Impossibile calcolare le statistiche sui rendimenti rolling.")

                        # Plot Rendimenti Rolling Annualizzati
                        st.subheader("Rendimenti Rolling Annualizzati")
                        fig_returns = plot_rolling_returns(rolling_returns, title=f"Rendimenti Rolling ({rolling_years} anni)")
                        st.plotly_chart(fig_returns, width='stretch')

                        # Box Plot, Violin Plot e Istogramma
                        st.subheader("Distribuzione dei Rendimenti Rolling")
                        tab_box, tab_violin, tab_hist = st.tabs(["Box Plot", "Violin Plot", "Istogramma di Frequenza"])
                        with tab_box:
                            fig_box = plot_boxplot(rolling_returns, title=f"Box Plot ({rolling_years} anni)")
                            st.plotly_chart(fig_box, width='stretch')
                        with tab_violin:
                            fig_violin = plot_violinplot(rolling_returns, title=f"Violin Plot ({rolling_years} anni)")
                            st.plotly_chart(fig_violin, width='stretch')
                        with tab_hist:
                            # Selezione modalità
                            hist_mode = st.radio(
                                "Seleziona modalità di visualizzazione:",
                                ["Confronto Sovrapposto (Tutti gli Asset)", "Dettaglio Singolo Asset"],
                                horizontal=True,
                                key="hist_mode"
                            )
                            
                            if hist_mode == "Confronto Sovrapposto (Tutti gli Asset)":
                                fig_hist = plot_overlaid_histogram(rolling_returns, title=f"Confronto Distribuzioni ({rolling_years} anni)")
                                st.plotly_chart(fig_hist, width='stretch')
                            else:
                                selected_asset_hist = st.selectbox(
                                    "Seleziona l'asset da analizzare nel dettaglio:",
                                    options=rolling_returns.columns.tolist(),
                                    key="selected_asset_hist"
                                )
                                
                                fig_hist_single = plot_single_histogram_with_normal(
                                    rolling_returns, 
                                    selected_asset_hist, 
                                    title=f"Distribuzione {selected_asset_hist} vs Normale ({rolling_years} anni)"
                                )
                                st.plotly_chart(fig_hist_single, width='stretch')
                                
                                # Calcolo metriche di forma e normalità
                                import scipy.stats as stats
                                asset_data = rolling_returns[selected_asset_hist].dropna()
                                
                                if len(asset_data) > 3: # Shapiro-Wilk richiede almeno 3 osservazioni
                                    skewness = stats.skew(asset_data)
                                    kurtosis = stats.kurtosis(asset_data) # Eccesso di curto-si
                                    try:
                                        shapiro_stat, shapiro_p = stats.shapiro(asset_data)
                                    except Exception:
                                        shapiro_p = np.nan
                                    
                                    # Calcolo VaR Storico 95%
                                    var_95 = asset_data.quantile(0.05)
                                    
                                    m1, m2, m3, m4 = st.columns(4)
                                    m1.metric("Asimmetria (Skewness)", f"{skewness:.3f}", help="Se < 0: coda sinistra più lunga (rischio perdite estreme). Se > 0: coda destra più lunga.")
                                    m2.metric("Curto-si in Eccesso", f"{kurtosis:.3f}", help="Se > 0 (leptocurtica): code più grasse della normale (più eventi estremi).")
                                    m3.metric("Test Shapiro-Wilk (p-value)", f"{shapiro_p:.4e}" if not np.isnan(shapiro_p) else "N/D", help="Se p-value < 0.05, la distribuzione NON è considerabile statisticamente normale.")
                                    m4.metric("VaR Storico (95%)", f"{var_95:.2%}", help="Rendimento minimo atteso nel 95% dei casi. C'è solo il 5% di probabilità di fare peggio di questo valore su base annua.")
                                else:
                                    st.warning("Numero insufficiente di dati per calcolare le statistiche di dettaglio.")
                    else:
                        st.warning("Nessun dato valido per l'analisi dei Rendimenti Rolling con le opzioni selezionate o dati insufficienti.")


                # --- Contenuto per la scheda "Andamento per Finestre" ---
                with tab_windows:
                    st.subheader("Andamento per Diverse Finestre Temporali")
                    # Verifica se min_median_data contiene dati validi con chiavi 'windows' non vuote
                    if min_median_data and any(data and 'windows' in data and data['windows'] for data in min_median_data.values()):
                        # Schede per i diversi grafici dell'andamento per finestra
                        tab_min, tab_median, tab_combined, tab_detailed = st.tabs([
                            "Rendimento Minimo",
                            "Rendimento Mediano",
                            "Min & Mediano Combinato",
                            "Analisi Dettagliata"
                        ])

                        with tab_min:
                             # Filtra gli asset che hanno dati validi per le finestre
                             assets_with_data = [asset for asset, data in min_median_data.items() if data and 'windows' in data and data['windows']]
                             if assets_with_data:
                                fig_min = plot_min_vs_window(min_median_data, assets=assets_with_data, title="Rendimento Minimo vs Finestra Temporale")
                                st.plotly_chart(fig_min, width='stretch')
                             else:
                                st.warning("Nessun dato disponibile per il grafico Rendimento Minimo vs Finestra Temporale con le finestre selezionate.")

                        with tab_median:
                             assets_with_data = [asset for asset, data in min_median_data.items() if data and 'windows' in data and data['windows']]
                             if assets_with_data:
                                fig_median = plot_median_vs_window(min_median_data, assets=assets_with_data, title="Rendimento Mediano vs Finestra Temporale")
                                st.plotly_chart(fig_median, width='stretch')
                             else:
                                st.warning("Nessun dato disponibile per il grafico Rendimento Mediano vs Finestra Temporale con le finestre selezionate.")

                        with tab_combined:
                             assets_with_data = [asset for asset, data in min_median_data.items() if data and 'windows' in data and data['windows']]
                             if assets_with_data:
                                fig_combined = plot_combined_min_median(min_median_data, assets=assets_with_data, title="Rendimenti Minimo e Mediano vs Finestra Temporale - Tutti gli Asset")
                                st.plotly_chart(fig_combined, width='stretch')
                             else:
                                st.warning("Nessun dato disponibile per il grafico Rendimenti Minimo e Mediano Combinato.")


                        with tab_detailed:
                            st.markdown("#### Seleziona 1 o 2 asset per l'analisi dettagliata con Box Plot")
                            detailed_selected_assets = st.multiselect(
                                "Seleziona asset(s)",
                                actual_analysis_indices, # Usa gli indici effettivamente analizzati
                                default=actual_analysis_indices[:min(2, len(actual_analysis_indices))],
                                max_selections=2,
                                key="detailed_asset_selection" # Aggiungi una chiave univoca
                            )
                            # Verifica che gli asset selezionati per il dettaglio abbiano dati validi per le finestre
                            if detailed_selected_assets and all(asset in min_median_data and min_median_data[asset] and 'windows' in min_median_data[asset] and min_median_data[asset]['windows'] for asset in detailed_selected_assets):
                                # Filtra min_median_data per includere solo gli asset selezionati
                                filtered_min_median_data = {asset: min_median_data[asset] for asset in detailed_selected_assets if asset in min_median_data}
                                fig_detailed = plot_detailed_window_analysis(filtered_min_median_data, detailed_selected_assets, title="Analisi Dettagliata per Finestra Temporale con Box Plot")
                                st.plotly_chart(fig_detailed, width='stretch')
                            elif detailed_selected_assets:
                                st.warning("Impossibile creare l'analisi dettagliata. I dati per l'asset/gli asset selezionati non sono disponibili per tutte le finestre.")
                            else:
                                st.info("Seleziona uno o due asset per visualizzare l'analisi dettagliata per finestra.")
                    else:
                        st.warning("Impossibile calcolare l'andamento per diverse finestre temporali con i dati disponibili.")


                with tab_risk_metrics:
                    if not risk_metrics_df.empty:
                        # TRASPOSIZIONE: Metriche sulle COLONNE, Asset sulle RIGHE
                        df_to_display = risk_metrics_df.T

                        # Prepara la configurazione delle colonne con i Tooltip
                        column_configs = {}
                        for metric_name, info in METRIC_EXPLANATIONS.items():
                            if metric_name in df_to_display.columns:
                                # Crea il testo del tooltip combinando Cos'è e Cosa Indica
                                tooltip_text = f"**Cos'è:** {info.get('Cos\'è', '')}\n\n**Indica:** {info.get('Cosa indica', '')}"
                                
                                # Formattazione automatica basata sul nome della metrica
                                is_percent = "(%)" in metric_name or "Ulcer Index" in metric_name or "Penalized Risk" in metric_name
                                format_str = "%.2f%%" if is_percent else "%.2f"
                                
                                column_configs[metric_name] = st.column_config.NumberColumn(
                                    metric_name,
                                    help=tooltip_text,
                                    format=format_str
                                )

                        # Definisci le metriche dove "Basso è Meglio" (per il gradiente)
                        lower_is_better_metrics = [
                            "Annualized Volatility (%)",
                            "Ulcer Index",
                            "Pitfall Indicator",
                            "Penalized Risk (%)",
                            "Downside Risk (%)",
                        ]
                        
                        # Filtra le metriche presenti
                        lower_cols = [m for m in lower_is_better_metrics if m in df_to_display.columns]
                        higher_cols = [m for m in df_to_display.columns if m not in lower_cols]

                        # Applica lo styling
                        styled_df = df_to_display.style
                        
                        # Gradiente: asse 0 (confronta gli asset per ogni metrica/colonna)
                        if lower_cols:
                            styled_df = styled_df.background_gradient(cmap='RdYlGn_r', axis=0, subset=lower_cols)

                        if higher_cols:
                            styled_df = styled_df.background_gradient(cmap='RdYlGn', axis=0, subset=higher_cols)

                        # Visualizza la tabella
                        st.dataframe(
                            styled_df,
                            column_config=column_configs,
                            width='stretch'
                        )
                        
                        st.info("💡 Passa il mouse sopra il nome di una metrica nell'intestazione per visualizzarne la spiegazione.")

                    else:
                         st.warning("Nessuna metrica di rischio calcolata per gli indici selezionati. Assicurati che i dati caricati contengano abbastanza punti e che gli indici siano stati selezionati correttamente.")

                    st.markdown("---")
                    st.subheader("Metriche di Rischio per l'intero Periodo Disponibile")
                    
                    st.markdown("""
                    Benvenuto nella sezione delle **Metriche di Rischio**. Per aiutarti a navigare tra questi indicatori, li abbiamo organizzati seguendo un percorso logico-matematico che va dalle basi fondamentali alla sintesi finale:

                    1. **Fondamenta (Metriche di Base)**  
                    Queste metriche sono calcolate direttamente dai prezzi o dai rendimenti periodici, senza dipendere da altri indicatori di rischio.
                    * **Total Return (%)**: Il guadagno assoluto. Indica il risultato finale assoluto dell'investimento.
                    * **Annualized Return (%)**: La base per tutti i rapporti di efficienza. Indica la velocità media di crescita del capitale.
                    * **Annualized Volatility (%)**: La misura di rischio standard (usata poi per Sharpe e Pitfall). Indica incertezza e instabilità; alta volatilità significa forti sbalzi di prezzo.
                    * **Max Drawdown (%)**: Il calo massimo storico (fondamento per il Calmar). Indica il rischio massimo storico; quanto avresti perso nel momento peggiore.
                    * **Downside Risk (%)**: Volatilità focalizzata solo sulle perdite (fondamento per il Sortino). Indica il vero rischio di perdere denaro, ignorando la volatilità "positiva" (rialzi).
                    * **VaR_Returns(95%) (%)**: La soglia di perdita "normale" nel periodo. Indica il rischio "normale" di mercato su base periodica (es. mensile).

                    2. **Efficienza Classica e Code (Primo Livello di Derivazione)**  
                    Indicatori che mettono in relazione il rendimento con una singola metrica di base o approfondiscono la statistica delle code.
                    * **Sharpe Ratio**: Relazione Rendimento / Volatilità. Indica l'efficienza classica: quanto "paga" assumersi dei rischi standard.
                    * **Sortino Ratio**: Relazione Rendimento / Downside Risk. Indica l'efficienza nel generare profitti minimizzando solo le perdite (non le oscillazioni).
                    * **Calmar Ratio**: Relazione Rendimento / Max Drawdown. Indica la resilienza: capacità di recuperare e guadagnare dopo il peggior disastro.
                    * **CVaR_Returns(95%) (%)**: Approfondimento del VaR (media delle perdite oltre la soglia). Indica quanto ci si aspetta di perdere quando le cose vanno molto male.

                    3. **Analisi Avanzata dello Stress (Il sistema Ulcer e DaR)**  
                    Metriche che analizzano l'intera distribuzione dei cali (drawdown) anziché solo il punto peggiore.
                    * **Ulcer Index**: Calcolato sull'intero storico dei drawdown (fondamento per UPI e Penalized Risk). Indica la "quantità di dolore" sofferta; penalizza periodi lunghi e profondi di perdita.
                    * **Ulcer Performance Index**: Relazione Rendimento / Ulcer Index. Indica l'efficienza nel generare profitti minimizzando la sofferenza dei cali.
                    * **DaR(95%) (%)**: Soglia di rischio sui drawdown (fondamento per CDaR). Indica quanto potresti perdere in una situazione di mercato negativa ma non estrema.
                    * **CDaR(95%) (%)**: Media dei drawdown oltre la soglia DaR (fondamento per il Pitfall). Indica il danno atteso durante un crollo di mercato grave ("Cigno Nero").

                    4. **Alternative Portfolio Theory (Sintesi Finale)**  
                    Il culmine del percorso, dove le metriche precedenti vengono combinate per definire il rischio "totale" (durata, profondità e sorpresa).
                    * **Pitfall Indicator**: Mette in relazione la gravità dei crolli (CDaR) con la Volatilità standard. Se alto, indica un rischio nascosto: l'asset sembra tranquillo ma ha crolli improvvisi e violenti.
                    * **Penalized Risk (%)**: Sintesi matematica tra lo stress continuo (Ulcer Index) e gli eventi estremi (Pitfall Indicator). Fornisce una visione completa del rischio: quanto a lungo perdi e quanto violentemente.
                    * **Serenity Ratio**: Il risultato finale. Indica la generazione di rendimento con il minimo rischio totale (stress + eventi estremi).
                    """)

                # --- Contenuto per la scheda "Regressione Fattoriale" ---
                with tab_factors:
                    st.subheader("Analisi di Regressione Fattoriale (Fama-French)")
                    st.markdown("""
                    Questo modulo consente di analizzare l'esposizione fattoriale di un asset utilizzando il **Modello a 5 Fattori di Fama-French con l'aggiunta del Momentum (6 fattori totali)**.
                    """)
                    
                    # Layout a due colonne per configurazione e parametri
                    col_cfg1, col_cfg2 = st.columns(2)
                    
                    with col_cfg1:
                        reg_data_source = st.radio(
                            "Sorgente Dati Asset",
                            options=["Indice Caricato", "Yahoo Finance Ticker"],
                            key="reg_data_source"
                        )
                        
                        if reg_data_source == "Indice Caricato":
                            selected_reg_asset = st.selectbox(
                                "Seleziona l'Indice da analizzare",
                                options=actual_analysis_indices,
                                key="selected_reg_asset"
                            )
                            # Rilevamento della frequenza dell'indice selezionato
                            try:
                                asset_series = analysis_data[selected_reg_asset].dropna()
                                if len(asset_series) > 2:
                                    diffs_price = asset_series.index.to_series().diff().dropna()
                                    median_days_price = diffs_price.median() / pd.Timedelta(days=1)
                                    detected_freq = "daily" if median_days_price < 10 else "monthly"
                                else:
                                    detected_freq = "monthly"
                            except Exception:
                                detected_freq = "monthly"
                                
                            st.info(f"Frequenza rilevata automaticamente per '{selected_reg_asset}': **{detected_freq}**")
                        else:
                            reg_ticker = st.text_input(
                                "Inserisci Ticker Yahoo Finance (es. SPY, VWCE.MI, AAPL)",
                                value="SPY",
                                key="reg_ticker"
                            )
                            yahoo_input_currency = st.selectbox(
                                "Valuta Ticker Yahoo Finance",
                                options=["EUR", "USD"],
                                index=1,  # Default a USD
                                key="yahoo_input_currency",
                                help="Valuta in cui sono denominati i prezzi storici del ticker su Yahoo Finance."
                            )
                            detected_freq = st.selectbox(
                                "Frequenza dei dati",
                                options=["monthly", "daily"],
                                index=0,
                                key="reg_freq_yfinance"
                            )
                        
                            
                    with col_cfg2:
                        ff_region = st.selectbox(
                            "Regione dei Fattori Kenneth French",
                            options=["USA", "Developed Markets", "Developed ex US", "Europe", "Emerging Markets", "Global"],
                            index=0,
                            key="ff_region"
                        )
                        
                        # Gestione frequenza Emerging Markets
                        if ff_region == "Emerging Markets":
                            if detected_freq == "daily":
                                st.warning("I fattori per 'Emerging Markets' sono disponibili solo a frequenza mensile. L'asset verrà risampillato su base mensile.")
                            detected_freq_to_use = "monthly"
                        else:
                            detected_freq_to_use = detected_freq
                            
                        # Configurazione finestra di rolling beta in base alla frequenza
                        if detected_freq_to_use == "monthly":
                            min_w, max_w, val_w = 12, 120, 36
                        else:
                            min_w, max_w, val_w = 60, 500, 252
                            
                        rolling_window = st.slider(
                            "Finestra Rolling Beta (osservazioni)",
                            min_value=min_w,
                            max_value=max_w,
                            value=val_w,
                            key="reg_rolling_window"
                        )
                        
                        cov_type = st.selectbox(
                            "Tipo Errore Standard OLS",
                            options=["HAC", "nonrobust", "HC3"],
                            index=0,
                            key="reg_cov_type"
                        )
                        
                        # Soglia alert R² fissata al 95% nel codice
                    
                    # Bottone di avvio
                    run_reg_btn = st.button("Avvia Regressione Fattoriale 🚀", type="primary", key="run_reg_btn")
                    
                    # Esecuzione e rendering dei risultati
                    # Salviamo i risultati nello stato di Streamlit per mantenere la visualizzazione attiva
                    if run_reg_btn or st.session_state.get("reg_completed", False):
                        
                        # Se cambia la configurazione, forziamo il ricalcolo se premuto il pulsante
                        if run_reg_btn:
                            st.session_state["reg_completed"] = False
                            st.session_state["reg_ack_proceed_check"] = False
                            
                        try:
                            reg_display_currency = st.session_state.get("reg_display_currency", "EUR")
                            
                            # Prepara i dati dei rendimenti
                            if reg_data_source == "Indice Caricato":
                                asset_raw_prices = pd.DataFrame(analysis_data[selected_reg_asset].dropna())
                                asset_raw_prices.columns = ['price']
                                asset_name = selected_reg_asset
                                asset_curr = reg_display_currency
                            else:
                                if not reg_ticker.strip():
                                    st.error("Inserisci un ticker Yahoo Finance valido.")
                                    st.stop()
                                    
                                with st.spinner("Download dei dati dell'asset da Yahoo Finance..."):
                                    from factor_regression.asset_parser import load_asset_from_yahoo
                                    # Determina data inizio e fine in base all'indice dei dati caricati (se disponibili) per allineamento temporale
                                    start_date_str = analysis_data.index.min().strftime("%Y-%m-%d")
                                    end_date_str = analysis_data.index.max().strftime("%Y-%m-%d")
                                    asset_raw_prices = load_asset_from_yahoo(reg_ticker.strip(), start_date_str, end_date_str, detected_freq_to_use)
                                    asset_name = reg_ticker.strip()
                                    asset_curr = st.session_state.get("yahoo_input_currency", "USD")
                                    
                            # 1. Carica i cambi EUR/USD (BCE + Fed proxy)
                            from factor_regression.exchange_rate import get_exchange_rates
                            rates_df, is_online, last_rate_date = get_exchange_rates()
                            
                            # Avviso sullo stato dei cambi
                            if not is_online:
                                st.warning(f"⚠️ **Modalità Offline Tassi di Cambio**: Impossibile collegarsi alla BCE per scaricare i cambi aggiornati. I tassi storici sono disponibili fino al **{last_rate_date}**. Le serie temporali dell'asset verranno troncate a questa data.")
                            else:
                                st.info(f"ℹ️ **Tassi di Cambio Aggiornati**: Dati BCE allineati ad oggi (**{last_rate_date}**).")
                                
                            # Troncamento se le date dell'asset vanno oltre quelle dei cambi
                            last_rate_dt = pd.to_datetime(last_rate_date)
                            if asset_raw_prices.index.max() > last_rate_dt:
                                asset_raw_prices = asset_raw_prices[asset_raw_prices.index <= last_rate_dt]
                                st.info(f"⚠️ I dati dell'asset sono stati troncati al **{last_rate_date}** per allineamento con i tassi di cambio disponibili.")
                                
                            # 2. Calcolo dei rendimenti nella valuta nativa dell'asset
                            from factor_regression.asset_parser import compute_returns
                            asset_returns_native = compute_returns(asset_raw_prices, detected_freq_to_use)
                            
                            # 3. Conversione dei rendimenti in USD se la valuta in input è EUR
                            rates_aligned = rates_df.reindex(asset_returns_native.index, method='ffill')
                            
                            if asset_curr == "EUR":
                                # Converte i rendimenti da EUR a USD direttamente per evitare il bug di shift della data dei prezzi:
                                # 1 + R_USD = (1 + R_EUR) * (EUR_USD_curr / EUR_USD_prev)
                                rate_ratio = rates_aligned['EUR_USD'] / rates_aligned['EUR_USD'].shift(1).ffill()
                                rate_ratio.iloc[0] = 1.0  # mantieni primo rendimento
                                
                                asset_returns_usd = asset_returns_native.copy()
                                asset_returns_usd['asset_return'] = (1 + asset_returns_native['asset_return']) * rate_ratio - 1
                            else:
                                asset_returns_usd = asset_returns_native.copy()
                                
                            # Download dei fattori Fama-French (in USD)
                            with st.spinner("Download dei fattori Fama-French da Kenneth French..."):
                                from factor_regression.factors_fetcher import fetch_factors
                                min_date = asset_returns_usd.index.min().strftime("%Y-%m-%d")
                                max_date = asset_returns_usd.index.max().strftime("%Y-%m-%d")
                                factors_df = fetch_factors(ff_region, detected_freq_to_use, min_date, max_date)
                                
                            # Allineamento dei Dati
                            from factor_regression.regression_engine import (
                                prepare_regression_dataset,
                                run_static_regression,
                                run_rolling_regression,
                                calculate_factor_contributions
                            )
                            reg_df = prepare_regression_dataset(asset_returns_usd, factors_df)
                            
                            # Validazioni
                            from factor_regression.config import MIN_OBSERVATIONS
                            min_obs_required = MIN_OBSERVATIONS[detected_freq_to_use]
                            n_obs = len(reg_df)
                            
                            if n_obs < min_obs_required:
                                st.error(f"Errore: Osservazioni allineate insufficienti ({n_obs}). Sono richieste almeno {min_obs_required} osservazioni per la frequenza '{detected_freq_to_use}'.")
                                st.stop()
                                
                            if rolling_window >= n_obs:
                                st.error(f"Errore: La finestra di rolling selezionata ({rolling_window}) è superiore o uguale al numero totale di osservazioni allineate ({n_obs}).")
                                st.stop()
                                
                            # Esecuzione regressioni (in USD)
                            static_results = run_static_regression(reg_df, cov_type=cov_type)
                            rolling_df = run_rolling_regression(reg_df, rolling_window)
                            
                            # 4. Conversione per la visualizzazione nella valuta scelta
                            rates_reg = rates_df.reindex(reg_df.index, method='ffill')
                            r2_threshold = 0.95  # Soglia alert R² fissata al 95%
                            
                            if reg_display_currency == "EUR":
                                # Converte i rendimenti e fattori da USD a EUR
                                display_df = reg_df.copy()
                                rate_ratio_usd_to_eur = rates_reg['EUR_USD'].shift(1).ffill() / rates_reg['EUR_USD']
                                rate_ratio_usd_to_eur.iloc[0] = 1.0
                                
                                display_df['asset_return'] = (1 + reg_df['asset_return']) * rate_ratio_usd_to_eur - 1
                                display_df['RF'] = (1 + reg_df['RF']) * rate_ratio_usd_to_eur - 1
                                display_df['asset_excess_return'] = display_df['asset_return'] - display_df['RF']
                                
                                factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
                                for f in factors_list:
                                    display_df[f] = (1 + reg_df[f]) * rate_ratio_usd_to_eur - 1
                                    
                                fitted_excess_usd = static_results['results_object'].fittedvalues
                                fitted_total_usd = fitted_excess_usd + reg_df['RF']
                                fitted_total_eur = (1 + fitted_total_usd) * rate_ratio_usd_to_eur - 1
                                fitted_excess_display = fitted_total_eur - display_df['RF']
                            else:
                                # Rimane in USD
                                display_df = reg_df.copy()
                                fitted_excess_display = static_results['results_object'].fittedvalues
                                
                            # Ricalcolo dei contributi nella valuta di visualizzazione
                            contrib_df = calculate_factor_contributions(static_results, display_df, annualize=True, frequency=detected_freq_to_use)
                            
                            # Memorizza nello stato di sessione per mantenere l'interfaccia persistente al cambio tab
                            st.session_state["reg_completed"] = True
                            st.session_state["reg_data"] = {
                                "reg_df": reg_df,
                                "display_df": display_df,
                                "static_results": static_results,
                                "rolling_df": rolling_df,
                                "contrib_df": contrib_df,
                                "fitted_excess_display": fitted_excess_display,
                                "asset_name": asset_name,
                                "ff_region": ff_region,
                                "detected_freq_to_use": detected_freq_to_use,
                                "rolling_window": rolling_window,
                                "r2_threshold": r2_threshold,
                                "reg_display_currency": reg_display_currency
                            }
                            
                        except Exception as e:
                            st.error(f"Errore durante l'elaborazione dell'analisi fattoriale: {e}")
                            st.exception(e)
                            st.session_state["reg_completed"] = False
                            
                    # Rendering dei risultati salvati nello stato di sessione
                    if st.session_state.get("reg_completed", False):
                        reg_data = st.session_state["reg_data"]
                        reg_df = reg_data["reg_df"]
                        display_df = reg_data["display_df"]
                        static_results = reg_data["static_results"]
                        rolling_df = reg_data["rolling_df"]
                        contrib_df = reg_data["contrib_df"]
                        fitted_excess_display = reg_data["fitted_excess_display"]
                        asset_name = reg_data["asset_name"]
                        ff_region = reg_data["ff_region"]
                        detected_freq_to_use = reg_data["detected_freq_to_use"]
                        rolling_window = reg_data["rolling_window"]
                        r2_threshold = reg_data["r2_threshold"]
                        reg_display_currency = reg_data["reg_display_currency"]
                        
                        # Sotto-schede dell'analisi fattoriale
                        st.markdown("---")
                        st.subheader(f"Risultati Analisi Fattoriale: **{asset_name}**")
                        
                        # Verifica bloccante dell'R-quadrato rispetto alla soglia
                        if static_results['rsquared'] < r2_threshold:
                            st.error(f"🚨 **ATTENZIONE: R-quadrato molto basso ({static_results['rsquared']*100:.2f}%) sotto la soglia di allerta ({r2_threshold*100:.0f}%)!**")
                            st.markdown("""
                            Il modello a 6 fattori Fama-French spiega meno della soglia impostata di variabilità dei rendimenti per questo asset. 
                            I coefficienti, le scomposizioni e le metriche di questa regressione **potrebbero non essere affidabili**.
                            """)
                            
                            proceed = st.checkbox(
                                "Ho capito la scarsa affidabilità del modello e desidero comunque visualizzare i risultati e i grafici dell'analisi.",
                                value=st.session_state.get("reg_ack_proceed_check", False),
                                key="reg_ack_proceed_check"
                            )
                            
                            if not proceed:
                                st.info("👉 Per sbloccare la visualizzazione dei risultati, spunta la casella di conferma qui sopra.")
                                st.stop()
                        
                        subtab_overview, subtab_static, subtab_rolling, subtab_dist, subtab_contrib, subtab_raw = st.tabs([
                            "📋 Overview",
                            "📊 Regressione Statica",
                            "📈 Beta Rolling",
                            "📉 Distribuzione Fattori",
                            "💰 Scomposizione Rendimenti",
                            "💾 Dati Allineati"
                        ])
                        
                        from factor_regression.plots import (
                            plot_cumulative_returns,
                            plot_factor_boxplot,
                            plot_rolling_betas,
                            plot_rolling_betas_boxplot,
                            plot_factor_contributions,
                            plot_factor_correlation
                        )
                        
                        # 1. Overview
                        with subtab_overview:
                            col_o1, col_o2, col_o3, col_o4 = st.columns(4)
                            col_o1.metric("Osservazioni", int(static_results['nobs']))
                            col_o2.metric("R-quadrato (R²)", f"{static_results['rsquared']:.4f}")
                            col_o3.metric("R² Adjusted", f"{static_results['rsquared_adj']:.4f}")
                            col_o4.metric("Durbin-Watson", f"{static_results['durbin_watson']:.2f}")
                            
                            st.plotly_chart(plot_cumulative_returns(display_df, static_results, fitted_excess_display), use_container_width=True)
                            
                            # Warning non bloccanti
                            if static_results['rsquared'] < r2_threshold:
                                st.warning(f"⚠️ **R-quadrato sotto la soglia ({r2_threshold*100:.0f}%)**: Il modello a 6 fattori Fama-French spiega solo il {static_results['rsquared']*100:.2f}% della variabilità dei rendimenti di questo asset.")
                                
                            pvals = static_results['pvalues']
                            factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
                            sig_factors = [f for f in factors_list if pvals[f] < 0.05]
                            if not sig_factors:
                                st.info("ℹ️ **Nessun Fattore Significativo**: Nessuno dei 6 fattori presenta un p-value inferiore a 0.05 (significatività 5%).")
                            else:
                                st.success(f"✔️ **Fattori Significativi (p-value < 0.05)**: {', '.join(sig_factors)}")
                                
                        # 2. Regressione Statica
                        with subtab_static:
                            st.subheader("Tabella dei Coefficienti (OLS)")
                            params = static_results['params']
                            bse = static_results['bse']
                            tvals = static_results['tvalues']
                            pvals = static_results['pvalues']
                            
                            labels_map = {
                                'const': 'Alpha (const)',
                                'Mkt-RF': 'Market Beta (Mkt-RF)',
                                'SMB': 'Size Beta (SMB)',
                                'HML': 'Value Beta (HML)',
                                'RMW': 'Profitability Beta (RMW)',
                                'CMA': 'Investment Beta (CMA)',
                                'Mom': 'Momentum Beta (Mom)'
                            }
                            
                            table_rows = []
                            for idx in params.index:
                                table_rows.append({
                                    "Componente": labels_map.get(idx, idx),
                                    "Coefficiente (Beta)": params[idx],
                                    "Std Error": bse[idx],
                                    "t-Stat": tvals[idx],
                                    "p-Value": pvals[idx],
                                    "Significativo (5%)": pvals[idx] < 0.05
                                })
                                
                            coef_df = pd.DataFrame(table_rows)
                            styled_coef_df = coef_df.style.format({
                                "Coefficiente (Beta)": "{:.6f}",
                                "Std Error": "{:.6f}",
                                "t-Stat": "{:.4f}",
                                "p-Value": "{:.4f}"
                            })
                            
                            st.dataframe(styled_coef_df, use_container_width=True, hide_index=True)
                            
                            st.subheader("Metriche Diagnostiche del Modello")
                            col_diag1, col_diag2 = st.columns(2)
                            with col_diag1:
                                st.write(f"**Tipo Errore Standard**: {static_results['covariance_type']}")
                                if static_results['covariance_type'] == "HAC":
                                    st.write(f"**Max Lags (HAC)**: {static_results['maxlags']}")
                                st.write(f"**F-Statistic**: {static_results['fvalue']:.4f} (p-value: {static_results['f_pvalue']:.4f})")
                            with col_diag2:
                                st.write(f"**R-quadrato (R²)**: {static_results['rsquared']:.6f}")
                                st.write(f"**R-quadrato Modificato**: {static_results['rsquared_adj']:.6f}")
                                st.write(f"**Durbin-Watson Stat**: {static_results['durbin_watson']:.4f}")
                                st.caption("Nota: DW vicino a 2.0 indica assenza di autocorrelazione nei residui.")
                                
                        # 3. Beta Rolling
                        with subtab_rolling:
                            st.subheader(f"Evoluzione Temporale delle Esposizioni (Finestra: {rolling_window} oss.)")
                            st.plotly_chart(plot_rolling_betas(rolling_df), use_container_width=True)
                            
                            st.subheader("Distribuzione e Stabilità dei Beta Rolling")
                            st.plotly_chart(plot_rolling_betas_boxplot(rolling_df), use_container_width=True)
                            
                            st.dataframe(rolling_df.style.format("{:.4f}"), use_container_width=True)
                            
                        # 4. Distribuzione dei Fattori
                        with subtab_dist:
                            st.subheader("Analisi Statistica Descrittiva dei Fattori")
                            st.plotly_chart(plot_factor_boxplot(display_df), use_container_width=True)
                            st.plotly_chart(plot_factor_correlation(display_df), use_container_width=True)
                            
                            factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
                            desc_df = (display_df[factors_list] * 100).describe().T[['mean', 'std', 'min', '50%', 'max', 'count']]
                            desc_df.columns = ['Media (%)', 'Dev. Std (%)', 'Min (%)', 'Mediana (%)', 'Max (%)', 'Osservazioni']
                            st.write(f"**Statistiche Descrittive Storiche dei Fattori (espresse in % e convertite in {reg_display_currency})**")
                            st.dataframe(desc_df.style.format("{:.4f}"), use_container_width=True)
                            
                        # 5. Scomposizione dei Rendimenti
                        with subtab_contrib:
                            st.subheader("Scomposizione della Performance Stima Media Annualizzata")
                            st.plotly_chart(plot_factor_contributions(contrib_df), use_container_width=True)
                            
                            styled_contrib = contrib_df.copy()
                            styled_contrib.columns = ["Fattore", "Beta Estimato", "Media Storica Fattore (Annuo %)", "Contributo Annuo Stimato (%)"]
                            styled_contrib["Media Storica Fattore (Annuo %)"] = styled_contrib["Media Storica Fattore (Annuo %)"] * 100
                            styled_contrib["Contributo Annuo Stimato (%)"] = styled_contrib["Contributo Annuo Stimato (%)"] * 100
                            
                            st.dataframe(styled_contrib.style.format({
                                "Beta Stimato": "{:.4f}",
                                "Media Storica Fattore (Annuo %)": "{:.2f}%",
                                "Contributo Annuo Stimato (%)": "{:.2f}%"
                            }, na_rep="-"), use_container_width=True, hide_index=True)
                            
                            st.info("ℹ️ **Nota Metodologica**: I contributi sono calcolati come $Beta_j \\times Media(Fattore_j)$. I fattori ed alpha sono annualizzati moltiplicando la media semplice per 12 (mensile) o per 252 (giornaliero).")
                            
                        # 6. Dati Allineati
                        with subtab_raw:
                            st.subheader(f"Dati Allineati (Valuta: {reg_display_currency})")
                            st.dataframe(display_df.style.format("{:.6f}"), use_container_width=True)
                            
                            csv_data = display_df.to_csv().encode('utf-8')
                            st.download_button(
                                label="Scarica Dati Allineati in CSV 📥",
                                data=csv_data,
                                file_name=f"dati_allineati_{asset_name}_{reg_display_currency}.csv",
                                mime="text/csv",
                                key="reg_download_csv"
                            )


            else:
                st.warning("I dati selezionati per l'analisi sono vuoti dopo il filtraggio. Verifica la selezione degli indici.")

        # --- (condizioni else esistenti per nessun indice selezionato o nessun dato combinato) ---
        else:
             if combined_data is not None and not combined_data.empty:
                 st.warning("Seleziona gli indici numerici da analizzare dalla barra laterale.")
             else:
                 st.info("Carica uno o più file CSV o Excel per iniziare l'analisi.")


if __name__ == "__main__":
    main()
