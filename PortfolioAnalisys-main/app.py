import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from data_loader import load_data # Assumi che esista data_loader.py con la funzione load_data
from rolling_calculations import (calculate_rolling_returns, calculate_min_median_by_window, calculate_risk_metrics, create_portfolio) # Assumi esista rolling_calculations.py
from plots import (plot_rolling_returns, plot_boxplot, plot_violinplot, plot_min_vs_window, plot_median_vs_window, plot_combined_min_median, plot_detailed_window_analysis) # Assumi esista plots.py

from risk_metrics import PortfolioRiskMetrics # Importa la classe delle metriche di rischio
import math # Needed for sqrt(12)

# Dizionario contenente le spiegazioni dettagliate per ciascuna metrica.
METRIC_EXPLANATIONS = {
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
     "Total Return (%)": {
        "Cos'è": "Guadagno complessivo cumulato dall'inizio alla fine.",
        "Come viene calcolato": "(Valore Finale - Iniziale) / Iniziale.",
        "Cosa indica": "Il risultato finale assoluto dell'investimento.",
        "Meglio": "**Alto**.",
    },
    "Downside Risk (%)": {
        "Cos'è": "Volatilità considerata solo quando i prezzi scendono.",
        "Come viene calcolato": "Deviazione standard dei soli rendimenti negativi.",
        "Cosa indica": "Il vero rischio di perdere denaro, ignorando la volatilità 'positiva' (rialzi).",
        "Meglio": "**Basso**.",
    },
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
     "VaR_Returns(95%) (%)": { 
        "Cos'è": "Value at Risk. Massima perdita periodica attesa nel 95% dei casi.",
        "Come viene calcolato": "Quantile 5% dei rendimenti periodici.",
        "Cosa indica": "Il rischio 'normale' di mercato su base periodica (es. mensile).",
        "Meglio": "**Basso (vicino a 0)**.",
    },
    "CVaR_Returns(95%) (%)": { 
        "Cos'è": "Conditional VaR. Perdita media periodica negli scenari peggiori.",
        "Come viene calcolato": "Media delle perdite che superano il VaR.",
        "Cosa indica": "Quanto ci si aspetta di perdere quando le cose vanno molto male.",
        "Meglio": "**Basso (vicino a 0)**.",
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
    analysis_mode = st.radio("Modalità di analisi", ["Confronto Completo", "Indice Singolo", "Confronto Indici", "Portafoglio"], index=0)
    rolling_years = st.slider("Periodo Rolling (anni)", min_value=1, max_value=20, value=3, step=1)

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
    else:
        # Logica per caricare il file di default se nessun file è stato caricato
        default_file = "chart_default.csv"
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
                    st.write("Anteprima dei dati combinati prima della pulizia:")
                    st.dataframe(combined_data.head())

                # --- INIZIO: Codice per la pulizia dei dati (rimozione date incomplete) ---
                if uploaded_files:
                    st.info("Pulizia dati: rimozione date senza valori per tutti gli indici...")
                
                # Identifica le colonne che sono numeri - queste sono le probabili colonne degli indici
                # Usa .copy() per evitare SettingWithCopyWarning
                combined_data_cleaned = combined_data.copy()
                numeric_cols_before_cleaning = combined_data_cleaned.select_dtypes(include=np.number).columns.tolist()

                if numeric_cols_before_cleaning:
                    # Rimuovi righe (date) dove almeno una delle colonne numeriche ha un valore NaN
                    # Questo mantiene solo le date per cui TUTTI gli indici numerici hanno un valore.
                    initial_rows = combined_data_cleaned.shape[0]
                    combined_data_cleaned.dropna(subset=numeric_cols_before_cleaning, inplace=True)
                    rows_after_cleaning = combined_data_cleaned.shape[0]

                    if combined_data_cleaned.empty:
                        st.error("Dopo la pulizia dei dati, non ci sono date rimanenti con valori completi per tutti gli indici.")
                        combined_data = None # Imposta combined_data originale a None
                    else:
                        dropped_rows_count = initial_rows - rows_after_cleaning
                        if dropped_rows_count > 0 and uploaded_files:
                            st.warning(f"Rimosse {dropped_rows_count} righe (date) a causa di valori mancanti per alcuni indici.")
                        
                        if uploaded_files:
                            st.success(f"Pulizia dati completata. Rimaste {combined_data_cleaned.shape[0]} date con dati completi per tutti gli indici numerici.")
                            st.write("Anteprima dei dati puliti:")
                            st.dataframe(combined_data_cleaned.head())
                        
                        combined_data = combined_data_cleaned # Aggiorna combined_data con il DataFrame pulito
                else:
                    if uploaded_files:
                        st.warning("Nessuna colonna numerica trovata per la pulizia dei dati.")
                    # Se non ci sono colonne numeriche, prosegui con i dati non puliti numericamente,
                    # ma combined_data è già stato assegnato e ordinato sopra.
                    if uploaded_files:
                        st.info("Proseguendo con i dati combinati senza pulizia specifica per colonne numeriche.")


                # --- FINE: Codice per la pulizia dei dati ---

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
                            portfolio_data = create_portfolio(combined_data[selected_portfolio_assets], weights)
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
            analysis_data = combined_data[[idx for idx in selected_indices if idx in combined_data.columns]]
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

                    # Calcola i rendimenti rolling e i dati per l'analisi per finestre
                    # (solo se ci sono abbastanza dati per almeno un asset per il periodo rolling specificato)
                    st.info("Preparazione dati per Rendimenti Rolling e Analisi per Finestre...")
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


                    min_median_data = {} # Initialize as empty
                    try:
                        min_median_data = calculate_min_median_by_window(analysis_data)
                    except Exception as e:
                        st.error(f"Errore durante il calcolo min/median per finestre: {e}")
                        min_median_data = {} # Set to empty on error

                # --- Aggiunge le schede di primo livello ---
                tab_rolling, tab_windows, tab_risk_metrics = st.tabs([
                    "Rendimenti Rolling & Distribuzioni",
                    "Andamento per Finestre",
                    "Metriche di Rischio" # Nuova scheda
                ])

                # --- Contenuto per la scheda "Rendimenti Rolling & Distribuzioni" ---
                with tab_rolling:
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
                        st.plotly_chart(fig_returns, use_container_width=True)

                        # Box Plot e Violin Plot
                        st.subheader("Distribuzione dei Rendimenti Rolling")
                        tab_box, tab_violin = st.tabs(["Box Plot", "Violin Plot"])
                        with tab_box:
                            fig_box = plot_boxplot(rolling_returns, title=f"Box Plot ({rolling_years} anni)")
                            st.plotly_chart(fig_box, use_container_width=True)
                        with tab_violin:
                            fig_violin = plot_violinplot(rolling_returns, title=f"Violin Plot ({rolling_years} anni)")
                            st.plotly_chart(fig_violin, use_container_width=True)
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
                                st.plotly_chart(fig_min, use_container_width=True)
                             else:
                                st.warning("Nessun dato disponibile per il grafico Rendimento Minimo vs Finestra Temporale con le finestre selezionate.")

                        with tab_median:
                             assets_with_data = [asset for asset, data in min_median_data.items() if data and 'windows' in data and data['windows']]
                             if assets_with_data:
                                fig_median = plot_median_vs_window(min_median_data, assets=assets_with_data, title="Rendimento Mediano vs Finestra Temporale")
                                st.plotly_chart(fig_median, use_container_width=True)
                             else:
                                st.warning("Nessun dato disponibile per il grafico Rendimento Mediano vs Finestra Temporale con le finestre selezionate.")

                        with tab_combined:
                             assets_with_data = [asset for asset, data in min_median_data.items() if data and 'windows' in data and data['windows']]
                             if assets_with_data:
                                fig_combined = plot_combined_min_median(min_median_data, assets=assets_with_data, title="Rendimenti Minimo e Mediano vs Finestra Temporale - Tutti gli Asset")
                                st.plotly_chart(fig_combined, use_container_width=True)
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
                                st.plotly_chart(fig_detailed, use_container_width=True)
                            elif detailed_selected_assets:
                                st.warning("Impossibile creare l'analisi dettagliata. I dati per l'asset/gli asset selezionati non sono disponibili per tutte le finestre.")
                            else:
                                st.info("Seleziona uno o due asset per visualizzare l'analisi dettagliata per finestra.")
                    else:
                        st.warning("Impossibile calcolare l'andamento per diverse finestre temporali con i dati disponibili.")


                # --- Contenuto per la scheda "Metriche di Rischio" (NUOVA) ---
                with tab_risk_metrics:
                    st.subheader("Metriche di Rischio per l'intero Periodo Disponibile")
                    if not risk_metrics_df.empty:
                        # Visualizza la tabella delle metriche
                        # Definisci le metriche dove "Basso è Meglio" (Verde basso, Rosso alto)
                        # SOLO metriche POSITIVE dove un valore assoluto più basso è preferibile.
                        lower_is_better_metrics = [
                            "Annualized Volatility (%)",
                            "Ulcer Index",
                            "Pitfall Indicator",
                            "Penalized Risk (%)",
                            "Downside Risk (%)",
                            # Le metriche negative (Drawdown, VaR, CVaR, DaR, CDaR) maticamente:
                            # -5 (Meglio) > -50 (Peggio).
                            # Quindi usano la logica standard "Alto è Meglio" (RdYlGn), non questa lista.
                        ]
                        
                        # Filtra le metriche presenti nel DataFrame
                        lower_subset = [m for m in lower_is_better_metrics if m in risk_metrics_df.index]
                        higher_subset = [m for m in risk_metrics_df.index if m not in lower_subset]

                        # Crea lo Styler object
                        styled_df = risk_metrics_df.style.format({
                            col: "{:.2f}%" if "(%)" in col or "Ulcer Index" in str(col) or "Penalized Risk" in str(col) else "{:.2f}"
                            for col in risk_metrics_df.columns
                        })

                        # Applica gradiente per "Lower is Better" (Verde=Basso, Rosso=Alto -> RdYlGn_r)
                        if lower_subset:
                            styled_df = styled_df.background_gradient(cmap='RdYlGn_r', axis=1, subset=pd.IndexSlice[lower_subset, :])

                        # Applica gradiente per "Higher is Better" (Rosso=Basso, Verde=Alto -> RdYlGn)
                        if higher_subset:
                            styled_df = styled_df.background_gradient(cmap='RdYlGn', axis=1, subset=pd.IndexSlice[higher_subset, :])

                        st.dataframe(styled_df)
                        
                        st.info("Metriche calcolate sull'intero periodo.")

                        # Aggiungi la sezione per le spiegazioni dettagliate con expander
                        st.markdown("---") # Linea separatrice
                        st.subheader("Spiegazione Dettagliata delle Metriche")

                        # Itera sulle metriche definite nel dizionario delle spiegazioni e usa st.expander
                        for metric_name, explanation_details in METRIC_EXPLANATIONS.items():
                            # Usa st.expander per creare il menu a tendina per ogni metrica
                            with st.expander(f"**{metric_name}**"): # Il titolo dell'expander sarà il nome della metrica
                                for section_title, content in explanation_details.items():
                                    # Utilizza markdown per formattare le sezioni (Cos'è, Come, ecc.) all'interno dell'expander
                                    st.markdown(f"**{section_title}:** {content}")

                    else:
                         st.warning("Nessuna metrica di rischio calcolata per gli indici selezionati. Assicurati che i dati caricati contengano abbastanza punti e che gli indici siano stati selezionati correttamente.")


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
