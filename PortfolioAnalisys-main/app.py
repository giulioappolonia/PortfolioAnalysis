import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from data_loader import load_data # Assumi che esista data_loader.py con la funzione load_data
from rolling_calculations import (calculate_rolling_returns, calculate_min_median_by_window, calculate_risk_metrics, create_portfolio) # Assumi esista rolling_calculations.py
from plots import (plot_rolling_returns, plot_boxplot, plot_violinplot, plot_min_vs_window, plot_median_vs_window, plot_combined_min_median, plot_detailed_window_analysis) # Assumi esista plots.py
from MSCI_to_Curvo_converter import convert_history_index # Assumi esista MSCI_to_Curvo_converter.py
from risk_metrics import PortfolioRiskMetrics # Importa la classe delle metriche di rischio
import math # Needed for sqrt(12)

# Dizionario contenente le spiegazioni dettagliate per ciascuna metrica.
METRIC_EXPLANATIONS = {
    "Annualized Return (%)": {
        "Cos'è": "Misura il guadagno o la perdita media per anno, supponendo che gli utili vengano reinvestiti. Consente il confronto diretto tra investimenti detenuti per periodi di tempo diversi.",
        "Come viene calcolato": "Si basa sul rendimento totale del periodo di investimento, proiettato su base annuale utilizzando un tasso di crescita composto. La formula comune è `(1 + Rendimento Totale)^(1 / Anni Totali) - 1`.",
        "Cosa indica per l'investitore": "Rappresenta il tasso di crescita effettivo medio annuo dell'investimento. È una misura chiave della redditività.",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica una maggiore redditività annuale.",
    },
    "Annualized Volatility (%)": {
        "Cos'è": "Misura la dispersione (deviazione standard) dei rendimenti dell'investimento attorno alla sua media, annualizzata. È una misura comune del rischio totale (sia al rialzo che al ribasso).",
        "Come viene calcolato": "La deviazione standard dei rendimenti periodici (es. mensili) moltiplicata per la radice quadrata del numero di periodi in un anno (fattore di annualizzazione). Formula tipica: `Deviazione Standard Periodica * sqrt(Fattore Annualizzazione)`.",
        "Cosa indica per l'investitore": "Indica quanto il valore dell'investimento tende a fluttuare. Una maggiore volatilità significa che il prezzo può cambiare drasticamente in un breve periodo, in entrambe le direzioni.",
        "Meglio alto o basso?": "**Meglio basso**. Un valore più basso indica un'investimento più stabile con minori oscillazioni di prezzo, il che è generalmente preferito dagli investitori avversi al rischio.",
    },
    "Max Drawdown (%)": {
        "Cos'è": "Rappresenta la massima perdita percentuale dal picco storico (punto più alto) al successivo punto più basso (punto di minimo) prima che un nuovo picco venga raggiunto. Misura il rischio di \"caduta\" di un investimento.",
        "Come viene calcolato": "Si identifica la serie storica dei massimi cumulativi del NAV. Il drawdown in qualsiasi momento è la perdita percentuale rispetto al massimo cumulativo raggiunto fino a quel momento. Il Max Drawdown è il valore più negativo di questa serie di drawdown.",
        "Cosa indica per l'investitore": "Indica la peggiore perdita potenziale che un investitore avrebbe subito se avesse acquistato al picco e venduto al punto più basso successivo. È una misura critica del rischio di coda e della capacità di recupero.",
        "Meglio alto o basso?": "**Meglio basso (meno negativo)**. Un valore più vicino allo zero (o meno negativo) indica che l'investimento ha subito perdite massime meno severe.",
    },
    "Ulcer Index": {
        "Cos'è": "Una misura del rischio di drawdown che considera sia la *magnitudine* che la *durata* dei drawdown. Penalizza i drawdown più ampi e più lunghi. È una metrica chiave nel contesto di teorie alternative che si concentrano sul rischio di drawdown.",
        "Come viene calcolato": "È la radice quadrata della media dei drawdown percentuali al quadrato. `UI = sqrt(mean(Drawdown_t^2))`, dove Drawdown_t è il drawdown percentuale (in decimale) al tempo t rispetto al picco precedente. Il valore viene spesso rappresentato come percentuale.",
        "Cosa indica per l'investitore": "Fornisce una metrica più completa del rischio di drawdown rispetto al solo Max Drawdown, considerando il \"dolore\" causato da drawdown prolungati. Un valore elevato suggerisce che l'investimento ha trascorso periodi significativi in uno stato di drawdown profondo.",
        "Meglio alto o basso?": "**Meglio basso**. Un valore più basso indica drawdown meno gravi o di durata inferiore nel tempo.",
    },
    "Ulcer Performance Index": {
        "Cos'è": "Un ratio di performance aggiustato per il rischio che utilizza l'Ulcer Index come misura del rischio. Proposto come un modo per valutare l'efficienza del rendimento rispetto al rischio di drawdown considerando la durata e l'ampiezza delle perdite.",
        "Come viene calcolato": "Rendimento Annualizzato diviso per l'Ulcer Index. `UPI = Rendimento Annualizzato / Ulcer Index`.",
        "Cosa indica per l'investitore": "Indica quanti \"punti\" di rendimento annualizzato sono stati generati per ogni \"punto\" di Ulcer Index. È una misura di efficienza del rendimento rispetto al rischio di drawdown (considerando durata e ampiezza).",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica un miglior rendimento rispetto al rischio di Ulcer Index.",
    },
    "DaR(95%) (%)": { # Adatta la chiave al formato usato nel DataFrame
        "Cos'è": "Stima la perdita massima di drawdown che non dovrebbe essere superata con una certa probabilità (es. 95%) in un dato periodo, basandosi sulla distribuzione storica dei drawdown. È il VaR applicato alla distribuzione dei drawdown, rilevante per valutare il rischio di coda.",
        "Come viene calcolato": "È il quantile α-esimo della serie storica dei drawdown (generalmente con α=0.05 per il 95%). `DaR(α) = Quantile(α) dei Drawdown`.",
        "Cosa indica per l'investitore": "Fornisce una stima della potenziale perdita di picco che l'investitore potrebbe affrontare nel 5% dei casi peggiori, basandosi sulla storia.",
        "Meglio alto o basso?": "**Meglio basso (meno negativo)**. Un valore più vicino allo zero indica un rischio di drawdown di coda inferiore.",
    },
    "CDaR(95%) (%)": { # Adatta la chiave al formato usato nel DataFrame
        "Cos'è": "Rappresenta la perdita media dei drawdown che si verificano quando il drawdown è peggiore del DaR. È il CVaR (o Expected Shortfall) applicato alla distribuzione dei drawdown, una misura del rischio di coda \"oltre il VaR\".",
        "Come viene calcolato": "È la media di tutti i drawdown storici che sono inferiori o uguali al DaR(α). `CDaR(α) = Media(Drawdown | Drawdown <= DaR(α))`.",
        "Cosa indica per l'investitore": "Fornisce una stima della perdita media che l'investitore subirebbe nel 5% dei casi peggiori *una volta che il DaR viene superato*. È una misura del rischio di coda più informativa del solo DaR.",
        "Meglio alto o basso?": "**Meglio basso (meno negativo)**. Un valore più vicino allo zero indica perdite medie meno severe nella coda peggiore della distribuzione dei drawdown.",
    },
    "Pitfall Indicator": {
        "Cos'è": "Una misura di rischio proposta nel contesto di teorie alternative, che confronta la gravità del CDaR (rischio di coda sui drawdown) con la volatilità totale (rischio di oscillazione generale). Aiuta a capire quanto il rischio di coda contribuisca al rischio complessivo.",
        "Come viene calcolato": "Valore Assoluto del CDaR (sui drawdown decimali) diviso per la Volatilità Annualizzata (decimale). `Pitfall Indicator = |CDaR| / Volatilità Annualizzata`.",
        "Cosa indica per l'investitore": "Aiuta a capire quanto del rischio totale (misurato dalla volatilità) sia concentrato nella coda peggiore dei drawdown (misurata dal CDaR). Un valore più alto suggerisce che il rischio di drawdown di coda è proporzionalmente maggiore rispetto alla volatilità generale.",
        "Meglio alto o basso?": "**Meglio basso**. Un valore più basso indica che il rischio di drawdown di coda è meno significativo rispetto alla volatilità complessiva.",
    },
    "Penalized Risk (%)": {
        "Cos'è": "Una misura composita di rischio proposta in teorie alternative, che combina l'Ulcer Index (durata e ampiezza dei drawdown) con il Pitfall Indicator (rischio di coda sui drawdown rispetto alla volatilità). Cerca di fornire una visione olistica del rischio legato ai drawdown e alle perdite estreme.",
        "Come viene calcolato": "Prodotto dell'Ulcer Index (decimale) e del Pitfall Indicator (decimale). `Penalized Risk = Ulcer Index * Pitfall Indicator`. Il risultato viene spesso rappresentato come percentuale.",
        "Cosa indica per l'investitore": "Questa metrica cerca di catturare il rischio da diverse angolazioni legate ai drawdown e al rischio di coda. Un valore elevato suggerisce che l'investimento presenta sia drawdown significativi sia un rischio di coda proporzionalmente alto.",
        "Meglio alto o basso?": "**Meglio basso**. Un valore più basso indica un profilo di rischio di drawdown più favorevole secondo questa metrica composita.",
    },
    "Serenity Ratio": {
        "Cos'è": "Il ratio di performance principale proposto nel contesto di teorie alternative, che confronta il Rendimento Annualizzato con il Penalized Risk. Simile allo Sharpe Ratio, ma utilizzando una misura di rischio alternativa incentrata sui drawdown e il rischio di coda.",
        "Come viene calcolato": "Rendimento Annualizzato (decimale) diviso per il Penalized Risk (decimale). `Serenity Ratio = Rendimento Annualizzato / Penalized Risk`.",
        "Cosa indica per l'investitore": "Misura il rendimento annualizzato generato per unità di \"rischio penalizzato\". È un indicatore di efficienza che preferisce gli investimenti con alti rendimenti e basso Penalized Risk, offrendo una prospettiva diversa sulla performance aggiustata per il rischio rispetto alle metriche tradizionali basate sulla volatilità totale.",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica una migliore performance aggiustata per il rischio secondo la definizione di rischio del PDF.",
    },
     "Total Return (%)": {
        "Cos'è": "Misura il cambiamento percentuale complessivo del valore dell'investimento dall'inizio alla fine del periodo considerato.",
        "Come viene calcolato": "`(Valore Finale - Valore Iniziale) / Valore Iniziale * 100`.",
        "Cosa indica per l'investitore": "Rappresenta il guadagno o la perdita complessiva sull'investimento per l'intero orizzonte temporale. Utile per vedere la performance assoluta, ma non consente confronti diretti tra periodi di durata diversa.",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica un maggiore guadagno complessivo.",
    },
    "Downside Risk (%)": {
        "Cos'è": "Misura la volatilità dei rendimenti negativi rispetto a un valore obiettivo (tipicamente 0% o il tasso privo di rischio), annualizzata. Ignora la volatilità dei rendimenti positivi, concentrandosi solo sul rischio di perdite.",
        "Come viene calcolato": "La deviazione standard dei rendimenti periodici inferiori al tasso obiettivo (es. 0% o tasso privo di rischio), rispetto a tale tasso obiettivo, annualizzata.",
        "Cosa indica per l'investitore": "Si concentra solo sul rischio di perdite. Una maggiore Downside Risk indica oscillazioni negative più ampie o più frequenti al di sotto del tasso obiettivo.",
        "Meglio alto o basso?": "**Meglio basso**. Un valore più basso indica un rischio di perdite inferiore.",
    },
    "Sharpe Ratio": {
        "Cos'è": "Misura il rendimento in eccesso (rendimento al di sopra del tasso privo di rischio) generato per unità di rischio totale (volatilità annualizzata). È uno dei ratio di performance aggiustata per il rischio più utilizzati, basato sull'assunzione che la volatilità rappresenti adeguatamente tutto il rischio.",
        "Come viene calcolato": "`(Rendimento Annualizzato - Tasso Privo di Rischio Annualizzato) / Volatilità Annualizzata`.",
        "Cosa indica per l'investitore": "Indica quanto rendimento aggiuntivo si ottiene per assumere una maggiore volatilità. Un valore più alto suggerisce un portafoglio più efficiente nel generare rendimento dato il rischio totale misurato dalla volatilità.",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica una migliore performance aggiustata per la volatilità.",
    },
    "Sortino Ratio": {
        "Cos'è": "Simile allo Sharpe Ratio, ma utilizza il Downside Risk (volatilità solo dei rendimenti negativi) al posto della volatilità totale. Si concentra sul rendimento in eccesso per unità di rischio di rendimenti negativi.",
        "Come viene calcolato": "`(Rendimento Annualizzato - Tasso Privo di Rischio Annualizzato) / Downside Risk Annualizzato`.",
        "Cosa indica per l'investitore": "Indica quanti rendimenti in eccesso si ottengono per ogni unità di rischio di perdite al di sotto di un obiettivo. È preferito da molti investitori rispetto allo Sharpe Ratio perché penalizza solo la volatilità negativa, allineandosi meglio con la percezione comune di rischio come \"perdita\".",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica una migliore performance aggiustata per il rischio di rendimenti negativi.",
    },
    "Calmar Ratio": {
        "Cos'è": "Misura il rendimento annualizzato rispetto alla massima perdita subita (Max Drawdown assoluto). È una metrica semplice ma efficace per confrontare il rendimento con il peggiore calo storico.",
        "Come viene calcolato": "Rendimento Annualizzato diviso per il valore assoluto del Max Drawdown. `Calmar Ratio = Rendimento Annualizzato / |Max Drawdown|`.",
        "Cosa indica per l'investitore": "È una metrica di performance aggiustata per il rischio che si concentra specificamente sul rischio di drawdown estremo. Un valore più alto indica che l'investimento ha generato un buon rendimento annualizzato rispetto alla sua peggiore caduta storica.",
        "Meglio alto o basso?": "**Meglio alto**. Un valore più elevato indica un miglior rendimento rispetto al massimo drawdown.",
    },
     "VaR_Returns(95%) (%)": { # Adatta la chiave al formato usato nel DataFrame
        "Cos'è": "Stima la perdita massima sui *rendimenti* periodici che non si prevede venga superata con una certa probabilità (es. 95%) in un dato periodo, basandosi sulla distribuzione storica dei rendimenti. È una misura del rischio di coda per i rendimenti periodici.",
        "Come viene calcolato": "È il quantile α-esimo della serie storica dei rendimenti periodici (generalmente con α=0.05 per il 95%). `VaR(α) = Quantile(α) dei Rendimenti`.",
        "Cosa indica per l'investitore": "Fornisce una stima della potenziale perdita di rendimento per un singolo periodo (es. un mese) nel 5% dei casi peggiori.",
        "Meglio alto o basso?": "**Meglio basso (meno negativo)**. Un valore più vicino allo zero indica un rischio di perdita periodica di coda inferiore sui rendimenti.",
    },
    "CVaR_Returns(95%) (%)": { # Adatta la chiave al formato usato nel DataFrame
        "Cos'è": "Rappresenta la perdita media sui *rendimenti* periodici che si verificano quando il rendimento è peggiore del VaR sui rendimenti. È il CVaR (o Expected Shortfall) applicato alla distribuzione dei rendimenti, una misura più completa del rischio di coda sui rendimenti periodici rispetto al solo VaR.",
        "Come viene calcolato": "È la media di tutti i rendimenti periodici storici che sono inferiori o uguali al VaR sui rendimenti (VaR_Returns). `CVaR(α) = Media(Rendimento | Rendimento <= VaR_Returns(α))`.",
        "Cosa indica per l'investitore": "Fornisce una stima della perdita media che l'investitore subirebbe in un singolo periodo nel 5% dei casi peggiori *una volta che il VaR sui rendimenti viene superato*. È il rischio medio in caso di eventi di rendimento estremi negativi.",
        "Meglio alto o basso?": "**Meglio basso (meno negativo)**. Un valore più vicino allo zero indica perdite medie meno severe nella coda peggiore della distribuzione dei rendimenti.",
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
        st.info(f"Caricati {len(uploaded_files)} file. Elaborazione...")
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                try:
                    temp_input_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_input_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    data = None
                    file_identifier = os.path.splitext(uploaded_file.name)[0]

                    if file_extension == ".csv":
                        st.write(f"Processando file CSV: {uploaded_file.name}")
                        data = load_data(temp_input_path)
                    elif file_extension in [".xls", ".xlsx"]:
                        st.write(f"Processando file Excel: {uploaded_file.name}. Conversione in CSV...")
                        temp_csv_path = os.path.join(temp_dir, f"converted_{file_identifier}.csv")
                        try:
                            convert_history_index(temp_input_path, temp_csv_path)
                            st.write(f"Conversione di {uploaded_file.name} completata. Caricamento dati...")
                            data = load_data(temp_csv_path)
                        except Exception as e:
                            st.error(f"Errore durante la conversione o il caricamento del file Excel {uploaded_file.name}: {e}")
                            data = None
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


        # Dopo aver processato tutti i file, combina i DataFrames
        if loaded_dfs:
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
                    st.success("Dati combinati con successo.")
                    st.write("Anteprima dei dati combinati prima della pulizia:")
                    st.dataframe(combined_data.head())

                    # --- INIZIO: Codice per la pulizia dei dati (rimozione date incomplete) ---
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
                             if dropped_rows_count > 0:
                                 st.warning(f"Rimosse {dropped_rows_count} righe (date) a causa di valori mancanti per alcuni indici.")
                             st.success(f"Pulizia dati completata. Rimaste {combined_data_cleaned.shape[0]} date con dati completi per tutti gli indici numerici.")
                             st.write("Anteprima dei dati puliti:")
                             st.dataframe(combined_data_cleaned.head())
                             combined_data = combined_data_cleaned # Aggiorna combined_data con il DataFrame pulito
                    else:
                        st.warning("Nessuna colonna numerica trovata per la pulizia dei dati.")
                        # Se non ci sono colonne numeriche, prosegui con i dati non puliti numericamente,
                        # ma combined_data è già stato assegnato e ordinato sopra.
                        st.info("Proseguendo con i dati combinati senza pulizia specifica per colonne numeriche.")


                    # --- FINE: Codice per la pulizia dei dati ---

            except Exception as e:
                st.error(f"Errore durante la combinazione o la pulizia dei dati: {e}")
                combined_data = None
        else:
            st.warning("Nessun file valido è stato caricato o processato con successo.")
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
                        st.dataframe(risk_metrics_df.style.format({
                            col: "{:.2f}%" if "(%)" in col or col == "Ulcer Index" or col == "Penalized Risk (%)" else "{:.2f}"
                            for col in risk_metrics_df.columns
                        }))
                        st.info(f"Metriche calcolate sull'intero periodo disponibile per ciascun asset. Fattore di annualizzazione per volatilità/rischi: {annualization_factor:.2f} (assumendo dati mensili). Tasso privo di rischio periodico: {period_risk_free_rate:.4f}.") # Mostra il tasso periodico usato

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
