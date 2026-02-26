import pandas as pd
import numpy as np
import math
from scipy.stats import norm

class PortfolioRiskMetrics:
    """
    Una classe per calcolare diverse metriche di rischio e rendimento per una serie storica di prezzi.
    Include metriche classiche e metriche basate sui drawdown, con implementazioni di PI,
    Penalized Risk e Serenity Ratio conformi alle formule del documento PDF "An Alternative Portfolio Theory",
    e aggiunge l'Ulcer Performance Index.
    """

    def __init__(self, nav_series, returns_series, annualization_factor=None, risk_free_rate=0.0):
        """
        Inizializza la classe con la serie storica del Net Asset Value (NAV) o Prezzi,
        la serie dei rendimenti periodici e un fattore di annualizzazione opzionale.
        Args:
            nav_series (pd.Series): Serie pandas con l'indice temporale e i valori NAV/Prezzo.
            returns_series (pd.Series): Serie pandas con l'indice temporale e i rendimenti periodici.
                                         Deve avere lo stesso indice di nav_series (o quasi, dopo pct_change).
            annualization_factor (float, optional): Il fattore per annualizzare i risultati (es. sqrt(12) per dati mensili).
                                                     Se None, viene utilizzato un fattore di 1.0 (nessuna annualizzazione).
            risk_free_rate (float, optional): Tasso privo di rischio periodico da usare per Sharpe/Sortino.
                                              Default a 0.0. Assicurati che sia nella stessa periodicità dei returns_series.
        """
        # Rimuove i NaN all'inizio se presenti (possono esserci nel NAV o nel primo rendimento)
        # Assicura che le serie siano allineate sull'indice
        self.nav_series = nav_series.dropna()
        self.returns_series = returns_series.dropna()

        # Allinea le serie sull'indice comune dopo la rimozione dei NaN
        common_index = self.nav_series.index.intersection(self.returns_series.index)
        self.nav_series = self.nav_series.loc[common_index]
        self.returns_series = self.returns_series.loc[common_index]

        if self.returns_series.empty:
            raise ValueError("Le serie di rendimenti non possono essere vuote dopo la pulizia.")

        if self.nav_series.empty:
             raise ValueError("La serie NAV non può essere vuota dopo la pulizia.")


        self.annualization_factor = annualization_factor if annualization_factor is not None else 1.0
        self.risk_free_rate = risk_free_rate # Questo è il tasso periodico

    def total_return(self):
        """Calcola il rendimento totale nel periodo."""
        if self.nav_series.empty:
            return np.nan
        return (self.nav_series.iloc[-1] / self.nav_series.iloc[0]) - 1

    def annualized_return(self):
        """Calcola il rendimento annualizzato."""
        if self.nav_series.empty or len(self.nav_series) < 2:
            return np.nan

        total_ret = self.total_return()
        if total_ret is np.nan:
             return np.nan

        # Calcola la durata totale in anni basata sulla differenza tra le date
        total_years = (self.nav_series.index[-1] - self.nav_series.index[0]).days / 365.25

        if total_years <= 0:
             # Gestisce il caso di meno di un giorno o dati non sequenziali per data
             # Potrebbe essere necessario un approccio diverso per dati a frequenza molto alta
             # Per dati mensili/giornalieri standard questo dovrebbe andare bene
             # Se la durata è < 1 anno ma > 0, l'annualizzazione è (1+tot_ret)^(1/anni_totali)
             # Se tot_ret è -1 (perdita del 100%), non si può annualizzare
             return np.nan if total_ret == -1.0 else float('-inf') if total_ret < -1 else (1 + total_ret)**(1 / total_years) - 1


        # Formula di annualizzazione: (1 + Rendimento Totale)^(1 / Anni Totali) - 1
        if (1 + total_ret) < 0: # Gestisce casi di rendimento totale -100% o peggio
             return np.nan if total_ret == -1.0 else float('-inf') # Non può annualizzare un rendimento < -100%


        return (1 + total_ret)**(1 / total_years) - 1

    def annualized_volatility(self):
        """Calcola la volatilità annualizzata."""
        if self.returns_series.empty:
            return np.nan
        # Deviazione standard periodica * fattore
        return self.returns_series.std() * self.annualization_factor

    def drawdowns(self):
        """Calcola la serie storica dei drawdown percentuali (in decimale, range [-1, 0])."""
        if self.nav_series.empty:
            return pd.Series(dtype=float)

        # Calcola i picchi cumulativi
        cumulative_max = self.nav_series.cummax()

        # Calcola i drawdown percentuali in decimale
        drawdowns = (self.nav_series - cumulative_max) / cumulative_max
        return drawdowns

    def max_drawdown(self):
        """Calcola il massimo drawdown."""
        drawdowns_series = self.drawdowns()
        if drawdowns_series.empty:
            return np.nan
        # Il massimo drawdown è il minimo (il valore negativo più grande) nei drawdown
        return drawdowns_series.min()

    def ulcer_index(self):
        """
        Calcola l'Ulcer Index.
        Restituisce un valore decimale, sqrt(mean(d_t^2)) dove d_t è il drawdown decimale.
        """
        drawdowns_series = self.drawdowns()
        if drawdowns_series.empty:
             return np.nan

        # Calcola i drawdown quadrati (in decimale)
        drawdown_squared = drawdowns_series**2

        # L'Ulcer Index è la radice quadrata della media dei drawdown quadrati
        return np.sqrt(drawdown_squared.mean()) # Risultato è in decimale (range [0, 1])

    # --- Nuova metrica aggiunta qui ---
    def ulcer_performance_index(self):
        """
        Calcola l'Ulcer Performance Index (UPI): Rendimento Annualizzato / Ulcer Index.
        Restituisce un valore decimale (ratio).
        """
        ann_ret = self.annualized_return() # Rendimento Annualizzato in decimale
        ulcer_val = self.ulcer_index() # Ulcer Index in decimale

        if ann_ret is np.nan or ulcer_val is np.nan:
            return np.nan

        if ulcer_val == 0: # Evita divisione per zero
            return np.inf if ann_ret > 0 else (-np.inf if ann_ret < 0 else np.nan) # Comportamento tipico per ratio

        return ann_ret / ulcer_val
    # ---------------------------------

    def value_at_risk_returns(self, alpha=0.05):
        """
        Calcola il Periodical Value at Risk (VaR) basato sulla distribuzione dei RENDIMENTI.
        Restituisce il VaR come perdita (valore negativo decimale).
        """
        if self.returns_series.empty:
            return np.nan
        # Quantile non-parametrico sui rendimenti
        return self.returns_series.quantile(alpha)

    def conditional_value_at_risk_returns(self, alpha=0.05):
        """
        Calcola il Periodical Conditional Value at Risk (CVaR) o Expected Shortfall (ES) basato sulla distribuzione dei RENDIMENTI.
        Restituisce il CVaR come perdita (valore negativo decimale).
        """
        if self.returns_series.empty:
            return np.nan
        # Calcola il VaR sui rendimenti
        var = self.value_at_risk_returns(alpha)
        if var is np.nan:
            return np.nan

        # Considera solo i rendimenti inferiori o uguali al VaR sui rendimenti
        shortfall_returns = self.returns_series[self.returns_series <= var]

        if shortfall_returns.empty:
             # Se nessun rendimento è <= VaR (es. VaR è 0 e nessun rendimento è negativo)
             # In questo caso, la coda non esiste sotto il VaR, il CVaR è 0.
             # Oppure, se VaR è positivo (guadagno) e nessun rendimento è <= di esso, CVaR è 0.
             return 0.0 if var >= 0 else var # Se var è negativo, e non ci sono rendimenti <= var (impossibile se var < min(returns)) o returns vuoti

        # Il CVaR è la media di questi rendimenti di "coda"
        return shortfall_returns.mean()

    def drawdown_at_risk(self, alpha=0.05):
        """
        Calcola il Drawdown at Risk (DaR).
        È il quantile alpha-esimo della distribuzione dei drawdown storici (decimali).
        Restituisce il DaR come valore negativo o zero (decimale).
        """
        drawdowns_series = self.drawdowns()
        if drawdowns_series.empty:
            return np.nan

        # Il DaR(alpha) è il quantile alpha della serie di drawdown (negativi o zero).
        return drawdowns_series.quantile(alpha)

    def conditional_drawdown_at_risk(self, alpha=0.05):
        """
        Calcola il Conditional Drawdown at Risk (CDaR).
        È la media dei drawdown storici (decimali) che sono inferiori o uguali al DaR.
        Restituisce il CDaR come valore negativo o zero (decimale).
        """
        drawdowns_series = self.drawdowns()
        if drawdowns_series.empty:
            return np.nan

        # Calcola il DaR
        dar = self.drawdown_at_risk(alpha)
        if dar is np.nan:
            return np.nan

        # Considera solo i drawdown inferiori o uguali al DaR
        shortfall_drawdowns = drawdowns_series[drawdowns_series <= dar]

        if shortfall_drawdowns.empty:
             # Se nessun drawdown è <= DaR (es. DaR è 0 e nessun drawdown è negativo)
             return 0.0

        # Il CDaR è la media di questi drawdown di "coda"
        return shortfall_drawdowns.mean()


    def downside_risk(self):
        """Calcola il Downside Risk (volatilità dei rendimenti negativi rispetto a MAR), annualizzato."""
        if self.returns_series.empty:
            return np.nan

        # Usiamo rendimenti < risk_free_rate periodico come nel codice originale
        # Se risk_free_rate è 0, consideriamo i rendimenti negativi
        target_returns = self.returns_series[self.returns_series < self.risk_free_rate]

        if target_returns.empty:
            # Se non ci sono rendimenti al di sotto del target, il rischio downside è 0
            return 0.0

        # Calcola la deviazione standard di questi rendimenti rispetto al target
        # Formula: sqrt( mean (max(0, target - r_t)^2) )
        # Questo è equivalente a sqrt( mean ((r_t - target)^2) ) solo per r_t < target
        downside_deviation = np.sqrt(np.mean((target_returns - self.risk_free_rate)**2))

        # Annualizza la downside deviation
        return downside_deviation * self.annualization_factor

    def sharpe_ratio(self):
        """Calcola lo Sharpe Ratio annualizzato."""
        ann_ret = self.annualized_return()
        ann_vol = self.annualized_volatility()

        # Il tasso privo di rischio annualizzato per lo Sharpe/Sortino ratio è r_f_periodico * annualization_factor^2
        annualized_risk_free_rate = self.risk_free_rate * self.annualization_factor

        # Controllo per evitare divisioni per zero o NaN
        if ann_vol is np.nan or ann_ret is np.nan or annualized_risk_free_rate is np.nan:
             return np.nan

        # Gestione divisione per zero per vol=0
        if ann_vol == 0:
             # Se volatilità è zero: +inf se rendimento > r_f, -inf se rendimento < r_f, NaN se rendimento == r_f
             if ann_ret > annualized_risk_free_rate:
                 return np.inf
             elif ann_ret < annualized_risk_free_rate:
                 return -np.inf
             else:
                 return np.nan # o 0.0 a seconda della convenzione, ma NaN più sicuro

        return (ann_ret - annualized_risk_free_rate) / ann_vol

    def sortino_ratio(self):
        """Calcola il Sortino Ratio annualizzato."""
        ann_ret = self.annualized_return()
        downside_vol = self.downside_risk()

        # Il tasso privo di rischio annualizzato per lo Sharpe/Sortino ratio è r_f_periodico * annualization_factor^2
        annualized_risk_free_rate = self.risk_free_rate * self.annualization_factor

        # Controllo per evitare divisioni per zero o NaN
        if downside_vol is np.nan or ann_ret is np.nan or annualized_risk_free_rate is np.nan:
             return np.nan

        # Gestione divisione per zero per downside_vol=0
        if downside_vol == 0:
             # Se downside volatilità è zero: +inf se rendimento > r_f, -inf se rendimento < r_f, NaN se rendimento == r_f
             if ann_ret > annualized_risk_free_rate:
                 return np.inf
             elif ann_ret < annualized_risk_free_rate:
                 return -np.inf
             else:
                 return np.nan # o 0.0

        return (ann_ret - annualized_risk_free_rate) / downside_vol

    def calmar_ratio(self):
        """Calcola il Calmar Ratio (Rendimento Annualizzato / Max Drawdown Assoluto)."""
        ann_ret = self.annualized_return()
        max_dd = self.max_drawdown()

        if max_dd is np.nan or ann_ret is np.nan:
            return np.nan

        absolute_max_dd = abs(max_dd)

        # Il Calmar ratio è definito per drawdown > 0 (ossia max_dd < 0).
        # Se max_dd è 0, non c'è drawdown, e il ratio non è applicabile nel modo standard.
        if absolute_max_dd == 0:
             return np.inf if ann_ret > 0 else (-np.inf if ann_ret < 0 else np.nan)

        return ann_ret / absolute_max_dd

    def pitfall_indicator(self, alpha_cdar=0.05):
        """
        Calcola il Pitfall Indicator secondo la formula del PDF: |CDaR| / Volatilità Annualizzata.
        Args:
            alpha_cdar (float): Alfa per il calcolo del CDaR (es. 0.05 per 95%).
        Restituisce un valore decimale (ratio).
        """
        cdar_val = self.conditional_drawdown_at_risk(alpha=alpha_cdar) # CDaR in decimale negativo o zero
        ann_vol = self.annualized_volatility() # Volatilità in decimale positivo o zero

        if cdar_val is np.nan or ann_vol is np.nan:
             return np.nan

        # CDaR è negativo o zero. Il valore assoluto |CDaR| è positivo o zero.
        absolute_cdar = abs(cdar_val)

        if ann_vol == 0: # Evita divisione per zero
             # Se volatilità zero, PI è infinito se CDaR non zero, altrimenti NaN
             return np.inf if absolute_cdar > 0 else np.nan

        # Calcola il rapporto tra il valore assoluto di CDaR decimale e la volatilità decimale
        return absolute_cdar / ann_vol


    def penalized_risk(self, alpha_tail=0.05):
        """
        Calcola il Penalized Risk secondo la formula del PDF: Ulcer x Pitfall.
        Args:
            alpha_tail (float): Alfa utilizzato per il calcolo del Pitfall (tramite CDaR).
        Restituisce un valore decimale.
        """
        ulcer_val = self.ulcer_index() # Ulcer Index in decimale
        pitfall_val = self.pitfall_indicator(alpha_cdar=alpha_tail) # Pitfall Indicator in decimale

        if ulcer_val is np.nan or pitfall_val is np.nan:
            return np.nan

        # Calcola il prodotto dei valori decimali
        return ulcer_val * pitfall_val

    def serenity_ratio(self, alpha_tail=0.05):
        """
        Calcola il Serenity Ratio secondo la formula del PDF: Annualized Return / Penalized Risk.
        Args:
            alpha_tail (float): Alfa utilizzato per il calcolo del Penalized Risk.
        Restituisce un valore decimale (ratio).
        """
        ann_ret = self.annualized_return() # Rendimento Annualizzato in decimale
        penalized_risk_val = self.penalized_risk(alpha_tail=alpha_tail) # Penalized Risk in decimale

        if ann_ret is np.nan or penalized_risk_val is np.nan:
            return np.nan

        if penalized_risk_val == 0: # Evita divisione per zero
             return np.inf if ann_ret > 0 else (-np.inf if ann_ret < 0 else np.nan)

        # Calcola il rapporto tra rendimento annualizzato decimale e penalized risk decimale
        return ann_ret / penalized_risk_val


    def get_all_metrics(self, alpha_tail=0.05):
        """
        Calcola e restituisce tutte le metriche disponibili, inclusi DaR e CDaR,
        con PI, Penalized Risk, Serenity Ratio conformi al PDF, e Ulcer Performance Index.
        Args:
            alpha_tail (float): Alfa per VaR/CVaR sui rendimenti e DaR/CDaR sui drawdown (es. 0.05 per 95%).
                                Tutte queste metriche di coda influenzeranno PI, Penalized Risk, SR.

        Returns:
            dict: Un dizionario con i nomi delle metriche come chiavi e i valori calcolati.
                  Le metriche percentuali sono restituite come valori percentuali (es. 10.5 per 10.5%),
                  mentre i ratio (Sharpe, Sortino, Calmar, Pitfall, Serenity, UPI) sono decimali.
                  VaR, CVaR, DaR e CDaR sono restituiti come perdite (valori negativi percentuali).
        """

        # Calcola le metriche (molte sono ora decimali internamente)
        total_ret = self.total_return() # Decimal
        ann_ret = self.annualized_return() # Decimal
        ann_vol = self.annualized_volatility() # Decimal
        max_dd = self.max_drawdown() # Decimal negative
        ulcer = self.ulcer_index() # Decimal

        # Calcola la nuova metrica Ulcer Performance Index
        upi = self.ulcer_performance_index() # Ratio decimale

        # VaR/CVaR sui RENDIMENTI (periodici, decimali negativi)
        # Nota: l'alpha qui è per il quantile di coda inferiore (es. 0.05 per il 5% peggiore)
        var_ret = self.value_at_risk_returns(alpha=alpha_tail) # Decimal negative
        cvar_ret = self.conditional_value_at_risk_returns(alpha=alpha_tail) # Decimal negative

        # DaR/CDaR sui DRAWDOWN (periodici, decimali negativi)
        # Nota: l'alpha qui è per il quantile di coda inferiore (es. 0.05 per il 5% peggiore dei drawdown)
        dar_dd = self.drawdown_at_risk(alpha=alpha_tail) # Decimal negative
        cdar_dd = self.conditional_drawdown_at_risk(alpha=alpha_tail) # Decimal negative

        # Downside Risk annualizzato (decimale positivo)
        downside_vol = self.downside_risk() # Decimal

        # Sharpe, Sortino, Calmar (ratio decimali)
        sharpe = self.sharpe_ratio()
        sortino = self.sortino_ratio()
        calmar = self.calmar_ratio()

        # Pitfall Indicator (ratio decimale, ora conforme al PDF)
        pitfall_indicator_val = self.pitfall_indicator(alpha_cdar=alpha_tail)

        # Penalized Risk (decimale, ora conforme al PDF)
        penalized_risk_val = self.penalized_risk(alpha_tail=alpha_tail)

        # Serenity Ratio (ratio decimale, ora conforme al PDF)
        serenity_ratio_val = self.serenity_ratio(alpha_tail=alpha_tail)


        # Prepara il dizionario dei risultati, formattando per la visualizzazione
        metrics = {}

        # Formatta in percentuale (%) dove appropriato, altrimenti lascia decimale/ratio
        metrics["Total Return (%)"] = total_ret * 100 if total_ret is not np.nan else np.nan
        metrics["Annualized Return (%)"] = ann_ret * 100 if ann_ret is not np.nan else np.nan
        metrics["Annualized Volatility (%)"] = ann_vol * 100 if ann_vol is not np.nan else np.nan
        metrics["Max Drawdown (%)"] = max_dd * 100 if max_dd is not np.nan else np.nan # Converti in % negativo
        metrics["Ulcer Index"] = ulcer * 100 if ulcer is not np.nan else np.nan # Converti Ulcer Index in % per visualizzazione (come suggerito dal PDF)
        metrics["Ulcer Performance Index"] = upi # Ratio decimale

        metrics[f"VaR_Returns({(1-alpha_tail)*100:.0f}%) (%)"] = var_ret * 100 if var_ret is not np.nan else np.nan # VaR sui rendimenti in % negativo
        metrics[f"CVaR_Returns({(1-alpha_tail)*100:.0f}%) (%)"] = cvar_ret * 100 if cvar_ret is not np.nan else np.nan # CVaR sui rendimenti in % negativo
        metrics[f"DaR({(1-alpha_tail)*100:.0f}%) (%)"] = dar_dd * 100 if dar_dd is not np.nan else np.nan # DaR sui drawdown in % negativo
        metrics[f"CDaR({(1-alpha_tail)*100:.0f}%) (%)"] = cdar_dd * 100 if cdar_dd is not np.nan else np.nan # CDaR sui drawdown in % negativo
        metrics["Downside Risk (%)"] = downside_vol * 100 if downside_vol is not np.nan else np.nan # Downside risk annualizzato in %

        metrics["Sharpe Ratio"] = sharpe # Ratio decimale
        metrics["Sortino Ratio"] = sortino # Ratio decimale
        metrics["Calmar Ratio"] = calmar # Ratio decimale

        metrics["Pitfall Indicator"] = pitfall_indicator_val # Ratio decimale
        metrics["Penalized Risk (%)"] = penalized_risk_val * 100 if penalized_risk_val is not np.nan else np.nan # Converti Penalized Risk in % per visualizzazione (come da PDF)
        metrics["Serenity Ratio"] = serenity_ratio_val # Ratio decimale


        # Ordine basato sull'immagine nel PDF + altre metriche calcolate + UPI
        ordered_metrics = {}

        # Metriche principali come nell'immagine del PDF, con UPI inserito
        if "Annualized Return (%)" in metrics: ordered_metrics["Annualized Return (%)"] = metrics["Annualized Return (%)"]
        if "Annualized Volatility (%)" in metrics: ordered_metrics["Annualized Volatility (%)"] = metrics["Annualized Volatility (%)"]
        if "Max Drawdown (%)" in metrics: ordered_metrics["Max Drawdown (%)"] = metrics["Max Drawdown (%)"]
        if "Ulcer Index" in metrics: ordered_metrics["Ulcer Index"] = metrics["Ulcer Index"]
        if "Ulcer Performance Index" in metrics: ordered_metrics["Ulcer Performance Index"] = metrics["Ulcer Performance Index"] # <-- UPI aggiunto qui
        if f"DaR({(1-alpha_tail)*100:.0f}%) (%)" in metrics: ordered_metrics[f"DaR({(1-alpha_tail)*100:.0f}%) (%)"] = metrics[f"DaR({(1-alpha_tail)*100:.0f}%) (%)"]
        if f"CDaR({(1-alpha_tail)*100:.0f}%) (%)" in metrics: ordered_metrics[f"CDaR({(1-alpha_tail)*100:.0f}%) (%)"] = metrics[f"CDaR({(1-alpha_tail)*100:.0f}%) (%)"]
        if "Pitfall Indicator" in metrics: ordered_metrics["Pitfall Indicator"] = metrics["Pitfall Indicator"]
        if "Penalized Risk (%)" in metrics: ordered_metrics["Penalized Risk (%)"] = metrics["Penalized Risk (%)"]
        if "Serenity Ratio" in metrics: ordered_metrics["Serenity Ratio"] = metrics["Serenity Ratio"]

        # Altre metriche calcolate (ordine come prima)
        if "Total Return (%)" in metrics: ordered_metrics["Total Return (%)"] = metrics["Total Return (%)"]
        if "Downside Risk (%)" in metrics: ordered_metrics["Downside Risk (%)"] = metrics["Downside Risk (%)"]
        if "Sharpe Ratio" in metrics: ordered_metrics["Sharpe Ratio"] = metrics["Sharpe Ratio"]
        if "Sortino Ratio" in metrics: ordered_metrics["Sortino Ratio"] = metrics["Sortino Ratio"]
        if "Calmar Ratio" in metrics: ordered_metrics["Calmar Ratio"] = metrics["Calmar Ratio"]
         # VaR/CVaR sui rendimenti per distinguerli da DaR/CDaR
        if f"VaR_Returns({(1-alpha_tail)*100:.0f}%) (%)" in metrics: ordered_metrics[f"VaR_Returns({(1-alpha_tail)*100:.0f}%) (%)"] = metrics[f"VaR_Returns({(1-alpha_tail)*100:.0f}%) (%)"]
        if f"CVaR_Returns({(1-alpha_tail)*100:.0f}%) (%)" in metrics: ordered_metrics[f"CVaR_Returns({(1-alpha_tail)*100:.0f}%) (%)"] = metrics[f"CVaR_Returns({(1-alpha_tail)*100:.0f}%) (%)"]


        return ordered_metrics
