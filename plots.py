import plotly.graph_objects as go
import plotly.express as px
from rolling_calculations import create_short_names # Mantenuto l'import
import plotly.colors # Importa i colori di plotly

def plot_rolling_returns(rolling_returns, title=None):
    """Plotta i rendimenti rolling annualizzati nel tempo."""
    if rolling_returns is None or rolling_returns.empty:
        return go.Figure().update_layout(title="Nessun dato disponibile per i Rendimenti Rolling")

    fig = go.Figure()
    for col in rolling_returns.columns:
        fig.add_trace(go.Scatter(x=rolling_returns.index,
                                 y=rolling_returns[col],
                                 mode='lines',
                                 name=col))

    fig.update_layout(title=title if title else "Rendimenti Rolling Annualizzati",
                      xaxis_title="Data",
                      yaxis_title="Rendimento Annualizzato",
                      yaxis_tickformat='.1%', # Formato percentuale con 1 decimale
                      template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode="x unified", # Mostra tooltip per tutti i trace alla stessa data sull'asse X
                      height=500)
    return fig

def plot_boxplot(rolling_returns, title=None):
    """Plotta la distribuzione dei rendimenti rolling con box plot."""
    if rolling_returns is None or rolling_returns.empty:
         return go.Figure().update_layout(title="Nessun dato disponibile per il Box Plot")

    # Usa create_short_names per abbreviare i nomi delle colonne per il box plot
    short_names_map = create_short_names(rolling_returns.columns.tolist())
    rolling_returns_short = rolling_returns.copy()
    rolling_returns_short.columns = [short_names_map[col] for col in rolling_returns.columns]

    fig = go.Figure()
    for col in rolling_returns_short.columns:
        # Utilizza .dropna() per rimuovere eventuali NaN rimanenti specifici di un box
        fig.add_trace(go.Box(y=rolling_returns_short[col].dropna(),
                             name=col,
                             boxpoints='outliers')) # Mostra gli outliers

    fig.update_layout(title=title if title else "Distribuzione dei Rendimenti Rolling (Box Plot)",
                      yaxis_title="Rendimento Annualizzato",
                      yaxis_tickformat='.1%', # Formato percentuale con 1 decimale
                      template="plotly_white",
                      height=500)
    return fig

def plot_violinplot(rolling_returns, title=None):
    """Plotta la distribuzione dei rendimenti rolling con violin plot."""
    if rolling_returns is None or rolling_returns.empty:
         return go.Figure().update_layout(title="Nessun dato disponibile per il Violin Plot")

    # Usa create_short_names per abbreviare i nomi delle colonne per il violin plot
    short_names_map = create_short_names(rolling_returns.columns.tolist())
    rolling_returns_short = rolling_returns.copy()
    rolling_returns_short.columns = [short_names_map[col] for col in rolling_returns.columns]

    # Melt il DataFrame per il formato necessario a px.violin
    # Assicurati che l'indice 'Data' esista prima di reset_index
    if not 'Data' in rolling_returns_short.columns:
        data_melted = rolling_returns_short.reset_index().melt(id_vars=rolling_returns_short.index.name if rolling_returns_short.index.name else 'index', # Gestisce indice senza nome
                                                               var_name='Indice',
                                                               value_name='Rendimento')
    else:
         data_melted = rolling_returns_short.melt(id_vars='Data', # Se 'Data' è già una colonna
                                                 var_name='Indice',
                                                 value_name='Rendimento')


    # Rimuove le righe con Rendimento NaN prima di plottare
    data_melted = data_melted.dropna(subset=['Rendimento'])

    fig = px.violin(data_melted,
                    x='Indice',
                    y='Rendimento',
                    box=True, # Aggiunge un box plot all'interno del violin
                    points="outliers", # Mostra i punti outlier
                    title=title if title else "Distribuzione dei Rendimenti Rolling (Violin Plot)",
                    template="plotly_white",
                    height=500)

    fig.update_layout(yaxis_title="Rendimento Annualizzato",
                      yaxis_tickformat='.1%', # Formato percentuale con 1 decimale
                      )
    return fig

def plot_min_vs_window(min_median_data, assets=None, title=None):
    """Plotta il rendimento minimo vs finestra temporale per asset."""
    if not min_median_data or not any(data and data['windows'] for data in min_median_data.values()):
        return go.Figure().update_layout(title="Nessun dato disponibile per Rendimento Minimo vs Finestra Temporale")

    fig = go.Figure()
    # Se assets è None o vuoto, usa tutti gli asset nel min_median_data
    assets_to_plot = assets if assets else list(min_median_data.keys())

    for asset in assets_to_plot:
        # Controlla se l'asset esiste nei dati e ha finestre valide
        if asset in min_median_data and min_median_data[asset] and min_median_data[asset]['windows']:
            fig.add_trace(go.Scatter(x=min_median_data[asset]['windows'],
                                     y=min_median_data[asset]['min_values'],
                                     mode='lines+markers',
                                     name=asset))

    fig.update_layout(title=title if title else "Rendimento Minimo vs Finestra Temporale",
                      xaxis_title="Finestra Temporale (anni)",
                      yaxis_title="Rendimento Annualizzato Minimo", # Titolo aggiornato
                      yaxis_tickformat='.1%', # Formato percentuale con 1 decimale
                      template="plotly_white",
                      xaxis=dict(tickmode='linear', tick0=1, dtick=1), # Assicura che l'asse X mostri solo interi per gli anni
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=500)
    return fig

def plot_median_vs_window(min_median_data, assets=None, title=None):
    """Plotta il rendimento mediano vs finestra temporale per asset."""
    if not min_median_data or not any(data and data['windows'] for data in min_median_data.values()):
        return go.Figure().update_layout(title="Nessun dato disponibile per Rendimento Mediano vs Finestra Temporale")

    fig = go.Figure()
    # Se assets è None o vuoto, usa tutti gli asset nel min_median_data
    assets_to_plot = assets if assets else list(min_median_data.keys())

    for asset in assets_to_plot:
         # Controlla se l'asset esiste nei dati e ha finestre valide
        if asset in min_median_data and min_median_data[asset] and min_median_data[asset]['windows']:
            fig.add_trace(go.Scatter(x=min_median_data[asset]['windows'],
                                     y=min_median_data[asset]['median_values'],
                                     mode='lines+markers',
                                     name=asset))

    fig.update_layout(title=title if title else "Rendimento Mediano vs Finestra Temporale",
                      xaxis_title="Finestra Temporale (anni)",
                      yaxis_title="Rendimento Annualizzato Mediano", # Titolo aggiornato
                      yaxis_tickformat='.1%', # Formato percentuale con 1 decimale
                      template="plotly_white",
                      xaxis=dict(tickmode='linear', tick0=1, dtick=1), # Assicura che l'asse X mostri solo interi per gli anni
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      height=500)
    return fig

# --- NUOVE FUNZIONI DI PLOTTING ---

def plot_combined_min_median(min_median_data, assets=None, title=None):
    """Plotta le linee di minimo e mediano combinate per asset con colori sincronizzati."""
    if not min_median_data or not any(data and data['windows'] for data in min_median_data.values()):
        return go.Figure().update_layout(title="Nessun dato disponibile per Rendimento Min/Mediano Combinato")

    fig = go.Figure()
    assets_to_plot = assets if assets else list(min_median_data.keys())

    # Usa una tavolozza di colori da Plotly
    colors = plotly.colors.qualitative.Plotly

    for i, asset in enumerate(assets_to_plot):
        if asset in min_median_data and min_median_data[asset] and min_median_data[asset]['windows']:
            # Seleziona un colore dalla tavolozza usando l'indice dell'asset
            asset_color = colors[i % len(colors)]

            # Linea del minimo (tratteggiata)
            fig.add_trace(go.Scatter(x=min_median_data[asset]['windows'],
                                     y=min_median_data[asset]['min_values'],
                                     mode='lines+markers',
                                     name=f"{asset} (Min)", # Nome per la legenda
                                     line=dict(dash='dash', color=asset_color), # Imposta colore e stile
                                     legendgroup=asset, # Raggruppa per legenda
                                     showlegend=True)) # Mostra nella legenda

            # Linea del mediano (continua)
            fig.add_trace(go.Scatter(x=min_median_data[asset]['windows'],
                                     y=min_median_data[asset]['median_values'],
                                     mode='lines+markers',
                                     name=f"{asset} (Med)", # Nome per la legenda
                                     line=dict(dash='solid', color=asset_color), # Imposta stesso colore e stile
                                     legendgroup=asset, # Raggruppa per legenda
                                     showlegend=True)) # Mostra nella legenda


    fig.update_layout(title=title if title else "Rendimenti Minimo e Mediano vs Finestra Temporale",
                      xaxis_title="Finestra Temporale (anni)",
                      yaxis_title="Rendimento Annualizzato",
                      yaxis_tickformat='.1%',
                      template="plotly_white",
                      xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode="x unified", # Mostra tooltip per tutti i trace alla stessa finestra
                      height=500)
    return fig

def plot_detailed_window_analysis(min_median_data, selected_assets, title=None):
    """Plotta linee min/median e box plot per finestra per 1 o 2 asset selezionati."""
    if not min_median_data or not selected_assets or not any(asset in min_median_data and min_median_data[asset] and min_median_data[asset]['windows'] for asset in selected_assets):
        return go.Figure().update_layout(title="Seleziona 1 o 2 asset per l'Analisi Dettagliata per Finestra")

    fig = go.Figure()

    # Determina lo shift per i box plot se ci sono due asset per non sovrapporli
    # Lo shift è relativo all'unità sull'asse x (finestra temporale)
    box_shift = 0.2 if len(selected_assets) == 2 else 0
    box_width = 0.3 if len(selected_assets) == 2 else 0.5 # Larghezza del box plot

    # Usa una tavolozza di colori da Plotly
    colors = plotly.colors.qualitative.Plotly

    # Itera sugli asset selezionati (massimo 2)
    for i, asset in enumerate(selected_assets):
        if asset in min_median_data and min_median_data[asset] and min_median_data[asset]['windows']:
            asset_data = min_median_data[asset]
            windows = asset_data['windows']
            min_values = asset_data['min_values']
            median_values = asset_data['median_values']
            all_returns_by_window = asset_data['all_returns_by_window'] # I dati grezzi per i box plot

            # Seleziona un colore dalla tavolozza usando l'indice dell'asset selezionato
            asset_color = colors[i % len(colors)]


            # Aggiungi le linee del minimo e del mediano per l'asset corrente
            fig.add_trace(go.Scatter(x=windows,
                                     y=min_values,
                                     mode='lines+markers',
                                     name=f"{asset} (Min)",
                                     line=dict(dash='dash', color=asset_color), # Colore basato sull'asset
                                     legendgroup=asset)) # Gruppo per la legenda

            fig.add_trace(go.Scatter(x=windows,
                                     y=median_values,
                                     mode='lines+markers',
                                     name=f"{asset} (Med)",
                                     line=dict(dash='solid', color=asset_color), # Colore basato sull'asset
                                     legendgroup=asset)) # Gruppo per la legenda

            # Aggiungi i box plot per ogni finestra
            # Usiamo asset_color come colore base per i box dell'asset
            box_color = asset_color

            for window_year in windows:
                if window_year in all_returns_by_window:
                    returns_for_window = all_returns_by_window[window_year]
                    if returns_for_window: # Assicurati che la lista non sia vuota
                        # Posiziona il box plot sulla finestra temporale corretta
                        # Applica uno shift se ci sono due asset
                        box_x_pos = window_year + (box_shift * (i - (len(selected_assets)-1)/2))

                        fig.add_trace(go.Box(
                            y=returns_for_window, # Dati per il box plot
                            name=f"Finestra {window_year} anni ({asset})", # Nome per tooltip/legenda (può diventare lungo)
                            boxpoints=False, # Non mostrare i punti individuali nel box plot per non affollare
                            marker_color=box_color, # Colore del box
                            x=[box_x_pos] * len(returns_for_window), # Posiziona i dati sulla coordinata x
                            boxmean=True, # Mostra la media
                            notched=True, # Mostra la mediana con 'tacche'
                            width=box_width, # Larghezza del box
                            showlegend=False, # Non mostrare una voce di legenda per ogni box
                            hoveron="boxes" # Mostra tooltip quando si passa il mouse sulle scatole
                        ))


    fig.update_layout(title=title if title else "Analisi Dettagliata per Finestra Temporale",
                      xaxis_title="Finestra Temporale (anni)",
                      yaxis_title="Rendimento Annualizzato",
                      yaxis_tickformat='.1%',
                      template="plotly_white",
                      xaxis=dict(
                          tickmode='linear', # Assicura che l'asse X mostri solo interi per gli anni
                          tick0=1,
                          dtick=1,
                          range=[0.5, max(min_median_data[selected_assets[0]]['windows'] if selected_assets and selected_assets[0] in min_median_data and min_median_data[selected_assets[0]] else [1]) + 0.5] # Imposta il range per includere i box plot agli estremi
                        ),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode="closest", # Tooltip specifico per l'elemento su cui si passa il mouse
                      height=600 # Aumenta l'altezza per accomodare meglio i box plot
                      )
    return fig
