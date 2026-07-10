# factor_regression/plots.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

def plot_cumulative_returns(df: pd.DataFrame, regression_result: Dict[str, Any], fitted_excess: pd.Series = None) -> go.Figure:
    """
    Grafico a linee dei rendimenti cumulati in eccesso: effettivo dell'asset rispetto a quello stimato (fitted) dal modello.
    
    Formula cumulata semplice: (1 + R).cumprod() - 1
    """
    # Estraiamo i valori fitted (in eccesso) dal modello di regressione se non forniti
    if fitted_excess is None:
        fitted_excess = regression_result['results_object'].fittedvalues
    
    # Calcolo dei rendimenti cumulati semplici
    cum_actual = (1 + df['asset_excess_return']).cumprod() - 1
    cum_fitted = (1 + fitted_excess).cumprod() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=cum_actual * 100, # In percentuale per leggibilità
        name="Asset (In Eccesso)",
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=cum_fitted * 100, # In percentuale per leggibilità
        name="Modello Fitted",
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Confronto Rendimenti Cumulati in Eccesso (Asset vs. Modello)",
        xaxis_title="Data",
        yaxis_title="Rendimento Cumulato (%)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_factor_boxplot(factors_df: pd.DataFrame) -> go.Figure:
    """
    Box plot dei rendimenti dei singoli fattori per mostrare la variabilità storica e la distribuzione.
    """
    factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    
    # Moltiplichiamo per 100 per visualizzare in percentuale
    df_pct = factors_df[factors_list] * 100
    
    # Riorganizziamo in formato long per Plotly Express
    df_long = df_pct.melt(var_name='Fattore', value_name='Rendimento (%)')
    
    fig = px.box(
        df_long,
        x='Fattore',
        y='Rendimento (%)',
        color='Fattore',
        title='Distribuzione e Variabilità dei Rendimenti dei Fattori',
        template='plotly_white'
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_rolling_betas(rolling_df: pd.DataFrame) -> go.Figure:
    """
    Grafico a linee dei beta dei fattori e di alpha nel tempo.
    """
    fig = go.Figure()
    
    # Mappatura colori per i fattori
    colors = {
        'beta_Mkt-RF': '#1f77b4',
        'beta_SMB': '#aec7e8',
        'beta_HML': '#ff7f0e',
        'beta_RMW': '#2ca02c',
        'beta_CMA': '#d62728',
        'beta_Mom': '#9467bd',
        'alpha': '#8c564b'
    }
    
    label_dict = {
        'beta_Mkt-RF': 'Beta Market (Mkt-RF)',
        'beta_SMB': 'Beta Size (SMB)',
        'beta_HML': 'Beta Value (HML)',
        'beta_RMW': 'Beta Profitability (RMW)',
        'beta_CMA': 'Beta Investment (CMA)',
        'beta_Mom': 'Beta Momentum (Mom)',
        'alpha': 'Alpha (Intercetta)'
    }
    
    for col in colors.keys():
        if col in rolling_df.columns:
            fig.add_trace(go.Scatter(
                x=rolling_df.index,
                y=rolling_df[col],
                name=label_dict[col],
                line=dict(color=colors[col], width=2)
            ))
            
    fig.update_layout(
        title="Andamento Temporale delle Esposizioni Fattoriali (Beta Rolling)",
        xaxis_title="Data",
        yaxis_title="Valore dei Coefficienti",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_factor_contributions(contrib_df: pd.DataFrame) -> go.Figure:
    """
    Grafico a barre dei contributi medi annualizzati (o periodici) dei fattori al rendimento stimato.
    """
    # Filtriamo l'eventuale riga 'Fitted' o totali se presente, contrib_df ha colonne ['factor', 'contribution']
    # Rimuoviamo righe nulle
    df_plot = contrib_df.dropna(subset=['contribution']).copy()
    
    # Moltiplichiamo per 100 per visualizzare in percentuale
    df_plot['contribution_pct'] = df_plot['contribution'] * 100
    
    # Determiniamo il colore (verde positivo, rosso negativo)
    df_plot['color'] = np.where(df_plot['contribution_pct'] >= 0, '#2ca02c', '#d62728')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot['factor'],
        y=df_plot['contribution_pct'],
        marker_color=df_plot['color'],
        text=df_plot['contribution_pct'].round(2).astype(str) + '%',
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Scomposizione del Rendimento Medio Annuo Atteso (Modello Fitted)",
        xaxis_title="Fattore / Componente",
        yaxis_title="Contributo Annuo Stimato (%)",
        template="plotly_white",
        hovermode="y unified"
    )
    return fig


def plot_factor_correlation(factors_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap delle correlazioni storiche tra i fattori.
    """
    factors_list = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']
    corr_matrix = factors_df[factors_list].corr()
    
    # Arrotondiamo per visualizzazione
    z_values = corr_matrix.values
    x_labels = corr_matrix.columns
    y_labels = corr_matrix.index
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale='RdBu_r', # Rosso negativo, Blu positivo
        zmin=-1,
        zmax=1,
        text=np.round(z_values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    
    fig.update_layout(
        title="Matrice delle Correlazioni tra i Fattori",
        template="plotly_white"
    )
    return fig


def plot_rolling_betas_boxplot(rolling_df: pd.DataFrame) -> go.Figure:
    """
    Crea un box plot che mostra la distribuzione e variabilità dei beta rolling nel tempo per ciascun fattore.
    Questo aiuta a capire la variabilità storica dell'esposizione ai fattori.
    """
    beta_cols = [
        'beta_Mkt-RF', 'beta_SMB', 'beta_HML',
        'beta_RMW', 'beta_CMA', 'beta_Mom'
    ]
    label_dict = {
        'beta_Mkt-RF': 'Market (Mkt-RF)',
        'beta_SMB': 'Size (SMB)',
        'beta_HML': 'Value (HML)',
        'beta_RMW': 'Profitability (RMW)',
        'beta_CMA': 'Investment (CMA)',
        'beta_Mom': 'Momentum (Mom)'
    }
    
    # Filtriamo solo le colonne presenti
    available_cols = [c for c in beta_cols if c in rolling_df.columns]
    df_subset = rolling_df[available_cols].rename(columns=label_dict)
    
    # Riorganizziamo in formato long per Plotly Express
    df_long = df_subset.melt(var_name='Fattore', value_name='Beta')
    
    fig = px.box(
        df_long,
        x='Fattore',
        y='Beta',
        color='Fattore',
        title='Distribuzione e Stabilità dei Beta Rolling nel Tempo',
        template='plotly_white'
    )
    
    fig.update_layout(showlegend=False)
    return fig
