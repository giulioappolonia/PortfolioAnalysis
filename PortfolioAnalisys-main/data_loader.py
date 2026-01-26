import pandas as pd
import streamlit as st

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'Data' not in data.columns:
            st.error("Errore: Il file CSV deve contenere una colonna 'Data'")
            return None
        try:
            data['Data'] = pd.to_datetime(data['Data'], format='%m/%Y')
        except ValueError:
            try:
                data['Data'] = pd.to_datetime(data['Data'])
            except:
                st.error("Errore: Impossibile convertire le date nel formato corretto")
                return None
        data.set_index('Data', inplace=True)
        if data.empty:
            st.error("Errore: Il file non contiene dati validi")
            return None
        return data
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        return None