import pandas as pd
import streamlit as st
import os

def load_data(file_path):
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.xls', '.xlsx']:
            # Logica per file Excel (MSCI)
            try:
                # 1. Carica il foglio con header a riga 7 (header=6 zero-based)
                # Tenta di caricare 'History Index', se fallisce prova il primo foglio
                try:
                    raw = pd.read_excel(file_path, sheet_name='History Index', header=6)
                except ValueError:
                    raw = pd.read_excel(file_path, header=6) # Fallback al primo foglio

                # 2. Trova la prima riga completamente vuota e tronca lì
                empty_mask = raw.isna().all(axis=1)
                if empty_mask.any():
                    first_empty_idx = empty_mask[empty_mask].index[0]
                    data = raw.iloc[:first_empty_idx].copy()
                else:
                    data = raw.copy()

                # 3. Rinomina la prima colonna in 'Data'
                data.rename(columns={data.columns[0]: 'Data'}, inplace=True)
                
                # 4. Rimuove righe dove 'Data' è NaT o NaN (pulizia extra)
                data = data.dropna(subset=['Data'])

                # 5. Format della colonna data
                # MSCI usa spesso formati diversi, proviamo a convertirlo in datetime
                # La logica originale faceva: pd.to_datetime(..., dayfirst=False).dt.strftime('%m/%Y')
                # Qui convertiamo direttamente in datetime objects per l'indice
                data['Data'] = pd.to_datetime(data['Data'], dayfirst=False, errors='coerce')
                
                # 6. Pulizia colonne numeriche (rimuovi virgole migliaia)
                for col in data.columns:
                    if col != 'Data':
                        # Converti in stringa, rimuovi virgole, converti in float
                        # Gestisce anche se sono già numerici o misti
                        data[col] = data[col].astype(str).str.replace(',', '', regex=False)
                        data[col] = pd.to_numeric(data[col], errors='coerce')

            except Exception as e:
                st.error(f"Errore specifico nel parsing del file Excel: {str(e)}")
                return None

        else:
            # Logica per file CSV (Curvo / Standard)
            try:
                # Usa il motore python per rilevare automaticamente il separatore (virgola o punto e virgola)
                data = pd.read_csv(file_path, sep=None, engine='python')
                
                # Normalizzazione nomi colonne: rimuove spazi extra
                data.columns = data.columns.str.strip()
                
            except Exception as e:
                st.error(f"Errore nella lettura del CSV: {str(e)}")
                return None
            
            # Normalizzazione nomi colonne: se esiste 'Date', rinominalo in 'Data'
            if 'Date' in data.columns and 'Data' not in data.columns:
                data.rename(columns={'Date': 'Data'}, inplace=True)

            if 'Data' not in data.columns:
                st.error("Errore: Il file CSV deve contenere una colonna 'Data' o 'Date'")
                return None
            
            # Formattazione dati numerici
            for col in data.columns:
                if col != 'Data':
                    # Se la colonna è object (stringa), proviamo a convertirla
                    if data[col].dtype == object:
                        try:
                            # Tenta conversione diretta (gestisce standard US con punto decimale)
                            data[col] = pd.to_numeric(data[col])
                        except:
                            # Se fallisce, potrebbe essere formato EU (punto migliaia, virgola decimale)
                            # O formato sporco. Proviamo a rimuovere 'punto' come migliaia se sembra EU.
                             # NOTA: Per sicurezza su questo specifico file che usa punti decimali, 
                             # evitiamo sostituzioni aggressive se non strettamente necessario.
                             # Facciamo solo conversioni 'coerce' dopo pulizia base.
                            clean_col = data[col].astype(str).str.replace(',', '', regex=False) # rimuove virgola migliaia US
                            data[col] = pd.to_numeric(clean_col, errors='coerce')

            # Parsing date CSV
            try:
                # Tenta prima il formato MM/YYYY tipico del file
                data['Data'] = pd.to_datetime(data['Data'], format='%m/%Y')
            except ValueError:
                try:
                    # Tenta formato generico
                    data['Data'] = pd.to_datetime(data['Data'])
                except:
                    st.error("Errore: Impossibile convertire le date nel formato corretto nel CSV")
                    return None

        # --- Logica Comune Finale ---
        
        # Rimuovi righe con Data NaT (se il parsing è fallito per alcune righe)
        data = data.dropna(subset=['Data'])

        data.set_index('Data', inplace=True)
        
        # Ordina per data per sicurezza
        data.sort_index(inplace=True)

        if data.empty:
            st.error("Errore: Il file non contiene dati validi dopo l'elaborazione")
            return None
            
        return data

    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        return None