import pandas as pd

def convert_history_index(input_file: str, output_file: str) -> None:
    """
    Estrae i prezzi mensili dal foglio 'History Index' di un file Excel
    e li esporta in formato CSV (MM/YYYY, numero decimale con '.', senza migliaia).
    """
    # 1. Carica il foglio con header a riga 7 (header=6 zero‑based)
    raw = pd.read_excel(input_file,
                        sheet_name='History Index',
                        header=6)

    # 2. Trova la prima riga completamente vuota e tronca lì
    empty_mask = raw.isna().all(axis=1)
    if empty_mask.any():
        first_empty_idx = empty_mask[empty_mask].index[0]
        df = raw.iloc[:first_empty_idx].copy()
    else:
        df = raw.copy()

    # 3. Format della colonna data (prima colonna) in MM/YYYY
    df.iloc[:, 0] = (
        pd.to_datetime(df.iloc[:, 0], dayfirst=False, errors='coerce')
          .dt.strftime('%m/%Y')
    )

    # 4. Rimuove il separatore migliaia e converte i prezzi a float
    for col in df.columns[1:]:
        df[col] = (
            df[col]
              .astype(str)
              .str.replace(',', '', regex=False)
              .astype(float)
        )

    # 5. Rinomina la prima colonna in 'Data'
    df.columns = ['Data'] + list(df.columns[1:])

    # 6. Esporta in CSV: punto come separatore decimale, senza indice
    df.to_csv(output_file,
              index=False,
              float_format='%.3f',
              sep=',')

    print(f"Conversione completata. File salvato in: {output_file}")


if __name__ == '__main__':
    input_path = 'historyIndex.xls'               # percorso del file Excel
    output_path = 'converted_history_index.csv'   # percorso di destinazione CSV
    convert_history_index(input_path, output_path)
