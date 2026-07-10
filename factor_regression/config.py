# factor_regression/config.py

# Mappatura dei dataset Fama-French per pandas-datareader in base alla regione e frequenza
REGION_DATASETS = {
    "USA": {
        "monthly": {
            "factors": "F-F_Research_Data_5_Factors_2x3",
            "momentum": "F-F_Momentum_Factor"
        },
        "daily": {
            "factors": "F-F_Research_Data_5_Factors_2x3_daily",
            "momentum": "F-F_Momentum_Factor_daily"
        }
    },
    "Developed Markets": {
        "monthly": {
            "factors": "Developed_5_Factors",
            "momentum": "Developed_Mom_Factor"
        },
        "daily": {
            "factors": "Developed_5_Factors_Daily",
            "momentum": "Developed_Mom_Factor_Daily"
        }
    },
    "Europe": {
        "monthly": {
            "factors": "Europe_5_Factors",
            "momentum": "Europe_Mom_Factor"
        },
        "daily": {
            "factors": "Europe_5_Factors_Daily",
            "momentum": "Europe_Mom_Factor_Daily"
        }
    },
    "Emerging Markets": {
        "monthly": {
            "factors": "Emerging_5_Factors",
            "momentum": "Emerging_Mom_Factor"
        },
        "daily": {
            "factors": None,
            "momentum": None
        }
    },
    "Global": {
        "monthly": {
            "factors": "Global_5_Factors",
            "momentum": "Global_Mom_Factor"
        },
        "daily": {
            "factors": "Global_5_Factors_Daily",
            "momentum": "Global_Mom_Factor_Daily"
        }
    }
}

# Parametri di default per la rolling regression
DEFAULT_ROLLING_WINDOW = {
    "monthly": 36,
    "daily": 252
}

# Limiti minimi per validazione osservazioni
MIN_OBSERVATIONS = {
    "monthly": 24,
    "daily": 126
}
