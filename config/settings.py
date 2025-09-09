from pathlib import Path
from typing import Dict, Any, List, Tuple

# Chemins de base
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Créer les dossiers s'ils n'existent pas
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "exports").mkdir(exist_ok=True)

# Configuration par défaut 
DEFAULT_CONFIG = {
    # Paramètres de lecture des fichiers
    "csv_read_params": {
        "parse_dates": ["date_sinistre", "date_cloture_sinistre", "egms_start_date", "egms_end_date"],
        "low_memory": False,
        "dtype": {
            "police": "str", 
            "cpostal": "str",
            "identifiantintermediaire": "str"
        }
    },
    
    "excel_read_params": {
        "parse_dates": ["date_sinistre", "date_cloture_sinistre", "egms_start_date", "egms_end_date"],
        "dtype": {
            "police": "str", 
            "cpostal": "str",
            "identifiantintermediaire": "str"
        }
    },    
    
    # Paramètres de filtrage - adaptés aux données sécheresse
    "filter_params": {
        "apply_quality": True,
        "apply_business": True,
        "date_range": ("2018-01-01", "2025-12-31"),  # Élargi pour plus de données
        "risk_types": ["faible", "moyen", "fort", "maximal"],  # Tous les niveaux RGA
        "min_montant": 10,  # Montant minimum plus réaliste
        "max_montant": 500000,  # Ajusté selon les sinistres sécheresse
    },
    
    "preprocess_params": {
        "outlier_strategy": "clip",
        "outlier_q": (0.005, 0.995),  # Plus conservateur pour les sinistres
        "impute_numeric": "median",
        "impute_categorical": "mode",
        "apply_encoding": True,
        "encoding_method": "label"
    },
    
    "stats_params": {
        "correlation_threshold": 0.05,  # Plus sensible pour détecter faibles corrélations
        "significance_level": 0.05,
    },
    
    "viz_params": {
        "style": "whitegrid",
        "figsize": (12, 8),
        "save_plots": True,
        "dpi": 300,
        "format": "png"
    },
    
    "export_params": {
        "format": "all",
        "include_raw": False,
        "timestamp": True
    }
}

# Variables métier spécifiques aux sinistres sécheresse
BUSINESS_VARIABLES = {
    # Variables cibles
    "target_variables": ["montant_charge_brut"],
    
    # Variables explicatives principales - organisées par groupe
    "base_variables": [
        "prime", "surface_police", "nb_pieces_police", "engagements"
    ],
    
    "risk_variables": [
        "risque_rga_gcl", "risque_rga_gcl_detail", "risk_retraitGonflementArgile"
    ],
    
    "building_variables": [
        "surface_batiment", "hauteur_batiment", "nb_etages_batiment",
        "annee_construction_batiment", "type_mur_batiment", "type_toit_batiment"
    ],
    
    "bdtopo_variables": [
        "bdtopo_surface", "bdtopo_hauteur", "bdtopo_nb_etages", 
        "bdtopo_leger_bool", "bdtopo_matmurs", "bdtopo_usage1"
    ],
    
    "soil_variables": [
        "compo_sol_has_argile", "compo_sol_has_sable", "compo_sol_has_calcaire",
        "dem_pente_maximale", "dem_altitude"
    ],
    
    "weather_variables": [
        "weather_precip_mean", "weather_precip_std", "weather_temp_mean",
        "weather_humidity_mean"
    ],
    
    "sma_variables": [
        "sma_mean", "sma_std", "sma_min", "sma_max"
    ],
    
    "egms_variables": [
        "egms_ts_slope", "egms_ts_std", "egms_ts_range", "egms_ts_velocity_max_abs"
    ],
    
    "tree_variables": [
        "tree_dist_3", "tree_dist_5", "tree_dist_8", "tree_dist_12"
    ],
    
    "temporal_variables": ["date_sinistre", "date_cloture_sinistre", "exercice"],
    "geographic_variables": ["cpostal", "ville", "latitude", "longitude"],
}

# Seuils et règles métier spécifiques sécheresse
BUSINESS_RULES = {
    "montant_min": 100,
    "montant_max": 500000,
    "surface_min": 10,
    "surface_max": 1000,
    "age_batiment_max": 200,
    "delai_cloture_max": 365 * 5,  # 5 ans max pour sécheresse
    "prime_min": 50,
    "prime_max": 50000,
}

# Filtres de qualité spécifiques aux données sécheresse
QUALITY_FILTERS = {
    "required_columns": [
        "montant_charge_brut", "prime", "date_sinistre", "risque_rga_gcl"
    ],
    "boolean_columns": [
        "est_sinistre", "geocoding_validity", "cas_ai6_is_3_or_4", 
        "est_sinistre_cas_valide", "compo_sol_has_argile"
    ],
    "categorical_columns": [
        "risque_rga_gcl", "mode_occupation_gcl", "type_mur_batiment",
        "statut_sinistre"
    ]
}