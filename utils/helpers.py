from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

def validate_data_structure(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Valide la structure des données"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    return {
        'valid': len(missing_columns) == 0,
        'missing_columns': missing_columns,
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'columns_found': [col for col in required_columns if col in df.columns]
    }

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Détecte automatiquement les types de colonnes métier"""
    column_types = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(keyword in col_lower for keyword in ['montant', 'cout', 'charge', 'reserve']):
            column_types[col] = 'monetary'
        elif any(keyword in col_lower for keyword in ['date', 'time']):
            column_types[col] = 'datetime'
        elif any(keyword in col_lower for keyword in ['surface', 'age', 'annee']):
            column_types[col] = 'numeric'
        elif any(keyword in col_lower for keyword in ['code', 'postal', 'cp']):
            column_types[col] = 'geographic'
        elif any(keyword in col_lower for keyword in ['risque', 'nature', 'type']):
            column_types[col] = 'categorical'
        else:
            column_types[col] = 'unknown'
    
    return column_types

def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Crée un dictionnaire des données"""
    dict_data = []
    
    for col in df.columns:
        col_info = {
            'column_name': col,
            'data_type': str(df[col].dtype),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique_values': df[col].nunique(),
            'memory_usage': df[col].memory_usage(deep=True)
        }
        
        # Ajouter des stats pour les colonnes numériques
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min_value': df[col].min(),
                'max_value': df[col].max(),
                'mean_value': df[col].mean(),
                'std_value': df[col].std()
            })
        
        # Ajouter les valeurs les plus fréquentes pour les colonnes catégorielles
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            top_values = df[col].value_counts().head(3)
            col_info['top_values'] = ', '.join([f"{val}({count})" for val, count in top_values.items()])
        
        dict_data.append(col_info)
    
    return pd.DataFrame(dict_data)

def safe_numeric_conversion(series: pd.Series) -> pd.Series:
    """Conversion sécurisée vers numérique"""
    # Nettoyer les valeurs communes non-numériques
    cleaned = series.astype(str).str.replace(r'[^\d.,+-]', '', regex=True)
    cleaned = cleaned.str.replace(',', '.')  # Gérer les décimales européennes
    
    return pd.to_numeric(cleaned, errors='coerce')

def ensure_directory_exists(path: Path) -> Path:
    """S'assurer qu'un répertoire existe"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_number(value: float, format_type: str = 'currency') -> str:
    """Formate un nombre selon le type demandé"""
    if pd.isna(value):
        return "N/A"
    
    if format_type == 'currency':
        return f"{value:,.0f} €"
    elif format_type == 'percentage':
        return f"{value:.2%}"
    elif format_type == 'decimal':
        return f"{value:.2f}"
    else:
        return str(value)