from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataFilter:
    """Gestionnaire de filtrage et de sélection des données"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.filters_applied: list[Dict[str, Any]] = []
        self.original_shape: Optional[Tuple[int, int]] = None

    def _ensure_original_shape(self, df: pd.DataFrame) -> None:
        if self.original_shape is None:
            self.original_shape = df.shape

    @staticmethod
    def _boolify(s: pd.Series) -> pd.Series:
        """Convertit diverses représentations en booléen."""
        if s.dtype == "bool":
            return s
        true_set = {"1", "true", "t", "yes", "y", "vrai", "oui"}
        false_set = {"0", "false", "f", "no", "n", "faux", "non"}

        def to_bool(x: Any) -> bool:
            if pd.isna(x):
                return False
            if isinstance(x, (int, float)):
                return bool(int(x))
            if isinstance(x, str):
                v = x.strip().lower()
                if v in true_set:
                    return True
                if v in false_set:
                    return False
            return bool(x)

        return s.map(to_bool)

    def _log_filter(self, filter_name: str, new_shape: Tuple[int, int], *, removed: int) -> None:
        """Log l'application d'un filtre"""
        if self.original_shape is None:
            self.original_shape = new_shape

        retention_rate = (new_shape[0] / self.original_shape[0]) * 100 if self.original_shape[0] > 0 else 0

        self.filters_applied.append({
            "filter": filter_name,
            "timestamp": datetime.now(),
            "new_shape": new_shape,
            "removed_rows": removed,
            "remaining_rows": new_shape[0],
            "remaining_columns": new_shape[1],
        })
        logger.info(f"Filtre appliqué: {filter_name} - {removed} lignes supprimées ({new_shape[0]} restantes, {retention_rate:.1f}% rétention)")

    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = df.copy()
        # 2. Filtrer sur les sinistres validés (critère essentiel)
        if 'est_sinistre' in filtered_df.columns:
            initial_rows = len(filtered_df)
            filtered_df['est_sinistre'] = self._boolify(filtered_df['est_sinistre_cas_valide'])
            filtered_df = filtered_df[filtered_df['est_sinistre'] == True]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter("Sinistres non validés", filtered_df.shape, removed=removed)

        # 3. Filtrer les sinistres clos uniquement
        if 'statut_sinistre' in filtered_df.columns:
            initial_rows = len(filtered_df)
            filtered_df = filtered_df[
                filtered_df['statut_sinistre'].str.lower().isin(['clos', 'fermé', 'closed'])
            ]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter("Sinistres non clos", filtered_df.shape, removed=removed)

        # 4. Filtrer les montants valides (critère critique)
        if 'montant_charge_brut' in filtered_df.columns:
            initial_rows = len(filtered_df)
            min_montant = self.config.get('min_montant', 10)
            max_montant = self.config.get('max_montant', 500000)
            
            montant_numeric = pd.to_numeric(filtered_df['montant_charge_brut'], errors='coerce')
            mask = (
                montant_numeric.notna() & 
                (montant_numeric >= min_montant) & 
                (montant_numeric <= max_montant)
            )
            filtered_df = filtered_df[mask]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter(f"Montants invalides (<{min_montant} ou >{max_montant})", 
                               filtered_df.shape, removed=removed)

        # 5. Filtrer le géocodage valide
        if 'geocoding_validity' in filtered_df.columns:
            initial_rows = len(filtered_df)
            filtered_df['geocoding_validity'] = self._boolify(filtered_df['geocoding_validity'])
            filtered_df = filtered_df[filtered_df['geocoding_validity'] == True]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter("Géocodage invalide", filtered_df.shape, removed=removed)

        # 6. Supprimer les doublons (en fin pour être plus efficace)
        initial_rows = len(filtered_df)
        filtered_df = filtered_df.drop_duplicates(subset=['police', 'cod_sin'], keep='first')
        removed = initial_rows - len(filtered_df)
        if removed > 0:
            self._log_filter("Doublons (police + cod_sin)", filtered_df.shape, removed=removed)

        return filtered_df

    def apply_business_filters(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Applique les filtres métier spécifiques sécheresse"""
        filtered_df = df.copy()
        
        # Fusionner config et kwargs
        filter_config = {**self.config, **kwargs}
        
        # 1. Filtre par période (important pour analyse temporelle)
        date_range = filter_config.get('date_range')
        if date_range and 'date_sinistre' in filtered_df.columns:
            initial_rows = len(filtered_df)
            start_date, end_date = date_range
            
            date_col = pd.to_datetime(filtered_df['date_sinistre'], errors='coerce')
            mask = (
                date_col.notna() &
                (date_col >= pd.to_datetime(start_date)) &
                (date_col <= pd.to_datetime(end_date))
            )
            filtered_df = filtered_df[mask]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter(f"Période {start_date} à {end_date}", 
                               filtered_df.shape, removed=removed)

        # 2. Filtre par types de risque RGA
        risk_types = filter_config.get('risk_types')
        if risk_types and 'risque_rga_gcl' in filtered_df.columns:
            initial_rows = len(filtered_df)
            mask = filtered_df['risque_rga_gcl'].isin(risk_types)
            filtered_df = filtered_df[mask]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter(f"Types de risque RGA", filtered_df.shape, removed=removed)

        # 3. Filtre par surface (cohérence métier)
        if 'surface_police' in filtered_df.columns:
            initial_rows = len(filtered_df)
            surface_numeric = pd.to_numeric(filtered_df['surface_police'], errors='coerce')
            mask = (
                surface_numeric.notna() & 
                (surface_numeric >= 10) & 
                (surface_numeric <= 1000)
            )
            filtered_df = filtered_df[mask]
            removed = initial_rows - len(filtered_df)
            if removed > 0:
                self._log_filter("Surfaces aberrantes", filtered_df.shape, removed=removed)

        return filtered_df

    def get_filter_summary(self) -> Dict[str, Any]:
        """Retourne un résumé détaillé et robuste des filtres appliqués."""
        logger.debug("get_filter_summary called")

        filters_applied = getattr(self, "filters_applied", None) or []
        original_shape = getattr(self, "original_shape", None)

        # Cas: aucun filtre appliqué
        if not filters_applied:
            rows0 = int(original_shape[0]) if original_shape else 0
            return {
                "message": "Aucun filtre appliqué",
                "filters_count": 0,
                "total_rows_removed": 0,
                "original_shape": tuple(original_shape) if original_shape else None,
                "final_shape": tuple(original_shape) if original_shape else None,
                "overall_retention_rate": round(100.0 if rows0 > 0 else 0.0, 1),
                "filters_detail": []
            }

        # Agrégation sécurisée
        total_removed = 0
        final_shape = None
        details = []

        prev_rows = None
        if original_shape and len(original_shape) >= 1:
            prev_rows = int(original_shape[0])

        for f in filters_applied:
            # Valeurs sûres
            old_shape = tuple(f.get("old_shape")) if f.get("old_shape") else (prev_rows, None)
            new_shape = tuple(f.get("new_shape")) if f.get("new_shape") else None
            removed = int(f.get("removed_rows", 0) or 0)

            total_removed += removed
            if new_shape:
                final_shape = new_shape
                prev_rows = int(new_shape[0]) if len(new_shape) > 0 and new_shape[0] is not None else prev_rows

            # Retention locale (si absente)
            retention = f.get("retention_rate")
            if retention is None:
                try:
                    prev_n = int(old_shape[0]) if old_shape and old_shape[0] is not None else None
                    new_n = int(new_shape[0]) if new_shape and new_shape[0] is not None else None
                    if prev_n and new_n is not None and prev_n > 0:
                        retention = round(100.0 * new_n / prev_n, 1)
                except Exception:
                    retention = None

            details.append({
                "filter": f.get("filter", "unknown"),
                "removed": removed,
                "retention": retention
            })

        # Si on n'a pas de final_shape mais qu'on connaît original_shape
        if final_shape is None:
            final_shape = tuple(original_shape) if original_shape else None

        # Rétention globale
        overall_retention = 0.0
        try:
            if original_shape and final_shape and original_shape[0]:
                overall_retention = round(100.0 * float(final_shape[0]) / float(original_shape[0]), 1)
        except Exception:
            overall_retention = 0.0

        return {
            "filters_count": len(filters_applied),
            "total_rows_removed": int(total_removed),
            "original_shape": tuple(original_shape) if original_shape else None,
            "final_shape": tuple(final_shape) if final_shape else None,
            "overall_retention_rate": overall_retention,
            "filters_detail": details
        }

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Génère un rapport de qualité des données filtrées"""
        report = {}
        
        # Statistiques générales
        report['shape'] = df.shape
        report['missing_data'] = df.isnull().sum().to_dict()
        report['memory_usage'] = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Analyse de la variable cible
        if 'montant_charge_brut' in df.columns:
            montant = pd.to_numeric(df['montant_charge_brut'], errors='coerce')
            report['target_stats'] = {
                'count': montant.count(),
                'mean': montant.mean(),
                'median': montant.median(),
                'std': montant.std(),
                'min': montant.min(),
                'max': montant.max(),
                'q25': montant.quantile(0.25),
                'q75': montant.quantile(0.75)
            }
        
        # Distribution des risques RGA
        if 'risque_rga_gcl' in df.columns:
            report['rga_distribution'] = df['risque_rga_gcl'].value_counts().to_dict()
        
        return report