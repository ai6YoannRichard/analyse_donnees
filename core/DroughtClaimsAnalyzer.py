from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta
import warnings

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

# Import de votre pipeline existant
from .pipeline import SinistreAnalysisPipeline
from .StatisticsManager import StatisticsManager

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def _load_config_file(path_like) -> dict:
        p = Path(path_like)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            import yaml  # pip install pyyaml si besoin
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        elif suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        else:
            raise ValueError(f"Unsupported config file type: {suffix}")

class DroughtClaimsAnalyzer(SinistreAnalysisPipeline):
    """
    Analyseur spécialisé pour les sinistres sécheresse
    Étend le pipeline existant avec des analyses métier spécifiques
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialise l'analyseur avec configuration spécialisée sécheresse"""

        # Configuration par défaut pour l'analyse sécheresse
        drought_config = {
            'target_variable': 'montant_charge_brut',
            'key_risk_factors': [
                'prime', 'engagements', 'risque_rga_gcl',
                'surface_batiment', 'nb_pieces_police', 'type_mur_batiment',
                'annee_construction_batiment', 'weather_precip_mean', 'weather_temp_mean'
            ],
            'drought_specific_vars': [
                'weather_precip_mean', 'weather_temp_mean', 'risque_rga_gcl'
            ],
            'output_dir': './outputs/drought_analysis',
            'generate_advanced_reports': True,
            'stats_params': {
                'outlier_strategy': 'clip',
                'winsorize_percentiles': (0.01, 0.99)
            }
        }

        # Merge avec la config utilisateur
        if config:
            drought_config.update(config)

        super().__init__(drought_config)

        # Extensions spécifiques sécheresse
        self.drought_insights: Dict[str, Any] = {}
        self.climate_analysis_results: Dict[str, Any] = {}
        self.vulnerability_scores: Dict[str, Any] = {}
        self.risk_segmentation: Dict[str, Any] = {}

        # Ces attributs sont souvent présents dans le pipeline parent :
        # self.raw_data, self.processed_data, self.stats_manager, self.synthesis_manager

        self.config.setdefault("load_params", {})
        self.config.setdefault("filter_params", {"passthrough": True})
        self.config.setdefault("preprocess_params", {})
        self.config.setdefault("visualizer_params", {"monthly_agg": "sum", "monthly_min_count": 3})
        self.config.setdefault("date_column", "date_sinistre")
        self.config.setdefault("target_variable", "montant_charge_brut")

        logger.info("Analyseur de sinistres sécheresse initialisé") 

    def _json_default(obj):
        
        # NumPy -> Python
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        # Pandas
        if isinstance(obj, pd.Series):    return obj.tolist()
        if isinstance(obj, pd.DataFrame): return obj.to_dict(orient="list")
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, pd.Period):    return str(obj)
        # Datetime natif
        if isinstance(obj, (datetime, date)): return obj.isoformat()
        # fallback
        return str(obj)   

    def _resolve_params(self, key_in_self_config: str, passed) -> dict:
        """
        Fusionne dans cet ordre:
        1) base = self.config.get(key_in_self_config, {})
        2) passed:
        - None  -> rien
        - dict  -> merge (écrase base)
        - str/Path -> charge YAML/JSON puis merge
        - callable(base_dict)-> dict -> merge
        """
        base = dict(self.config.get(key_in_self_config, {}) or {})
        if passed is None:
            return base
        if isinstance(passed, dict):
            return {**base, **passed}
        if isinstance(passed, (str, os.PathLike)):
            loaded = _load_config_file(passed)
            return {**base, **(loaded or {})}
        if callable(passed):
            produced = passed(dict(base)) or {}
            if not isinstance(produced, dict):
                raise TypeError("callable config must return a dict")
            return {**base, **produced}
        raise TypeError(f"Unsupported param type for {key_in_self_config}: {type(passed)}")

    # ----------------------------- CLIMAT ---------------------------------

    def analyze_climate_impact(self,
                               precipitation_threshold: float = 20.0,
                               temperature_threshold: float = 25.0) -> Dict[str, Any]:
        """Analyse l'impact des conditions climatiques sur les sinistres."""
        logger.info("🌡️ Analyse de l'impact climatique sur les sinistres")

        if getattr(self, "processed_data", None) is None:
            raise ValueError("Données non préprocessées. Exécuter preprocess_data() d'abord.")

        df = self.processed_data.copy()
        target_var = self.config['target_variable']

        climate_results: Dict[str, Any] = {}

        if 'weather_precip_mean' in df.columns:
            climate_results['precipitation_analysis'] = self._analyze_precipitation_impact(
                df, target_var, precipitation_threshold
            )

        if 'weather_temp_mean' in df.columns:
            climate_results['temperature_analysis'] = self._analyze_temperature_impact(
                df, target_var, temperature_threshold
            )

        if {'weather_precip_mean', 'weather_temp_mean'}.issubset(df.columns):
            climate_results['drought_conditions'] = self._analyze_drought_conditions(
                df, target_var, precipitation_threshold, temperature_threshold
            )

        climate_results['seasonal_climate'] = self._analyze_seasonal_climate_impact(df, target_var)
        self.climate_analysis_results = climate_results
        return climate_results

    def _analyze_precipitation_impact(self, df: pd.DataFrame, target_var: str,
                                      threshold: float) -> Dict[str, Any]:
        precip_data = df[['weather_precip_mean', target_var]].copy().dropna()
        precip_data[target_var] = pd.to_numeric(precip_data[target_var], errors='coerce')
        precip_data = precip_data[precip_data[target_var] > 0]
        if precip_data.empty:
            return {'error': 'Données précipitations insuffisantes'}

        precip_data['precip_category'] = pd.cut(
            precip_data['weather_precip_mean'],
            bins=[-np.inf, threshold, threshold * 2, np.inf],
            labels=['Très sec', 'Sec', 'Normal+']
        )

        precip_stats = precip_data.groupby('precip_category')[target_var].agg(
            count='count', mean='mean', median='median', std='std',
            p90=lambda x: x.quantile(0.9), p95=lambda x: x.quantile(0.95)
        ).round(2)

        groups = [g[target_var].values for _, g in precip_data.groupby('precip_category')]
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            h_stat, p_value = stats.kruskal(*groups)
        else:
            h_stat, p_value = np.nan, np.nan

        correlation = precip_data['weather_precip_mean'].corr(precip_data[target_var])

        return {
            'statistics_by_category': precip_stats.to_dict(),
            'correlation': float(correlation),
            'kruskal_test': {'statistic': float(h_stat), 'p_value': float(p_value)},
            'interpretation': self._interpret_precipitation_results(precip_stats, correlation, p_value)
        }

    def _analyze_temperature_impact(self, df: pd.DataFrame, target_var: str,
                                    threshold: float) -> Dict[str, Any]:
        temp_data = df[['weather_temp_mean', target_var]].copy().dropna()
        temp_data[target_var] = pd.to_numeric(temp_data[target_var], errors='coerce')
        temp_data = temp_data[temp_data[target_var] > 0]
        if temp_data.empty:
            return {'error': 'Données température insuffisantes'}

        temp_data['temp_category'] = pd.cut(
            temp_data['weather_temp_mean'],
            bins=[-np.inf, threshold - 5, threshold, np.inf],
            labels=['Fraîche', 'Modérée', 'Chaude']
        )

        temp_stats = temp_data.groupby('temp_category')[target_var].agg(
            count='count', mean='mean', median='median', std='std',
            p90=lambda x: x.quantile(0.9), p95=lambda x: x.quantile(0.95)
        ).round(2)

        groups = [g[target_var].values for _, g in temp_data.groupby('temp_category')]
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            h_stat, p_value = stats.kruskal(*groups)
        else:
            h_stat, p_value = np.nan, np.nan

        correlation = temp_data['weather_temp_mean'].corr(temp_data[target_var])

        return {
            'statistics_by_category': temp_stats.to_dict(),
            'correlation': float(correlation),
            'kruskal_test': {'statistic': float(h_stat), 'p_value': float(p_value)},
            'interpretation': self._interpret_temperature_results(temp_stats, correlation, p_value)
        }

    def _analyze_drought_conditions(self, df: pd.DataFrame, target_var: str,
                                    precip_threshold: float, temp_threshold: float) -> Dict[str, Any]:
        drought_data = df[['weather_precip_mean', 'weather_temp_mean', target_var]].copy().dropna()
        drought_data[target_var] = pd.to_numeric(drought_data[target_var], errors='coerce')
        drought_data = drought_data[drought_data[target_var] > 0]
        if drought_data.empty:
            return {'error': 'Données climatiques insuffisantes'}

        drought_data['is_drought'] = (
            (drought_data['weather_precip_mean'] < precip_threshold) &
            (drought_data['weather_temp_mean'] > temp_threshold)
        )

        precip_norm = (precip_threshold - drought_data['weather_precip_mean']) / max(precip_threshold, 1e-6)
        temp_norm = (drought_data['weather_temp_mean'] - temp_threshold) / max(temp_threshold, 1e-6)
        drought_data['drought_intensity'] = np.maximum(0, precip_norm + temp_norm)

        drought_data['drought_category'] = pd.cut(
            drought_data['drought_intensity'],
            bins=[-np.inf, 0, 0.5, 1, np.inf],
            labels=['Pas de sécheresse', 'Sécheresse légère', 'Sécheresse modérée', 'Sécheresse sévère']
        )

        drought_stats = drought_data.groupby('drought_category')[target_var].agg(
            count='count', mean='mean', median='median', std='std',
            p90=lambda x: x.quantile(0.9), p95=lambda x: x.quantile(0.95)
        ).round(2)

        drought_vs_normal = drought_data.groupby('is_drought')[target_var].agg(
            count='count', mean='mean', median='median', std='std'
        ).round(2)

        dg = drought_data[drought_data['is_drought']][target_var]
        ng = drought_data[~drought_data['is_drought']][target_var]
        if len(dg) >= 3 and len(ng) >= 3:
            u_stat, p_value = stats.mannwhitneyu(dg, ng, alternative='two-sided')
        else:
            u_stat, p_value = np.nan, np.nan

        return {
            'drought_conditions_stats': drought_stats.to_dict(),
            'drought_vs_normal': drought_vs_normal.to_dict(),
            'mann_whitney_test': {'statistic': float(u_stat), 'p_value': float(p_value)},
            'drought_percentage': float((drought_data['is_drought'].sum() / len(drought_data)) * 100),
            'interpretation': self._interpret_drought_results(drought_stats, p_value)
        }

    def _analyze_seasonal_climate_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        """Climat moyen par mois et lien avec le target."""
        out = {}
        date_col = None
        for c in ['date_exercice', 'date_sinistre', 'date']:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            return out

        d = df[[date_col, target_var, 'weather_precip_mean', 'weather_temp_mean']].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
        d[target_var] = pd.to_numeric(d[target_var], errors='coerce')
        d = d.dropna(subset=[date_col, target_var])
        d['month'] = d[date_col].dt.month

        monthly = d.groupby('month').agg(
            montant_mean=(target_var, 'mean'),
            precip_mean=('weather_precip_mean', 'mean'),
            temp_mean=('weather_temp_mean', 'mean'),
            n=(target_var, 'size')
        )
        out['by_month'] = monthly.round(2).to_dict()

        # corrélation entre climat moyen mensuel et montant moyen mensuel
        if monthly[['precip_mean']].notna().any().all():
            out['corr_montant_precip'] = float(monthly['montant_mean'].corr(monthly['precip_mean']))
        if monthly[['temp_mean']].notna().any().all():
            out['corr_montant_temp'] = float(monthly['montant_mean'].corr(monthly['temp_mean']))
        return out

    # --------------------------- VULNÉRABILITÉ ----------------------------

    def analyze_building_vulnerability(self) -> Dict[str, Any]:
        """Analyse la vulnérabilité des bâtiments aux sinistres sécheresse."""
        logger.info("🏗️ Analyse de la vulnérabilité des bâtiments")

        if getattr(self, "processed_data", None) is None:
            raise ValueError("Données non préprocessées. Exécuter preprocess_data() d'abord.")

        df = self.processed_data.copy()
        target_var = self.config['target_variable']

        vulnerability_results: Dict[str, Any] = {}
        vulnerability_results['age_analysis'] = self._analyze_building_age_impact(df, target_var)
        vulnerability_results['surface_analysis'] = self._analyze_surface_impact(df, target_var)

        if 'type_mur_batiment' in df.columns:
            vulnerability_results['wall_type_analysis'] = self._analyze_wall_type_impact(df, target_var)

        if 'nb_pieces_police' in df.columns:
            vulnerability_results['rooms_analysis'] = self._analyze_rooms_impact(df, target_var)

        vulnerability_results['vulnerability_score'] = self._compute_vulnerability_score(df, target_var)
        self.vulnerability_scores = vulnerability_results
        return vulnerability_results

    def _analyze_building_age_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        age_col = None
        if 'age_batiment' in df.columns:
            age_col = 'age_batiment'
        elif 'annee_construction_batiment' in df.columns:
            current_year = datetime.now().year
            df['age_batiment_calculated'] = current_year - pd.to_numeric(df['annee_construction_batiment'], errors='coerce')
            age_col = 'age_batiment_calculated'
        if age_col is None:
            return {'error': 'Données âge bâtiment non disponibles'}

        age_data = df[[age_col, target_var]].copy().dropna()
        age_data[target_var] = pd.to_numeric(age_data[target_var], errors='coerce')
        age_data = age_data[(age_data[target_var] > 0) & (age_data[age_col] >= 0) & (age_data[age_col] <= 200)]
        if age_data.empty:
            return {'error': 'Données âge insuffisantes'}

        age_data['age_category'] = pd.cut(
            age_data[age_col],
            bins=[0, 20, 40, 60, 100, np.inf],
            labels=['Très récent (0-20)', 'Récent (20-40)', 'Moyen (40-60)', 'Ancien (60-100)', 'Très ancien (100+)']
        )

        age_stats = age_data.groupby('age_category')[target_var].agg(
            count='count', mean='mean', median='median', std='std', p90=lambda x: x.quantile(0.9)
        ).round(2)
        correlation = age_data[age_col].corr(age_data[target_var])
        rho, p_value = stats.spearmanr(age_data[age_col], age_data[target_var])

        return {
            'age_categories_stats': age_stats.to_dict(),
            'correlation': float(correlation) if pd.notna(correlation) else None,
            'spearman_trend': {'rho': float(rho) if pd.notna(rho) else None,
                               'p_value': float(p_value) if pd.notna(p_value) else None},
            'interpretation': self._interpret_age_results(age_stats, correlation, rho, p_value)
        }

    def _analyze_surface_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        if 'surface_batiment' not in df.columns:
            return {'error': 'surface_batiment indisponible'}
        d = df[['surface_batiment', target_var]].copy().dropna()
        d['surface_batiment'] = pd.to_numeric(d['surface_batiment'], errors='coerce')
        d[target_var] = pd.to_numeric(d[target_var], errors='coerce')
        d = d[(d['surface_batiment'] > 0) & (d[target_var] > 0)]
        if d.empty:
            return {'error': 'Données surface insuffisantes'}

        d['surf_quartile'] = pd.qcut(d['surface_batiment'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        stats_quart = d.groupby('surf_quartile')[target_var].agg(
            count='count', mean='mean', median='median', std='std'
        ).round(2)
        rho, p = stats.spearmanr(d['surface_batiment'], d[target_var])
        return {
            'quartiles_stats': stats_quart.to_dict(),
            'spearman': {'rho': float(rho) if pd.notna(rho) else None,
                         'p_value': float(p) if pd.notna(p) else None}
        }

    def _analyze_wall_type_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        d = df[['type_mur_batiment', target_var]].copy().dropna()
        d[target_var] = pd.to_numeric(d[target_var], errors='coerce')
        d = d[d[target_var] > 0]
        if d.empty:
            return {'error': 'Données type_mur insuffisantes'}

        stats_by_type = d.groupby('type_mur_batiment')[target_var].agg(
            count='count', mean='mean', median='median', std='std'
        ).round(2)
        groups = [g[target_var].values for _, g in d.groupby('type_mur_batiment')]
        if len(groups) >= 2 and all(len(g) >= 3 for g in groups):
            H, p = stats.kruskal(*groups, nan_policy='omit')
        else:
            H, p = np.nan, np.nan
        return {
            'by_wall_type': stats_by_type.to_dict(),
            'kruskal': {'H': float(H) if pd.notna(H) else None,
                        'p_value': float(p) if pd.notna(p) else None}
        }

    def _analyze_rooms_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        col = 'nb_pieces_police'
        d = df[[col, target_var]].copy().dropna()
        d[col] = pd.to_numeric(d[col], errors='coerce')
        d[target_var] = pd.to_numeric(d[target_var], errors='coerce')
        d = d[(d[col] >= 0) & (d[col] <= 20) & (d[target_var] > 0)]
        if d.empty:
            return {'error': 'Données pièces insuffisantes'}

        stats_by_room = d.groupby(col)[target_var].agg(
            count='count', mean='mean', median='median', std='std'
        ).round(2)
        rho, p = stats.spearmanr(d[col], d[target_var])
        return {
            'by_rooms': stats_by_room.to_dict(),
            'spearman': {'rho': float(rho) if pd.notna(rho) else None,
                         'p_value': float(p) if pd.notna(p) else None}
        }

    def _compute_vulnerability_score(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        """Score composite basé sur l’âge, la surface et (optionnel) type de mur."""
        base = self._compute_building_vulnerability_score(df)
        add_parts = [base]

        if 'type_mur_batiment' in df.columns:
            # Barème simple (modifiable)
            weights = {
                'parpaing': 0.5, 'brique': 0.6, 'pierre': 0.7,
                'bois': 0.55, 'béton': 0.45
            }
            wall = df['type_mur_batiment'].astype(str).str.lower().map(weights).fillna(0.5)
            add_parts.append(wall)

        score = pd.concat(add_parts, axis=1).mean(axis=1).clip(0, 1)
        return {
            'per_row': score,
            'summary': score.describe().to_dict()
        }

    # ---------------------------- EXPOSITION ------------------------------

    def analyze_risk_exposure(self) -> Dict[str, Any]:
        """Analyse de l'exposition au risque basée sur prime, engagements, RGA."""
        logger.info("💰 Analyse de l'exposition au risque")

        if getattr(self, "processed_data", None) is None:
            raise ValueError("Données non préprocessées. Exécuter preprocess_data() d'abord.")

        df = self.processed_data.copy()
        target_var = self.config['target_variable']

        exposure_results: Dict[str, Any] = {}

        if 'prime' in df.columns:
            exposure_results['prime_analysis'] = self._analyze_premium_impact(df, target_var)

        if 'engagements' in df.columns:
            exposure_results['engagement_analysis'] = self._analyze_engagement_impact(df, target_var)

        if 'risque_rga_gcl' in df.columns:
            exposure_results['rga_detailed_analysis'] = self._analyze_rga_detailed(df, target_var)

        exposure_results['combined_exposure'] = self._analyze_combined_exposure(df, target_var)
        return exposure_results

    def _analyze_premium_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        premium_data = df[['prime', target_var]].copy().dropna()
        premium_data[target_var] = pd.to_numeric(premium_data[target_var], errors='coerce')
        premium_data['prime'] = pd.to_numeric(premium_data['prime'], errors='coerce')
        premium_data = premium_data[(premium_data[target_var] > 0) & (premium_data['prime'] > 0)]
        if premium_data.empty:
            return {'error': 'Données prime insuffisantes'}

        premium_data['prime_quartile'] = pd.qcut(
            premium_data['prime'], q=4,
            labels=['Q1 (Faible)', 'Q2 (Moyen-)', 'Q3 (Moyen+)', 'Q4 (Élevé)']
        )

        quartile_stats = premium_data.groupby('prime_quartile')[target_var].agg(
            count='count', mean='mean', median='median', std='std', p90=lambda x: x.quantile(0.9)
        ).round(2)

        correlation = premium_data['prime'].corr(premium_data[target_var])
        premium_data['loss_ratio'] = premium_data[target_var] / premium_data['prime']
        loss_ratio_stats = premium_data['loss_ratio'].describe()

        return {
            'quartile_stats': quartile_stats.to_dict(),
            'correlation': float(correlation) if pd.notna(correlation) else None,
            'loss_ratio_stats': loss_ratio_stats.to_dict(),
            'interpretation': f"Corrélation prime-sinistre: {correlation:.3f}" if pd.notna(correlation) else "Corrélation non calculable"
        }

    def _analyze_engagement_impact(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        data = df[['engagements', target_var]].copy().dropna()
        data['engagements'] = pd.to_numeric(data['engagements'], errors='coerce')
        data[target_var] = pd.to_numeric(data[target_var], errors='coerce')
        data = data[(data['engagements'] > 0) & (data[target_var] > 0)]
        if data.empty:
            return {'error': 'Données engagements insuffisantes'}

        data['eng_quartile'] = pd.qcut(data['engagements'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        stats_q = data.groupby('eng_quartile')[target_var].agg(count='count', mean='mean', median='median', std='std').round(2)
        corr = data['engagements'].corr(data[target_var])
        return {
            'quartile_stats': stats_q.to_dict(),
            'correlation': float(corr) if pd.notna(corr) else None
        }

    def _analyze_rga_detailed(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        d = df[['risque_rga_gcl', target_var]].copy().dropna()
        d['risque_rga_gcl'] = d['risque_rga_gcl'].astype(str).str.lower()
        order = ['faible', 'moyen', 'fort', 'maximal']
        mapping = {
            'faible': 'faible', 'bas': 'faible',
            'moyen': 'moyen', 'modere': 'moyen', 'modéré': 'moyen',
            'fort': 'fort', 'eleve': 'fort', 'élevé': 'fort',
            'tres eleve': 'maximal', 'très élevé': 'maximal', 'maximal': 'maximal'
        }
        d['rga'] = d['risque_rga_gcl'].map(mapping)
        d = d.dropna(subset=['rga'])
        d[target_var] = pd.to_numeric(d[target_var], errors='coerce')
        d = d[d[target_var] > 0]
        if d.empty:
            return {'error': 'Données RGA insuffisantes'}

        d['rga'] = pd.Categorical(d['rga'], categories=order, ordered=True)
        stats_rga = d.groupby('rga')[target_var].agg(count='count', mean='mean', median='median', std='std').round(2)
        groups = [g[target_var].values for _, g in d.groupby('rga')]
        if all(len(g) >= 3 for g in groups):
            H, p = stats.kruskal(*groups, nan_policy='omit')
        else:
            H, p = np.nan, np.nan

        codes = d['rga'].cat.codes
        rho, p_s = stats.spearmanr(codes, d[target_var], nan_policy='omit')
        return {
            'by_rga': stats_rga.to_dict(),
            'kruskal': {'H': float(H) if pd.notna(H) else None, 'p_value': float(p) if pd.notna(p) else None},
            'spearman': {'rho': float(rho) if pd.notna(rho) else None, 'p_value': float(p_s) if pd.notna(p_s) else None}
        }

    def _analyze_combined_exposure(self, df: pd.DataFrame, target_var: str) -> Dict[str, Any]:
        score = self._compute_exposure_score(df)
        t = pd.to_numeric(df[target_var], errors='coerce')
        d = pd.DataFrame({'exposure_score': score, 'montant': t}).dropna()
        if d.empty:
            return {'error': 'Insuffisant'}
        d['score_quart'] = pd.qcut(d['exposure_score'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        stats_q = d.groupby('score_quart')['montant'].agg(count='count', mean='mean', median='median', std='std').round(2)
        rho, p = stats.spearmanr(d['exposure_score'], d['montant'])
        return {
            'quartiles_stats': stats_q.to_dict(),
            'spearman': {'rho': float(rho) if pd.notna(rho) else None, 'p_value': float(p) if pd.notna(p) else None}
        }

    # ------------------------- INSIGHTS & VISUS ---------------------------

    def generate_drought_specific_insights(self) -> Dict[str, Any]:
        """Génère des insights spécifiques aux sinistres sécheresse."""
        logger.info("💡 Génération d'insights spécifiques sécheresse")

        insights = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'Sinistres Sécheresse',
            'key_findings': [],
            'recommendations': [],
            'risk_factors_ranking': [],
            'cost_drivers': [],
            'geographical_insights': [],
            'temporal_patterns': []
        }

        if self.climate_analysis_results:
            climate_insights = self._extract_climate_insights()
            insights['key_findings'].extend(climate_insights['findings'])
            insights['recommendations'].extend(climate_insights['recommendations'])

        if self.vulnerability_scores:
            vulnerability_insights = self._extract_vulnerability_insights()
            insights['key_findings'].extend(vulnerability_insights['findings'])
            insights['cost_drivers'].extend(vulnerability_insights['cost_drivers'])

        exposure_insights = self._extract_exposure_insights()
        insights['risk_factors_ranking'] = exposure_insights['ranking']

        temporal_insights = self._analyze_drought_temporal_patterns()
        insights['temporal_patterns'] = temporal_insights

        insights['recommendations'].extend([
            "Renforcer la surveillance des zones à fort risque RGA identifiées",
            "Développer des produits adaptés aux bâtiments anciens en zone sèche",
            "Mettre en place un système d'alerte précoce basé sur les données météo",
            "Optimiser les tarifs selon les scores de vulnérabilité calculés"
        ])

        self.drought_insights = insights
        return insights

    def _extract_climate_insights(self) -> Dict[str, List[str]]:
        findings, recommendations = [], []
        if 'precipitation_analysis' in self.climate_analysis_results:
            corr = self.climate_analysis_results['precipitation_analysis'].get('correlation')
            p = self.climate_analysis_results['precipitation_analysis'].get('kruskal_test', {}).get('p_value', 1)
            if corr is not None and corr < -0.2:
                findings.append(f"Corrélation négative entre précipitations et montants (r={corr:.3f}).")
            if p is not None and p < 0.05:
                recommendations.append("Intégrer un facteur ‘sécheresse’ aux modèles tarifaires.")

        if 'temperature_analysis' in self.climate_analysis_results:
            corr = self.climate_analysis_results['temperature_analysis'].get('correlation')
            if corr is not None and corr > 0.2:
                findings.append(f"Corrélation positive entre chaleur et montants (r={corr:.3f}).")
                recommendations.append("Surveiller les vagues de chaleur pour l’anticipation.")

        if 'drought_conditions' in self.climate_analysis_results:
            pct = self.climate_analysis_results['drought_conditions'].get('drought_percentage')
            if pct is not None:
                findings.append(f"{pct:.1f}% des sinistres en conditions de sécheresse estimée.")
        return {'findings': findings, 'recommendations': recommendations}

    def _extract_vulnerability_insights(self) -> Dict[str, List[str]]:
        findings, cost_drivers = [], []
        age = self.vulnerability_scores.get('age_analysis', {})
        sp = age.get('spearman_trend', {})
        if sp and sp.get('rho') and sp.get('p_value') and abs(sp['rho']) > 0.2 and sp['p_value'] < 0.05:
            if sp['rho'] > 0:
                findings.append(f"Les bâtiments plus anciens coûtent davantage (ρ={sp['rho']:.3f}).")
                cost_drivers.append("Âge du bâtiment")
        return {'findings': findings, 'cost_drivers': cost_drivers}

    def _extract_exposure_insights(self) -> Dict[str, List[str]]:
        ranking = []
        if hasattr(self, 'stats_manager') and self.stats_manager and 'correlations' in self.stats_manager.results:
            corr_df = self.stats_manager.results['correlations']
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                drought_vars = ['weather_precip_mean', 'weather_temp_mean', 'risque_rga_gcl',
                                'age_batiment', 'surface_batiment', 'prime', 'engagements']
                for var in drought_vars:
                    if var in corr_df['variable'].values:
                        corr_val = corr_df[corr_df['variable'] == var]['correlation'].iloc[0]
                        ranking.append(f"{var}: {corr_val:.3f}")
        return {'ranking': ranking}

    def _analyze_drought_temporal_patterns(self) -> List[str]:
        patterns: List[str] = []
        if getattr(self, "processed_data", None) is not None and 'date_sinistre' in self.processed_data.columns:
            df = self.processed_data.copy()
            target_var = self.config['target_variable']
            df['date_sinistre'] = pd.to_datetime(df['date_sinistre'], errors='coerce')
            df = df.dropna(subset=['date_sinistre', target_var])
            if not df.empty:
                df['month'] = df['date_sinistre'].dt.month
                monthly_avg = df.groupby('month')[target_var].mean()
                peaks = monthly_avg.nlargest(3).index.tolist()
                nom = {1:'Jan',2:'Fév',3:'Mar',4:'Avr',5:'Mai',6:'Juin',7:'Juil',8:'Août',9:'Sep',10:'Oct',11:'Nov',12:'Déc'}
                patterns.append("Pics saisonniers: " + ", ".join(nom.get(m, str(m)) for m in peaks))
                df['year'] = df['date_sinistre'].dt.year
                if df['year'].nunique() > 1:
                    yearly = df.groupby('year')[target_var].mean()
                    corr = pd.Series(yearly.index).corr(pd.Series(yearly.values))
                    if pd.notna(corr) and abs(corr) > 0.5:
                        patterns.append(f"Tendance {('haussière' if corr>0 else 'baissière')} (corr={corr:.3f}).")
        return patterns

    # ---------------------- VISUALISATIONS SPÉCIALISÉES ------------------

    def create_advanced_visualizations(self) -> Dict[str, Any]:
        logger.info("📈 Création des visualisations avancées sécheresse")
        if getattr(self, "processed_data", None) is None:
            raise ValueError("Données non préprocessées")

        df = self.processed_data.copy()
        target_var = self.config['target_variable']
        plt.style.use('default')
        sns.set_palette("husl")

        figures: Dict[str, Optional[plt.Figure]] = {}
        if all(c in df.columns for c in ['weather_precip_mean', 'weather_temp_mean']):
            figures['climate_heatmap'] = self._create_climate_heatmap(df, target_var)
        figures['vulnerability_dashboard'] = self._create_vulnerability_dashboard(df, target_var)
        if 'date_sinistre' in df.columns:
            figures['temporal_analysis'] = self._create_temporal_analysis_plot(df, target_var)
        figures['risk_matrix'] = self._create_risk_matrix(df, target_var)
        figures['risk_distributions'] = self._create_risk_profile_distributions(df, target_var)

        # Publier dans le générateur de rapports avancés si dispo
        if hasattr(self, 'synthesis_manager') and hasattr(self.synthesis_manager, 'advanced_generator'):
            for name, fig in figures.items():
                if fig is not None:
                    self.synthesis_manager.advanced_generator.add_figure(
                        name, fig, f"Analyse spécialisée sécheresse: {name.replace('_', ' ').title()}"
                    )
        return figures

    def _create_climate_heatmap(self, df: pd.DataFrame, target_var: str) -> Optional[plt.Figure]:
        climate_data = df[['weather_precip_mean', 'weather_temp_mean', target_var]].copy().dropna()
        climate_data[target_var] = pd.to_numeric(climate_data[target_var], errors='coerce')
        climate_data = climate_data.dropna()
        if climate_data.empty:
            return None

        climate_data['precip_bin'] = pd.cut(climate_data['weather_precip_mean'],
                                            bins=5, labels=['Très sec', 'Sec', 'Modéré', 'Humide', 'Très humide'])
        climate_data['temp_bin'] = pd.cut(climate_data['weather_temp_mean'],
                                          bins=5, labels=['Froid', 'Frais', 'Modéré', 'Chaud', 'Très chaud'])
        pivot_data = climate_data.groupby(['temp_bin', 'precip_bin'])[target_var].mean().unstack()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Montant moyen (€)'})
        ax.set_title('Heatmap: Montant des sinistres selon les conditions climatiques')
        ax.set_xlabel('Niveau de précipitations')
        ax.set_ylabel('Niveau de température')
        fig.tight_layout()
        return fig

    def _create_vulnerability_dashboard(self, df: pd.DataFrame, target_var: str) -> Optional[plt.Figure]:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dashboard Vulnérabilité des Bâtiments', fontsize=16, y=0.98)

        # 1. Montants par décennie de construction
        if 'annee_construction_batiment' in df.columns:
            age_data = df[['annee_construction_batiment', target_var]].copy().dropna()
            age_data[target_var] = pd.to_numeric(age_data[target_var], errors='coerce')
            age_data['annee_construction_batiment'] = pd.to_numeric(age_data['annee_construction_batiment'], errors='coerce')
            age_data = age_data[(age_data[target_var] > 0) & (age_data['annee_construction_batiment'] > 1800)]
            if not age_data.empty:
                age_data['decade'] = (age_data['annee_construction_batiment'] // 10) * 10
                decade_stats = age_data.groupby('decade')[target_var].mean()
                axes[0, 0].plot(decade_stats.index, decade_stats.values, marker='o', linewidth=2)
                axes[0, 0].set_title('Montant moyen par décennie de construction')
                axes[0, 0].set_xlabel('Décennie de construction')
                axes[0, 0].set_ylabel('Montant moyen (€)')
                axes[0, 0].grid(True, alpha=0.3)

        # 2. Montant vs surface
        if 'surface_batiment' in df.columns:
            surf_data = df[['surface_batiment', target_var]].copy().dropna()
            surf_data[target_var] = pd.to_numeric(surf_data[target_var], errors='coerce')
            surf_data['surface_batiment'] = pd.to_numeric(surf_data['surface_batiment'], errors='coerce')
            surf_data = surf_data[(surf_data[target_var] > 0) & (surf_data['surface_batiment'] > 0)]
            if not surf_data.empty:
                axes[0, 1].scatter(surf_data['surface_batiment'], surf_data[target_var], alpha=0.5)
                if len(surf_data) > 10:
                    z = np.polyfit(surf_data['surface_batiment'], surf_data[target_var], 1)
                    p = np.poly1d(z)
                    axes[0, 1].plot(surf_data['surface_batiment'], p(surf_data['surface_batiment']),
                                    "r--", alpha=0.8, linewidth=2)
                axes[0, 1].set_title('Montant vs Surface du bâtiment')
                axes[0, 1].set_xlabel('Surface (m²)')
                axes[0, 1].set_ylabel('Montant (€)')
                axes[0, 1].grid(True, alpha=0.3)

        # 3. Type de mur
        if 'type_mur_batiment' in df.columns:
            wall_data = df[['type_mur_batiment', target_var]].copy().dropna()
            wall_data[target_var] = pd.to_numeric(wall_data[target_var], errors='coerce')
            wall_data = wall_data[wall_data[target_var] > 0]
            if not wall_data.empty:
                wall_stats = wall_data.groupby('type_mur_batiment')[target_var].agg(['mean', 'count'])
                wall_stats = wall_stats[wall_stats['count'] >= 10]
                if not wall_stats.empty:
                    axes[1, 0].bar(range(len(wall_stats)), wall_stats['mean'])
                    axes[1, 0].set_xticks(range(len(wall_stats)))
                    axes[1, 0].set_xticklabels(wall_stats.index, rotation=45, ha='right')
                    axes[1, 0].set_title('Montant moyen par type de mur')
                    axes[1, 0].set_ylabel('Montant moyen (€)')

        # 4. Score de vulnérabilité
        vuln_score = self._compute_building_vulnerability_score(df)
        target_data = pd.to_numeric(df[target_var], errors='coerce')
        score_bins = pd.cut(vuln_score, bins=5, labels=['Très faible', 'Faible', 'Modéré', 'Élevé', 'Très élevé'])
        score_df = pd.DataFrame({'score_bin': score_bins, target_var: target_data}).dropna()
        if not score_df.empty:
            box_data = [score_df[score_df['score_bin'] == cat][target_var].values
                        for cat in score_df['score_bin'].cat.categories]
            non_empty = [a for a in box_data if len(a) > 0]
            if non_empty:
                axes[1, 1].boxplot(non_empty, labels=score_df['score_bin'].cat.categories[:len(non_empty)])
                axes[1, 1].set_title('Distribution des montants par score de vulnérabilité')
                axes[1, 1].set_ylabel('Montant (€)')
                axes[1, 1].tick_params(axis='x', rotation=45)

        fig.tight_layout()
        return fig

    def _create_temporal_analysis_plot(self, df: pd.DataFrame, target_var: str) -> Optional[plt.Figure]:
        temp_data = df[['date_sinistre', target_var]].copy()
        temp_data['date_sinistre'] = pd.to_datetime(temp_data['date_sinistre'], errors='coerce')
        temp_data[target_var] = pd.to_numeric(temp_data[target_var], errors='coerce')
        temp_data = temp_data.dropna()
        if temp_data.empty:
            return None

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Analyse Temporelle des Sinistres Sécheresse', fontsize=16)

        temp_data['year_month'] = temp_data['date_sinistre'].dt.to_period('M')
        monthly_data = temp_data.groupby('year_month')[target_var].agg(['mean', 'count', 'sum'])

        axes[0].plot(monthly_data.index.to_timestamp(), monthly_data['mean'], marker='o', linewidth=2, label='Montant moyen')
        axes[0].set_title('Évolution du montant moyen par mois')
        axes[0].set_ylabel('Montant moyen (€)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].bar(monthly_data.index.to_timestamp(), monthly_data['count'], alpha=0.7, color='orange')
        axes[1].set_title('Nombre de sinistres par mois')
        axes[1].set_ylabel('Nombre de sinistres')
        axes[1].grid(True, alpha=0.3)

        temp_data['month'] = temp_data['date_sinistre'].dt.month
        seasonal_data = temp_data.groupby('month')[target_var].agg(['mean', 'count'])

        ax3_twin = axes[2].twinx()
        axes[2].plot(seasonal_data.index, seasonal_data['mean'], marker='o', linewidth=2, label='Montant moyen')
        ax3_twin.bar(seasonal_data.index, seasonal_data['count'], alpha=0.5, label='Nombre de sinistres')

        axes[2].set_title('Pattern saisonnier: Montant vs Fréquence')
        axes[2].set_xlabel('Mois')
        axes[2].set_ylabel('Montant moyen (€)')
        ax3_twin.set_ylabel('Nombre de sinistres')
        axes[2].set_xticks(range(1, 13))
        axes[2].grid(True, alpha=0.3)

        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        fig.tight_layout()
        return fig

    def _create_risk_matrix(self, df: pd.DataFrame, target_var: str) -> Optional[plt.Figure]:
        exposure_score = self._compute_exposure_score(df)
        vulnerability_score = self._compute_building_vulnerability_score(df)
        target_data = pd.to_numeric(df[target_var], errors='coerce')

        risk_df = pd.DataFrame({
            'exposure': exposure_score,
            'vulnerability': vulnerability_score,
            'montant': target_data
        }).dropna()
        if risk_df.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(risk_df['exposure'], risk_df['vulnerability'],
                             c=risk_df['montant'], cmap='YlOrRd',
                             s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)

        ax.text(0.25, 0.75, 'Faible expos.\nHaute vuln.', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0.75, 0.75, 'Haute expos.\nHaute vuln.', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        ax.text(0.25, 0.25, 'Faible expos.\nFaible vuln.', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        ax.text(0.75, 0.25, 'Haute expos.\nFaible vuln.', ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))

        ax.set_xlabel('Score d\'exposition au risque')
        ax.set_ylabel('Score de vulnérabilité du bâtiment')
        ax.set_title('Matrice de Risque: Exposition vs Vulnérabilité')
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Montant du sinistre (€)')

        fig.tight_layout()
        return fig

    def _compute_exposure_score(self, df: pd.DataFrame) -> pd.Series:
        score_components = []
        if 'prime' in df.columns:
            prime = pd.to_numeric(df['prime'], errors='coerce')
            if prime.notna().any():
                prime = prime.clip(prime.quantile(0.01), prime.quantile(0.99))
                score_components.append((prime - prime.min()) / (prime.max() - prime.min()))
        if 'engagements' in df.columns:
            eng = pd.to_numeric(df['engagements'], errors='coerce')
            if eng.notna().any():
                eng = eng.clip(eng.quantile(0.01), eng.quantile(0.99))
                score_components.append((eng - eng.min()) / (eng.max() - eng.min()))
        if 'risque_rga_gcl' in df.columns:
            rga_mapping = {'faible': 0.2, 'moyen': 0.5, 'fort': 0.8, 'maximal': 1.0}
            rga_score = df['risque_rga_gcl'].astype(str).str.lower().map(rga_mapping)
            score_components.append(rga_score.fillna(0.5))
        if score_components:
            exposure_score = pd.concat(score_components, axis=1).mean(axis=1)
        else:
            exposure_score = pd.Series(0.5, index=df.index)
        return exposure_score.clip(0, 1)

    def _create_risk_profile_distributions(self, df: pd.DataFrame, target_var: str) -> Optional[plt.Figure]:
        exposure_score = self._compute_exposure_score(df)
        vulnerability_score = self._compute_building_vulnerability_score(df)
        target_data = pd.to_numeric(df[target_var], errors='coerce')

        profiles = []
        for e, v in zip(exposure_score.fillna(0.5), vulnerability_score.fillna(0.5)):
            if e < 0.4 and v < 0.4:
                profiles.append('Faible risque')
            elif e > 0.7 and v > 0.7:
                profiles.append('Très haut risque')
            elif e > 0.6 or v > 0.6:
                profiles.append('Haut risque')
            else:
                profiles.append('Risque modéré')

        profile_df = pd.DataFrame({'risk_profile': profiles, 'montant': target_data}).dropna()
        if profile_df.empty:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analyse des Distributions par Profil de Risque', fontsize=16)

        order = ['Faible risque', 'Risque modéré', 'Haut risque', 'Très haut risque']
        avail = [p for p in order if p in profile_df['risk_profile'].unique()]
        if avail:
            sns.boxplot(data=profile_df, x='risk_profile', y='montant', order=avail, ax=axes[0, 0])
            axes[0, 0].set_title('Distribution des montants par profil de risque')
            axes[0, 0].tick_params(axis='x', rotation=45)

        colors = ['green', 'orange', 'red', 'darkred']
        for i, profile in enumerate(avail[:4]):
            data = profile_df[profile_df['risk_profile'] == profile]['montant']
            if len(data) > 0:
                axes[0, 1].hist(data, alpha=0.6, label=profile, color=colors[i % len(colors)], bins=20)
        axes[0, 1].set_title('Histogrammes superposés par profil')
        axes[0, 1].legend()

        profile_stats = profile_df.groupby('risk_profile')['montant'].agg(count='count', mean='mean', median='median', std='std').round(0)
        axes[1, 0].axis('tight'); axes[1, 0].axis('off')
        table_data = [[p, f"{int(r['count'])}", f"{r['mean']:.0f}€", f"{r['median']:.0f}€", f"{r['std']:.0f}€"]
                      for p, r in profile_stats.iterrows()]
        table = axes[1, 0].table(cellText=table_data,
                                 colLabels=['Profil', 'N', 'Moyenne', 'Médiane', 'Écart-type'],
                                 cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.5)
        axes[1, 0].set_title('Statistiques descriptives par profil')

        counts = profile_df['risk_profile'].value_counts()
        axes[1, 1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors[:len(counts)])
        axes[1, 1].set_title('Répartition des sinistres par profil de risque')

        fig.tight_layout()
        return fig

    # -------------------------- SEGMENTATION ------------------------------

    def create_risk_segmentation(self) -> Dict[str, Any]:
        """Segmentation des risques basée sur les scores calculés."""
        logger.info("📊 Création de la segmentation des risques")

        if getattr(self, "processed_data", None) is None:
            raise ValueError("Données non préprocessées. Exécuter preprocess_data() d'abord.")

        df = self.processed_data.copy()
        target_var = self.config['target_variable']

        seg_vars = []
        if {'weather_precip_mean', 'weather_temp_mean'}.issubset(df.columns):
            df['climate_risk_score'] = self._compute_climate_risk_score(df)
            seg_vars.append('climate_risk_score')
        if {'annee_construction_batiment', 'age_batiment'}.intersection(df.columns):
            df['building_vulnerability_score'] = self._compute_building_vulnerability_score(df)
            seg_vars.append('building_vulnerability_score')
        if {'prime', 'engagements', 'risque_rga_gcl'}.intersection(df.columns):
            df['exposure_score'] = self._compute_exposure_score(df)
            seg_vars.append('exposure_score')

        if len(seg_vars) < 2:
            self.risk_segmentation = {'error': 'Variables insuffisantes pour la segmentation'}
            return self.risk_segmentation

        X = df[seg_vars].dropna()
        if len(X) < 10:
            self.risk_segmentation = {'error': 'Échantillon insuffisant pour segmenter'}
            return self.risk_segmentation

        k = min(4, max(2, X.shape[0] // 100))  # 2 à 4 clusters selon taille
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        seg_df = pd.DataFrame(X, copy=True)
        seg_df['segment'] = labels
        seg_df['montant'] = pd.to_numeric(df.loc[seg_df.index, target_var], errors='coerce')

        seg_stats = seg_df.groupby('segment')['montant'].agg(count='count', mean='mean', median='median', std='std').round(2)
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=seg_vars)

        self.risk_segmentation = {'segments': seg_stats.to_dict(), 'centers': centers.to_dict(orient='index')}
        return self.risk_segmentation

    def _compute_climate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        if 'weather_precip_mean' in df.columns:
            precip = pd.to_numeric(df['weather_precip_mean'], errors='coerce')
            pr_norm = 1 - (precip - precip.min()) / max(precip.max() - precip.min(), 1e-9)
        else:
            pr_norm = pd.Series(0.5, index=df.index)
        if 'weather_temp_mean' in df.columns:
            temp = pd.to_numeric(df['weather_temp_mean'], errors='coerce')
            t_norm = (temp - temp.min()) / max(temp.max() - temp.min(), 1e-9)
        else:
            t_norm = pd.Series(0.5, index=df.index)
        score = (pr_norm * 0.5 + t_norm * 0.5).fillna(0.5)
        return score.clip(0, 1)

    def _compute_building_vulnerability_score(self, df: pd.DataFrame) -> pd.Series:
        parts = []
        if 'age_batiment' in df.columns:
            age = pd.to_numeric(df['age_batiment'], errors='coerce')
        elif 'annee_construction_batiment' in df.columns:
            current_year = datetime.now().year
            age = current_year - pd.to_numeric(df['annee_construction_batiment'], errors='coerce')
        else:
            age = None
        if age is not None:
            age = age.clip(0, 150)
            parts.append((age / 150).fillna(0.5))
        if 'surface_batiment' in df.columns:
            surface = pd.to_numeric(df['surface_batiment'], errors='coerce')
            surface = surface.clip(0, surface.quantile(0.99))
            parts.append(((surface - surface.min()) / max(surface.max() - surface.min(), 1e-9)).fillna(0.5))
        if parts:
            v = pd.concat(parts, axis=1).mean(axis=1)
        else:
            v = pd.Series(0.5, index=df.index)
        return v.clip(0, 1)

    # -------------------------- INTERPRÉTATIONS ---------------------------

    def _interpret_precipitation_results(self, stats_df: pd.DataFrame, corr: float, p: float) -> str:
        msg = "Précipitations: "
        if pd.notna(corr):
            msg += f"corrélation {('négative' if corr<0 else 'positive')} r={corr:.3f}. "
        if pd.notna(p):
            msg += f"Différences entre catégories {'significatives' if p<0.05 else 'non significatives'} (Kruskal p={p:.3f})."
        return msg

    def _interpret_temperature_results(self, stats_df: pd.DataFrame, corr: float, p: float) -> str:
        msg = "Température: "
        if pd.notna(corr):
            msg += f"corrélation {('positive' if corr>0 else 'négative')} r={corr:.3f}. "
        if pd.notna(p):
            msg += f"Effets par catégories {'significatifs' if p<0.05 else 'non significatifs'} (Kruskal p={p:.3f})."
        return msg

    def _interpret_drought_results(self, stats_df: pd.DataFrame, p: float) -> str:
        return f"Sécheresse: effet {'significatif' if (pd.notna(p) and p<0.05) else 'non significatif'} (Mann-Whitney p={p:.3f})."

    def _interpret_age_results(self, stats_df: pd.DataFrame, corr: float, rho: float, p: float) -> str:
        parts = []
        if pd.notna(corr):
            parts.append(f"corr(age, montant)={corr:.3f}")
        if pd.notna(rho) and pd.notna(p):
            parts.append(f"Spearman ρ={rho:.3f} (p={p:.3f})")
        return " ; ".join(parts) if parts else "Analyse âge non concluante."

    # -------------------- PIPELINE GLOBAL (FINI) --------------------------

    def run_comprehensive_drought_analysis(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Workflow complet spécialisé sécheresse :
        load -> filter (pass-through si absent) -> preprocess (fallback auto) -> analyses -> visus.
        Crée une série mensuelle (mmAAAA), exporte HTML/PNG/CSV/JSON et retourne les chemins d’export.
        """
        logger.info("🚀 LANCEMENT DE L'ANALYSE COMPLÈTE SÉCHERESSE")

        # --- helper local pour fusionner config et kwargs ---
        def _resolve_params(key: str, local: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            base = {}
            if isinstance(self.config.get(key), dict):
                base.update(self.config.get(key))
            if isinstance(local, dict):
                base.update(local)
            return base

        from pathlib import Path
        import json
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime

        target_var = self.config.get("target_variable", "montant_charge_brut")
        date_col   = self.config.get("date_column", "date_sinistre")

        load_params       = _resolve_params("load_params",       kwargs.get("load_params"))
        filter_params     = _resolve_params("filter_params",     kwargs.get("filter_params"))
        preprocess_params = _resolve_params("preprocess_params", kwargs.get("preprocess_params"))
        visualizer_params = _resolve_params("visualizer_params", kwargs.get("visualizer_params"))

        # -------- 1) LOAD --------
        ext = str(Path(file_path).suffix).lower()
        if ext in (".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"):
            for bad in ("sep", "delimiter"):
                load_params.pop(bad, None)

        logger.info("Étape 1: Chargement des données")
        self.load_data(file_path, **load_params)
        if self.raw_data is None or len(self.raw_data) == 0:
            raise ValueError("Aucune donnée chargée.")

        # -------- 2) FILTER (pass-through si absent) --------
        logger.info("Étape 2: Filtrage des données")
        try:
            if filter_params:
                self.filter_data(**filter_params)
            else:
                self.filtered_data = self.raw_data.copy()
                logger.info("Aucun filtre fourni → filtered_data = raw_data (pass-through).")
        except Exception:
            logger.exception("Échec du filtrage, fallback pass-through.")
            self.filtered_data = self.raw_data.copy()

        # Garde-fou : si None ou vide → fallback
        if (getattr(self, "filtered_data", None) is None
            or (isinstance(self.filtered_data, pd.DataFrame) and self.filtered_data.empty)):
            logger.warning("filtered_data absent ou vide → fallback sur raw_data.")
            self.filtered_data = self.raw_data.copy()

        # -------- 3) PREPROCESS (avec retry si besoin) --------
        logger.info("Étape 3: Préprocessing des données : Nettoyage & encodage")
        try:
            self.preprocess_data(**preprocess_params)
        except Exception as e:
            msg = str(e)
            if "Aucune donnée filtrée" in msg:
                logger.warning("Retry preprocess: force fallback filtered_data = raw_data.")
                self.filtered_data = self.raw_data.copy()
                self.preprocess_data(**preprocess_params)
            else:
                logger.exception("Échec du prétraitement")
                raise

        if self.processed_data is None or len(self.processed_data) == 0:
            raise ValueError("Prétraitement terminé mais aucune donnée exploitable.")

        # -------- 4) ANALYSES --------
        logger.info("Étape 4: Analyses spécialisées sécheresse")
        try:
            climate = self.analyze_climate_impact(
                precipitation_threshold=kwargs.get("precipitation_threshold", 20.0),
                temperature_threshold=kwargs.get("temperature_threshold", 25.0),
            )
        except Exception:
            logger.exception("Erreur analyse climat")
            climate = {}

        try:
            vulnera = self.analyze_building_vulnerability()
        except Exception:
            logger.exception("Erreur analyse vulnérabilité")
            vulnera = {}

        try:
            exposure = self.analyze_risk_exposure()
        except Exception:
            logger.exception("Erreur analyse exposition")
            exposure = {}

        try:
            segment = self.create_risk_segmentation()
        except Exception:
            logger.exception("Erreur segmentation risque")
            segment = {}

        try:
            insights = self.generate_drought_specific_insights()
        except Exception as e:
            logger.exception("Erreur génération insights")
            insights = {"error": str(e)}

        # -------- 5) VISUALISATIONS (incl. série mensuelle mmAAAA) --------
        figures: Dict[str, plt.Figure] = {}
        try:
            figs = self.create_advanced_visualizations() or {}
            figures.update(figs)
        except Exception:
            logger.warning("Visualisations avancées non générées.", exc_info=True)

        # Série mensuelle (mmAAAA)
        monthly_frames: Dict[str, pd.DataFrame] = {}
        try:
            df = self.processed_data.copy()
            if date_col in df.columns and target_var in df.columns:
                agg = str(visualizer_params.get("monthly_agg", "sum"))
                min_count = int(visualizer_params.get("monthly_min_count", 0))

                d = df[[date_col, target_var]].copy()
                d[date_col]   = pd.to_datetime(d[date_col], errors="coerce")
                d[target_var] = pd.to_numeric(d[target_var], errors="coerce")
                d = d.dropna(subset=[date_col, target_var])

                if len(d):
                    d["ym"] = d[date_col].dt.to_period("M")
                    grouped = d.groupby("ym")[target_var]
                    try:
                        series = getattr(grouped, agg)()
                    except Exception:
                        logger.warning(f"Agrégation '{agg}' non supportée, fallback 'sum'.")
                        series = grouped.sum()

                    full_idx = pd.period_range(series.index.min(), series.index.max(), freq="M")
                    series   = series.reindex(full_idx, fill_value=0)

                    if min_count > 0:
                        counts = d.groupby("ym").size().reindex(full_idx, fill_value=0)
                        series = series.where(counts >= min_count, np.nan)

                    x_labels = [p.strftime("%m%Y") for p in full_idx]

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(x_labels, series.values, marker="o")
                    ax.set_xlabel("Mois (mmAAAA)")
                    ax.set_ylabel(f"{target_var} ({agg})")
                    ax.set_title(f"Série mensuelle {target_var} – ({agg})")
                    ax.tick_params(axis="x", rotation=90)
                    step = max(1, len(x_labels) // 24)
                    for i, lbl in enumerate(ax.get_xticklabels()):
                        lbl.set_visible(i % step == 0)
                    ax.grid(True, axis="y", alpha=0.3)
                    plt.tight_layout()

                    figures["monthly_series_mmAAAA"] = fig

                    monthly_frame = pd.DataFrame({
                        "ym": [str(p) for p in full_idx],        # ex: '2022-07'
                        "mmAAAA": x_labels,                      # ex: '072022'
                        f"{target_var}_{agg}": series.values
                    })
                    # ranger sans écraser la clé
                    key = "mmAAAA"
                    i = 2
                    while key in monthly_frames:
                        key = f"mmAAAA_{i}"
                        i += 1
                    monthly_frames[key] = monthly_frame

                    # Push au rapport si AdvancedReportGenerator dispo
                    try:
                        sm  = getattr(self, "synthesis_manager", None)
                        gen = getattr(sm, "advanced_generator", None) if sm else None
                        if gen:
                            gen.add_figure(
                                "serie_mensuelle_mmAAAA",
                                fig,
                                f"Série mensuelle de {target_var} agrégée ({agg}) au format mmAAAA."
                            )
                            # Injecter un minimum d'infos textuelles dans le rapport
                            gen.analysis_results = getattr(gen, "analysis_results", {})
                            gen.analysis_results.setdefault("metadata", {})["generated_at"] = datetime.now().isoformat()
                            gen.analysis_results.setdefault("insights", insights)
                            gen.analysis_results.setdefault("climate", climate)
                            gen.analysis_results.setdefault("vulnerability", vulnera)
                            gen.analysis_results.setdefault("exposure", exposure)
                            gen.analysis_results.setdefault("segmentation", segment)
                    except Exception:
                        logger.debug("Impossible d’attacher la figure/sections au rapport.", exc_info=True)
        except Exception:
            logger.warning("Échec génération série mensuelle mmAAAA.", exc_info=True)

        # -------- 6) EXPORTS (HTML/PNG/CSV/JSON) --------
        export_paths: Dict[str, str] = {}
        index_path: Optional[str] = None

        try:
            base_out = Path(self.config.get("output_dir", "./outputs/drought_analysis")).resolve()
            ts_dir = base_out / datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_dir.mkdir(parents=True, exist_ok=True)

            # 6.a) Sauvegarde figures en PNG
            for name, fig in figures.items():
                try:
                    png_path = ts_dir / f"{name}.png"
                    fig.savefig(png_path, dpi=150, bbox_inches="tight")
                    export_paths[name] = str(png_path)
                except Exception:
                    logger.warning(f"Impossible d'enregistrer la figure {name}.", exc_info=True)

            # 6.b) Sauvegarde des séries mensuelles en CSV
            for key, frame in monthly_frames.items():
                try:
                    csv_path = ts_dir / f"monthly_{key}.csv"
                    frame.to_csv(csv_path, index=False, encoding="utf-8")
                    export_paths[f"monthly_{key}"] = str(csv_path)
                except Exception:
                    logger.warning(f"Impossible d'enregistrer la série mensuelle {key}.", exc_info=True)

            # 6.c) JSON complet des résultats
            try:
                json_path = ts_dir / "results_drought.json"
                dumpable = {
                    "insights": insights,
                    "climate": climate,
                    "vulnerability": vulnera,
                    "exposure": exposure,
                    "segmentation": segment,
                    "figures": list(figures.keys()),
                    "monthly_frames": {k: v.to_dict(orient="list") for k, v in monthly_frames.items()},
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(dumpable, f, ensure_ascii=False, indent=2, default=self._json_default)
                export_paths["results_json"] = str(json_path)
            except Exception:
                logger.warning("Impossible d'enregistrer le JSON des résultats.", exc_info=True)

            # 6.d) Rapport HTML minimal (index.html)
            try:
                html_path = ts_dir / "index.html"
                title = f"Analyse Sécheresse – {Path(file_path).name}"
                # simple HTML
                parts = [
                    "<!doctype html><html lang='fr'><head><meta charset='utf-8'/>",
                    f"<title>{title}</title>",
                    "<style>body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;}",
                    "h1{margin:0 0 12px;} h2{margin-top:28px;} .fig{margin:18px 0;} img{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;} pre{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto;} table{border-collapse:collapse;} td,th{border:1px solid #ddd;padding:6px 8px;}</style>",
                    "</head><body>",
                    f"<h1>{title}</h1>",
                    f"<p>Généré le {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>",
                    "<h2>Insights</h2>",
                    "<pre>" + json.dumps(insights, ensure_ascii=False, indent=2) + "</pre>",
                ]

                if monthly_frames:
                    parts.append("<h2>Séries mensuelles (mmAAAA)</h2>")
                    for k in monthly_frames:
                        csv_name = Path(export_paths.get(f"monthly_{k}", "")).name
                        if csv_name:
                            parts.append(f"<p>• <a href='{csv_name}' download>Télécharger {csv_name}</a></p>")

                if export_paths:
                    parts.append("<h2>Figures</h2>")
                    for k, pth in export_paths.items():
                        if pth.endswith(".png"):
                            parts.append(f"<div class='fig'><h3>{k}</h3><img src='{Path(pth).name}' alt='{k}'/></div>")

                parts.append("</body></html>")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write("".join(parts))

                index_path = str(html_path)
                export_paths["index_html"] = index_path
            except Exception:
                logger.warning("Impossible de générer le rapport HTML.", exc_info=True)

            # 6.e) Si AdvancedReportGenerator est dispo, on tente aussi son export multi-format
            try:
                sm  = getattr(self, "synthesis_manager", None)
                gen = getattr(sm, "advanced_generator", None) if sm else None
                if gen:
                    # S'assurer d'un minimum de contenu textuel déjà injecté ci-dessus
                    title = f"Analyse Sécheresse - {Path(file_path).stem}"
                    paths = gen.export_all_formats(title)
                    # on ajoute au dict export_paths
                    if isinstance(paths, dict):
                        for k, v in paths.items():
                            export_paths[f"advanced_{k}"] = str(v)
                        if not index_path:
                            for v in paths.values():
                                if isinstance(v, (str, Path)) and str(v).lower().endswith(".html"):
                                    index_path = str(v)
                                    break
                        
            except Exception:
                logger.debug("Export AdvancedReportGenerator ignoré (non configuré ou erreur).", exc_info=True)

        except Exception:
            logger.exception("Bloc d'export: erreur inattendue.")

        # -------- 7) SORTIE --------
        out: Dict[str, Any] = {
            "climate": climate,
            "vulnerability": vulnera,
            "exposure": exposure,
            "segmentation": segment,
            "insights": insights,
            "figures": list(figures.keys()),
            "monthly_frames": monthly_frames,  # DataFrames
            "reports": export_paths,           # chemins des fichiers exportés
            "index": index_path,               # chemin vers l'HTML principal
        }
        logger.info("✅ Analyse sécheresse terminée")
        return out