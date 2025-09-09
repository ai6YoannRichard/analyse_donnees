from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class StatisticsManager:
    """Gestionnaire des analyses statistiques enrichi pour sinistres sécheresse"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.results: Dict[str, Any] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.temporal_results: Dict[str, Any] = {}
        self.spatial_results: Dict[str, Any] = {}
        self.figures: Dict[str, plt.Figure] = {}  # <— permet d’exposer des figures au rapport
        logger.info("StatisticsManager initialisé")

    def analyze_distribution_by_groups(
        self,
        df: pd.DataFrame,
        target_var: str,
        group_var: str,
        order: Optional[List[Any]] = None,       # ex. ["faible","moyen","fort","maximal"]
        min_count: int = 20,                     # n minimum par groupe pour être reporté
        positive_only: bool = True,              # ne garder que montants > 0
        winsor: Optional[Tuple[float, float]] = (0.01, 0.99),  # clip 1–99% par défaut
        log1p: bool = False,                     # si True, applique log1p avant l'analyse
        bins: Optional[List[float]] = None,      # pour discrétiser nb d'étages, etc.
        bin_labels: Optional[List[str]] = None,
        coerce_group_numeric: bool = False,      # True pour forcer group_var en numérique (ex: étages)
    ) -> pd.DataFrame:
        """
        Applique _analyze_target_distribution à chaque niveau de group_var et
        renvoie un DataFrame synthétique par groupe (n, mean/median, p90/p95/p99, etc.).
        Le résultat est aussi stocké dans self.results[f'distribution_by_{group_var}'].
        """
        try:
            if target_var not in df.columns or group_var not in df.columns:
                logger.warning(f"Colonnes manquantes: {group_var}/{target_var}")
                return pd.DataFrame()

            d = df[[group_var, target_var]].copy()
            d[target_var] = pd.to_numeric(d[target_var], errors="coerce")

            if coerce_group_numeric:
                d[group_var] = pd.to_numeric(d[group_var], errors="coerce")

            d = d.dropna(subset=[group_var, target_var])
            if positive_only:
                d = d[d[target_var] > 0]
            if d.empty:
                logger.warning("Données vides après nettoyage (analyze_distribution_by_groups)")
                return pd.DataFrame()

            # Binning éventuel (utile pour nb_etages_batiment)
            if bins is not None:
                d[group_var] = pd.cut(d[group_var], bins=bins, labels=bin_labels, right=True)

            # Ordre métier éventuel (ex: RGA)
            if order is not None:
                d[group_var] = (
                    d[group_var].astype(str).str.lower()
                    .astype("category")
                    .cat.set_categories([str(x).lower() for x in order], ordered=True)
                )

            # Winsorize léger
            val_col = target_var
            if winsor is not None:
                lo, hi = winsor
                qlo, qhi = d[val_col].quantile([lo, hi])
                d[val_col] = d[val_col].clip(qlo, qhi)

            # Log1p si demandé
            if log1p:
                d["_val"] = np.log1p(d[val_col])
                val_col = "_val"

            rows: List[Dict[str, Any]] = []
            for lvl, sub in d.groupby(group_var, observed=True):
                s = sub[val_col].dropna()
                if len(s) < min_count:
                    continue

                dist = self._analyze_target_distribution(s)
                if isinstance(dist, dict) and "error" in dist:
                    continue

                bs = dist["basic_stats"]
                pc = dist["percentiles"]
                ta = dist["tail_analysis"]

                rows.append({
                    "group": lvl,
                    "n": int(bs["count"]),
                    "mean": float(bs["mean"]),
                    "median": float(bs["median"]),
                    "std": float(bs["std"]),
                    "skewness": float(bs["skewness"]),
                    "kurtosis": float(bs["kurtosis"]),
                    "p05": float(pc.get("p5")),
                    "p50": float(pc.get("p50")),
                    "p90": float(pc.get("p90")),
                    "p95": float(pc.get("p95")),
                    "p99": float(pc.get("p99")),
                    "heavy_tails": bool(ta.get("heavy_tails")),
                    "right_skewed": bool(ta.get("right_skewed")),
                    "concentration_ratio": float(ta.get("concentration_ratio", np.nan)),
                })

            out = pd.DataFrame(rows)
            if not out.empty:
                # essayer de trier par l’ordre catégoriel si présent
                try:
                    out = out.sort_values(
                        by="group",
                        key=lambda s: s.astype("category").cat.codes
                    ).reset_index(drop=True)
                except Exception:
                    out = out.sort_values(by="group").reset_index(drop=True)

            self.results[f"distribution_by_{group_var}"] = out
            logger.info(f"Distribution par groupes calculée pour {group_var} ({len(out)} lignes)")
            return out

        except Exception as e:
            logger.error(f"Erreur analyze_distribution_by_groups({group_var}): {e}")
            return pd.DataFrame()
    def monthly_group_stats(        self,        df: pd.DataFrame,        target_var: str = "montant_charge_brut",        date_col: str = "date_sinistre",        min_count: int = 10,    ) -> Optional[pd.DataFrame]:
        """
        Prépare une colonne 'mmaaaa' et délègue à analyze_distribution_by_groups(...).
        """
        if target_var not in df.columns:
            return None

        tmp = df[[target_var]].copy()
        if date_col in df.columns:
            tmp["date"] = pd.to_datetime(df[date_col], errors="coerce")
            tmp = tmp.dropna(subset=["date", target_var])
            tmp["periode"] = tmp["date"].dt.to_period("M")
        elif {"mois_exercice", "annee_exercice"}.issubset(df.columns):
            m = pd.to_numeric(df["mois_exercice"], errors="coerce")
            y = pd.to_numeric(df["annee_exercice"], errors="coerce")
            tmp = pd.DataFrame({target_var: pd.to_numeric(df[target_var], errors="coerce")})
            tmp["periode"] = pd.PeriodIndex(pd.to_datetime(dict(year=y, month=m, day=1)), freq="M")
            tmp = tmp.dropna(subset=["periode", target_var])
        else:
            return None

        tmp["mmaaaa"] = tmp["periode"].to_timestamp().dt.strftime("%m%Y")
        order = sorted(tmp["mmaaaa"].unique(), key=lambda s: (int(s[-4:]), int(s[:2])))
        # Réutilise ta méthode existante :
        return self.analyze_distribution_by_groups(
            df=tmp,
            target_var=target_var,
            group_var="mmaaaa",
            order=order,
            min_count=min_count,
            positive_only=True,
            winsor=(0.01, 0.99),
            coerce_group_numeric=False,
        )
    
    def analyze_floors_vs_amount(        self,        df: pd.DataFrame,        target_var: str = "montant_charge_brut",        floors_var: str = "nb_etages_batiment",    ) -> Optional[Dict[str, Any]]:
        """Analyse du montant vs nombre d’étages (agrégats, tendance, tests, tailles d’effet)."""
        try:
            if floors_var not in df.columns or target_var not in df.columns:
                logger.warning(f"Colonnes manquantes: {floors_var}/{target_var}")
                return None

            d = pd.DataFrame({
                floors_var: pd.to_numeric(df[floors_var], errors="coerce"),
                target_var: pd.to_numeric(df[target_var], errors="coerce"),
            }).dropna()

            d = d[(d[target_var] > 0) & (d[floors_var] >= 0) & (d[floors_var] <= 30)]
            if d.empty:
                logger.warning("Aucune donnée valide après nettoyage pour nb étages")
                return None

            d[floors_var] = d[floors_var].round().astype(int)

            # Winsorize léger (1–99%)
            q1, q99 = d[target_var].quantile([0.01, 0.99])
            d[target_var] = d[target_var].clip(q1, q99)

            # Résumé par nombre d’étages
            def trimmed_mean(s: pd.Series) -> float:
                s = s.dropna().sort_values()
                n = len(s)
                if n < 20:
                    return float(s.mean())
                a, b = int(0.05*n), int(0.95*n)
                b = max(b, a+1)
                return float(s.iloc[a:b].mean())

            grp = d.groupby(floors_var, observed=True)[target_var]
            summary = pd.DataFrame({
                "n": grp.size(),
                "median": grp.median(),
                "iqr": grp.quantile(0.75) - grp.quantile(0.25),
                "mean_trimmed": grp.apply(trimmed_mean),
                "mean": grp.mean(),
                "std": grp.std()
            }).reset_index().rename(columns={floors_var: "floors"})

            # Test global (Kruskal–Wallis)
            groups = [g.values for _, g in d.groupby(floors_var, observed=True)[target_var]]
            out: Dict[str, Any] = {"summary": summary}
            valid_groups = [g for g in groups if len(g) >= 3]
            if len(valid_groups) >= 2:
                H, p = stats.kruskal(*valid_groups, nan_policy="omit")
                out["kruskal"] = {
                    "H": float(H),
                    "p_value": float(p),
                    "epsilon2": float(self._epsilon_squared_kw(H, len(valid_groups), len(d))),
                }

            # Tendance monotone (Spearman)
            rho, p_s = stats.spearmanr(d[floors_var].values, d[target_var].values, nan_policy="omit")
            out["trend"] = {
                "spearman_rho": float(rho),
                "p_value": float(p_s),
                "monotone_increase": bool((rho > 0) and (p_s < 0.05)),
            }

            # Tailles d’effet (Cliff’s delta) entre étages adjacents
            deltas: Dict[str, float] = {}
            levels = sorted(summary["floors"].unique().tolist())
            for i in range(len(levels) - 1):
                a = d.loc[d[floors_var] == levels[i], target_var].values
                b = d.loc[d[floors_var] == levels[i+1], target_var].values
                if len(a) >= 3 and len(b) >= 3:
                    deltas[f"{levels[i]} vs {levels[i+1]}"] = float(self._cliffs_delta(a, b))
            out["cliffs_delta_adjacent"] = deltas

            # Régression quantile (médiane & 90e pct) optionnelle
            try:
                import statsmodels.formula.api as smf
                d2 = d.copy()
                res50 = smf.quantreg(f"{target_var} ~ {floors_var}", d2).fit(q=0.5)
                res90 = smf.quantreg(f"{target_var} ~ {floors_var}", d2).fit(q=0.9)
                out["quantiles"] = {
                    "q50_slope": float(res50.params.get(floors_var, np.nan)),
                    "q50_p": float(res50.pvalues.get(floors_var, np.nan)),
                    "q90_slope": float(res90.params.get(floors_var, np.nan)),
                    "q90_p": float(res90.pvalues.get(floors_var, np.nan)),
                }
            except Exception:
                out["quantiles"] = None

            self.results["floors_vs_amount"] = out
            logger.info("Analyse nb étages vs montant terminée")
            return out

        except Exception as e:
            logger.error(f"Erreur dans analyze_floors_vs_amount: {e}")
            return None


    def plot_floors_vs_amount(
        self,
        df: pd.DataFrame,
        target_var: str = "montant_charge_brut",
        floors_var: str = "nb_etages_batiment",
        figure_name: str = "floors_vs_amount",
        logy: bool = True,
    ) -> Optional[plt.Figure]:
        """Boxplot + nuage de points par bin d’étages, avec médianes."""
        try:
            need_cols = {floors_var, target_var}
            if not need_cols.issubset(df.columns):
                return None

            d = df[[floors_var, target_var]].copy()
            d[floors_var] = pd.to_numeric(d[floors_var], errors="coerce")
            d[target_var] = pd.to_numeric(d[target_var], errors="coerce")
            d = d.dropna(subset=[floors_var, target_var])
            d = d[d[target_var] > 0]
            if d.empty:
                return None

            q1, q99 = d[target_var].quantile([0.01, 0.99])
            d[target_var] = d[target_var].clip(q1, q99)

            # Bins lisibles
            edges = [-0.5, 1.5, 2.5, 4.5, 7.5, np.inf]
            labels = ["1", "2", "3-4", "5-7", "8+"]
            d["floors_bin"] = pd.cut(d[floors_var], bins=edges, labels=labels, right=True)
            if d["floors_bin"].nunique(dropna=True) < 2:
                try:
                    d["floors_bin"] = pd.qcut(d[floors_var], q=min(4, d[floors_var].nunique()), duplicates="drop")
                except Exception:
                    d["floors_bin"] = d[floors_var].round().astype(int).astype(str)

            fig, ax = plt.subplots(figsize=(9, 6))
            sns.boxplot(data=d, x="floors_bin", y=target_var, showfliers=False, ax=ax)
            sns.stripplot(data=d, x="floors_bin", y=target_var, alpha=0.25, jitter=0.25, dodge=True, ax=ax)

            # Ligne des médianes
            med = d.groupby("floors_bin", observed=True)[target_var].median()
            ax.plot(range(len(med)), med.values, marker="o", linewidth=2)

            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Nombre d’étages (bins)")
            ax.set_ylabel("Montant brut")
            ax.set_title("Montant brut vs nombre d’étages (bins)")

            self.figures[figure_name] = fig
            return fig
        except Exception as e:
            logger.error(f"Erreur dans plot_floors_vs_amount: {e}")
            return None
    
    def descriptive_statistics(self, df: pd.DataFrame, target_var: str = 'montant_charge_brut') -> pd.DataFrame:
        """Calcule les statistiques descriptives enrichies"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                logger.warning("Aucune variable numérique pour les statistiques descriptives")
                return pd.DataFrame()

            desc_stats = numeric_df.describe()

            # Statistiques supplémentaires
            additional_stats = pd.DataFrame({
                'missing_count': numeric_df.isnull().sum(),
                'missing_percent': (numeric_df.isnull().sum() / len(df) * 100),
                'skewness': numeric_df.skew(),
                'kurtosis': numeric_df.kurtosis(),
                'cv': numeric_df.std() / numeric_df.mean().replace(0, np.nan),  # éviter division par zéro
                'q05': numeric_df.quantile(0.05),
                'q95': numeric_df.quantile(0.95),
                'iqr': numeric_df.quantile(0.75) - numeric_df.quantile(0.25)
            }).T

            desc_stats = pd.concat([desc_stats, additional_stats])

            # Focus cible
            if target_var in numeric_df.columns:
                target_stats = self._analyze_target_distribution(numeric_df[target_var])
                self.results['target_distribution'] = target_stats

            self.results['descriptive'] = desc_stats
            logger.info(f"Statistiques descriptives calculées pour {len(numeric_df.columns)} variables")
            return desc_stats

        except Exception as e:
            logger.error(f"Erreur dans descriptive_statistics: {e}")
            return pd.DataFrame()

    def _analyze_target_distribution(self, target_series: pd.Series) -> Dict[str, Any]:
        """Analyse approfondie de la distribution de la variable cible"""
        try:
            target_clean = target_series.dropna()
            if len(target_clean) == 0:
                return {'error': 'Aucune donnée valide'}

            sample_size = min(5000, len(target_clean))
            if sample_size < 3:
                return {'error': 'Échantillon trop petit pour les tests'}

            sample_data = target_clean.sample(sample_size) if len(target_clean) > sample_size else target_clean

            # Tests de normalité
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            jarque_bera_stat, jarque_bera_p = stats.jarque_bera(target_clean)

            # Percentiles / queues
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = {f'p{p}': float(target_clean.quantile(p/100)) for p in percentiles}

            return {
                'basic_stats': {
                    'count': len(target_clean),
                    'mean': float(target_clean.mean()),
                    'median': float(target_clean.median()),
                    'mode': float(target_clean.mode().iloc[0]) if len(target_clean.mode()) > 0 else None,
                    'std': float(target_clean.std()),
                    'var': float(target_clean.var()),
                    'skewness': float(target_clean.skew()),
                    'kurtosis': float(target_clean.kurtosis())
                },
                'normality_tests': {
                    'shapiro_wilk': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_p)},
                    'jarque_bera': {'statistic': float(jarque_bera_stat), 'p_value': float(jarque_bera_p)}
                },
                'percentiles': percentile_values,
                'tail_analysis': {
                    'heavy_tails': float(target_clean.kurtosis()) > 3,
                    'right_skewed': float(target_clean.skew()) > 1,
                    'concentration_ratio': float((percentile_values['p90'] - percentile_values['p10']) / target_clean.median())
                }
            }
        except Exception as e:
            logger.error(f"Erreur dans _analyze_target_distribution: {e}")
            return {'error': str(e)}

    def temporal_analysis(self, df: pd.DataFrame, target_var: str = 'montant_charge_brut',
                          date_col: str = 'date_sinistre') -> Dict[str, Any]:
        """Analyse temporelle complète"""
        try:
            if date_col not in df.columns or target_var not in df.columns:
                logger.error(f"Colonnes {date_col} ou {target_var} manquantes")
                return {}

            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
            temp_df[target_var] = pd.to_numeric(temp_df[target_var], errors='coerce')
            temp_df = temp_df.dropna(subset=[date_col, target_var])
            if len(temp_df) == 0:
                logger.warning("Pas de données valides pour l'analyse temporelle")
                return {}

            # Dimensions temporelles
            temp_df['year'] = temp_df[date_col].dt.year
            temp_df['month'] = temp_df[date_col].dt.month
            temp_df['quarter'] = temp_df[date_col].dt.quarter
            temp_df['day_of_year'] = temp_df[date_col].dt.dayofyear
            temp_df['season'] = temp_df['month'].map({
                12: 'Hiver', 1: 'Hiver', 2: 'Hiver',
                3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
                6: 'Été', 7: 'Été', 8: 'Été',
                9: 'Automne', 10: 'Automne', 11: 'Automne'
            })

            temporal_results: Dict[str, Any] = {}

            yearly_stats = temp_df.groupby('year')[target_var].agg(['count', 'mean', 'median', 'std', 'sum']).round(2)
            temporal_results['yearly'] = yearly_stats.to_dict('index')

            seasonal_stats = temp_df.groupby('season')[target_var].agg(['count', 'mean', 'median', 'std']).round(2)
            temporal_results['seasonal'] = seasonal_stats.to_dict('index')

            monthly_stats = temp_df.groupby('month')[target_var].agg(['count', 'mean', 'median', 'std']).round(2)
            temporal_results['monthly'] = monthly_stats.to_dict('index')

            # Tendance
            temp_df_sorted = temp_df.sort_values(date_col)
            temp_df_sorted['days_since_start'] = (temp_df_sorted[date_col] - temp_df_sorted[date_col].min()).dt.days

            if len(temp_df_sorted) >= 3:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    temp_df_sorted['days_since_start'], temp_df_sorted[target_var]
                )
                temporal_results['trend'] = {
                    'slope_per_day': float(slope),
                    'slope_per_year': float(slope * 365.25),
                    'r_squared': float(r_value**2),
                    'p_value': float(p_value),
                    'significant_trend': p_value < 0.05
                }

            # Pics saisonniers
            monthly_means = temp_df.groupby('month')[target_var].mean()
            peak_months = monthly_means.nlargest(3).index.tolist()
            low_months = monthly_means.nsmallest(3).index.tolist()
            temporal_results['seasonality'] = {
                'peak_months': peak_months,
                'low_months': low_months,
                'seasonal_amplitude': float((monthly_means.max() - monthly_means.min()) / monthly_means.mean()),
                'summer_vs_winter': float(monthly_means[[6, 7, 8]].mean() / monthly_means[[12, 1, 2]].mean())
                if monthly_means[[12, 1, 2]].mean() > 0 else None
            }

            self.temporal_results = temporal_results
            logger.info("Analyse temporelle terminée")
            return temporal_results

        except Exception as e:
            logger.error(f"Erreur dans temporal_analysis: {e}")
            return {}

    def spatial_analysis(self, df: pd.DataFrame, target_var: str = 'montant_charge_brut') -> Dict[str, Any]:
        """Analyse spatiale par région/département"""
        try:
            if target_var not in df.columns:
                logger.error(f"Variable {target_var} manquante")
                return {}

            spatial_results: Dict[str, Any] = {}

            # Par département (cpostal -> 2 premiers chiffres)
            if 'cpostal' in df.columns:
                df_spatial = df.copy()
                df_spatial[target_var] = pd.to_numeric(df_spatial[target_var], errors='coerce')
                df_spatial = df_spatial.dropna(subset=[target_var, 'cpostal'])

                if len(df_spatial) > 0:
                    df_spatial['departement'] = df_spatial['cpostal'].astype(str).str[:2]
                    dept_stats = df_spatial.groupby('departement')[target_var].agg(
                        ['count', 'mean', 'median', 'std', 'sum']
                    ).round(2)

                    dept_stats = dept_stats[dept_stats['count'] >= 10]
                    if not dept_stats.empty:
                        dept_stats = dept_stats.sort_values('mean', ascending=False)
                        spatial_results['departements'] = {
                            'stats': dept_stats.to_dict('index'),
                            'top_3_montant_moyen': dept_stats.head(3)['mean'].to_dict(),
                            'top_3_volume': dept_stats.nlargest(3, 'count')['count'].to_dict(),
                            'concentration': {
                                'top_5_pct_volume': float(
                                    dept_stats.nlargest(5, 'count')['count'].sum() / dept_stats['count'].sum() * 100
                                )
                            }
                        }

            # Par ville
            if 'ville' in df.columns:
                ville_stats = df.groupby('ville')[target_var].agg(['count', 'mean', 'median']).round(2)
                ville_stats = ville_stats[ville_stats['count'] >= 5]
                if not ville_stats.empty:
                    spatial_results['villes'] = {
                        'total_cities': len(ville_stats),
                        'top_10_montant': ville_stats.nlargest(10, 'mean')['mean'].to_dict(),
                        'top_10_volume': ville_stats.nlargest(10, 'count')['count'].to_dict()
                    }

            # Par coordonnées
            if all(col in df.columns for col in ['latitude', 'longitude']):
                geo_df = df.dropna(subset=['latitude', 'longitude', target_var])
                geo_df[target_var] = pd.to_numeric(geo_df[target_var], errors='coerce').dropna()
                if len(geo_df) > 0:
                    lat_corr = geo_df['latitude'].corr(geo_df[target_var])
                    lon_corr = geo_df['longitude'].corr(geo_df[target_var])
                    spatial_results['geographic'] = {
                        'centroid': {
                            'latitude': float(geo_df['latitude'].mean()),
                            'longitude': float(geo_df['longitude'].mean())
                        },
                        'bounds': {
                            'lat_min': float(geo_df['latitude'].min()),
                            'lat_max': float(geo_df['latitude'].max()),
                            'lon_min': float(geo_df['longitude'].min()),
                            'lon_max': float(geo_df['longitude'].max())
                        },
                        'spatial_correlation': {
                            'lat_corr': float(lat_corr) if not pd.isna(lat_corr) else None,
                            'lon_corr': float(lon_corr) if not pd.isna(lon_corr) else None
                        }
                    }

            self.spatial_results = spatial_results
            logger.info(f"Analyse spatiale terminée - {len(spatial_results)} dimensions analysées")
            return spatial_results

        except Exception as e:
            logger.error(f"Erreur dans spatial_analysis: {e}")
            return {}

    def advanced_outlier_detection(self, df: pd.DataFrame, target_var: str = 'montant_charge_brut') -> Dict[str, Any]:
        """Détection d'outliers avancée avec plusieurs méthodes"""
        try:
            if target_var not in df.columns:
                logger.error(f"Variable {target_var} manquante")
                return {}

            target_data = pd.to_numeric(df[target_var], errors='coerce').dropna()
            if len(target_data) == 0:
                return {'error': 'Aucune donnée valide'}

            outlier_results: Dict[str, Any] = {}

            # IQR
            Q1 = target_data.quantile(0.25)
            Q3 = target_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_iqr = Q1 - 1.5 * IQR
                upper_iqr = Q3 + 1.5 * IQR
                iqr_outliers = target_data[(target_data < lower_iqr) | (target_data > upper_iqr)]
                outlier_results['iqr_method'] = {
                    'count': len(iqr_outliers),
                    'percentage': float(len(iqr_outliers) / len(target_data) * 100),
                    'threshold_lower': float(lower_iqr),
                    'threshold_upper': float(upper_iqr),
                    'values_sample': iqr_outliers.nlargest(10).tolist()
                }

            # Percentiles
            p95 = target_data.quantile(0.95)
            p5 = target_data.quantile(0.05)
            percentile_outliers_high = target_data[target_data > p95]
            percentile_outliers_low = target_data[target_data < p5]
            outlier_results['percentile_method'] = {
                'high_outliers': {
                    'count': len(percentile_outliers_high),
                    'percentage': float(len(percentile_outliers_high) / len(target_data) * 100),
                    'threshold': float(p95),
                    'values_sample': percentile_outliers_high.nlargest(10).tolist()
                },
                'low_outliers': {
                    'count': len(percentile_outliers_low),
                    'percentage': float(len(percentile_outliers_low) / len(target_data) * 100),
                    'threshold': float(p5),
                    'values_sample': percentile_outliers_low.nsmallest(10).tolist()
                }
            }

            # Z-score modifié (MAD)
            median = target_data.median()
            mad = np.median(np.abs(target_data - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (target_data - median) / mad
                mad_outliers = target_data[np.abs(modified_z_scores) > 3.5]
                outlier_results['mad_method'] = {
                    'count': len(mad_outliers),
                    'percentage': float(len(mad_outliers) / len(target_data) * 100),
                    'threshold': 3.5,
                    'values_sample': mad_outliers.nlargest(10).tolist()
                }

            # Z-score classique
            if target_data.std() > 0:
                z_scores = np.abs(stats.zscore(target_data))
                zscore_outliers = target_data[z_scores > 3]
                outlier_results['zscore_method'] = {
                    'count': len(zscore_outliers),
                    'percentage': float(len(zscore_outliers) / len(target_data) * 100),
                    'threshold': 3.0,
                    'values_sample': zscore_outliers.nlargest(10).tolist()
                }

            # Outliers exceptionnels
            p99 = target_data.quantile(0.99)
            exceptional_outliers = target_data[target_data > p99]
            outlier_results['exceptional_analysis'] = {
                'p99_threshold': float(p99),
                'exceptional_count': len(exceptional_outliers),
                'exceptional_percentage': float(len(exceptional_outliers) / len(target_data) * 100),
                'exceptional_mean': float(exceptional_outliers.mean()) if len(exceptional_outliers) > 0 else None,
                'ratio_to_median': float(exceptional_outliers.mean() / median) if median > 0 and len(exceptional_outliers) > 0 else None,
                'top_10_values': exceptional_outliers.nlargest(10).tolist()
            }

            outlier_results['recommendations'] = self._get_outlier_recommendations(outlier_results, target_data)

            self.results['advanced_outliers'] = outlier_results
            iqr_count = outlier_results.get('iqr_method', {}).get('count', 0)
            logger.info(f"Détection d'outliers avancée terminée - {iqr_count} outliers IQR détectés")
            return outlier_results

        except Exception as e:
            logger.error(f"Erreur dans advanced_outlier_detection: {e}")
            return {}

    def _get_outlier_recommendations(self, outlier_results: Dict, target_data: pd.Series) -> Dict[str, Any]:
        """Génère des recommandations pour le traitement des outliers"""
        try:
            total_outliers_iqr = outlier_results.get('iqr_method', {}).get('count', 0)
            total_data = len(target_data)
            outlier_percentage = (total_outliers_iqr / total_data * 100) if total_data > 0 else 0

            recommendations = {
                'action_needed': outlier_percentage > 5,
                'severity': 'low' if outlier_percentage < 2 else 'medium' if outlier_percentage < 10 else 'high',
                'outlier_percentage': float(outlier_percentage),
                'suggested_actions': []
            }

            if outlier_percentage > 10:
                recommendations['suggested_actions'].extend([
                    'Vérifier la qualité des données source',
                    'Investiguer les cas exceptionnels manuellement',
                    'Considérer une transformation logarithmique'
                ])
            elif outlier_percentage > 5:
                recommendations['suggested_actions'].extend([
                    'Appliquer un winsorizing (clip aux percentiles 1-99)',
                    'Segmenter l\'analyse par type de sinistre'
                ])
            else:
                recommendations['suggested_actions'].append('Conserver les outliers pour l\'analyse')

            return recommendations
        except Exception as e:
            logger.error(f"Erreur dans _get_outlier_recommendations: {e}")
            return {'error': str(e)}

    def correlation_analysis(self, df: pd.DataFrame, target_variable: str = 'montant_charge_brut',
                             threshold: float = 0.05, selected_vars: List[str] = None) -> pd.DataFrame:
        """Analyse les corrélations avec la variable cible"""
        try:
            if selected_vars:
                available_vars = [var for var in selected_vars if var in df.columns]
                numeric_df = df[available_vars].select_dtypes(include=[np.number])
            else:
                numeric_df = df.select_dtypes(include=[np.number])

            if target_variable not in numeric_df.columns:
                logger.error(f"Variable cible {target_variable} non trouvée")
                return pd.DataFrame()

            # Corrélations
            full_corr = numeric_df.corr()
            correlations = full_corr[target_variable].abs().sort_values(ascending=False)
            correlations = correlations.drop(target_variable, errors='ignore')

            significant_corr = correlations[correlations >= threshold]
            if significant_corr.empty:
                logger.warning(f"Aucune corrélation significative trouvée (seuil={threshold})")
                return pd.DataFrame()

            corr_df = pd.DataFrame({
                'variable': significant_corr.index,
                'correlation_abs': significant_corr.values,
                'correlation': full_corr[target_variable].loc[significant_corr.index].values,
                'data_count': numeric_df[significant_corr.index].count().values,
                'missing_pct': (numeric_df[significant_corr.index].isnull().sum() / len(df) * 100).round(2).values
            })
            corr_df.reset_index(drop=True, inplace=True)

            corr_df['correlation_strength'] = pd.cut(
                corr_df['correlation_abs'],
                bins=[0, 0.1, 0.3, 0.5, 0.7, 1.0],
                labels=['Très faible', 'Faible', 'Modérée', 'Forte', 'Très forte']
            )

            self.results['correlations'] = corr_df
            logger.info(f"Analyse de corrélation terminée - {len(corr_df)} variables significatives (seuil={threshold})")
            return corr_df

        except Exception as e:
            logger.error(f"Erreur dans correlation_analysis: {e}")
            return pd.DataFrame()

    def group_analysis(self, df: pd.DataFrame, cat_var: str, target_var: str) -> Optional[pd.DataFrame]:
        """Agrégations par groupe avec gestion d'erreurs"""
        try:
            if cat_var not in df.columns or target_var not in df.columns:
                logger.warning(f"Colonnes manquantes pour group_analysis: {cat_var} / {target_var}")
                return None

            tmp = pd.DataFrame({
                cat_var: df[cat_var].astype(str),
                target_var: pd.to_numeric(df[target_var], errors='coerce'),
            }).dropna()

            if tmp.empty:
                logger.warning(f"Aucune donnée valide pour group_analysis({cat_var})")
                return None

            agg = tmp.groupby(cat_var, observed=True)[target_var].agg([
                ('count', 'count'),
                ('mean', 'mean'),
                ('median', 'median'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max')
            ]).round(2)

            total_count = tmp.shape[0]
            agg['percentage'] = (agg['count'] / total_count * 100).round(1)

            result_df = agg.sort_values('mean', ascending=False).reset_index()
            self.results[f'group_analysis_{cat_var}'] = result_df
            logger.info(f"Analyse par groupe effectuée: {cat_var} ({len(result_df)} niveaux)")
            return result_df

        except Exception as e:
            logger.error(f"Erreur dans group_analysis pour {cat_var}: {e}")
            return None

    def statistical_tests(self, df: pd.DataFrame, cat_var: str, target_var: str) -> Optional[Dict[str, Any]]:
        """Test de Kruskal-Wallis avec gestion d'erreurs"""
        try:
            if cat_var not in df.columns or target_var not in df.columns:
                logger.warning(f"Colonnes manquantes pour statistical_tests: {cat_var} / {target_var}")
                return None

            tmp = pd.DataFrame({
                cat_var: df[cat_var].astype(str),
                target_var: pd.to_numeric(df[target_var], errors='coerce'),
            }).dropna()

            if tmp.empty:
                logger.warning(f"Aucune donnée valide pour statistical_tests({cat_var})")
                return None

            groups = [group[target_var].values for _, group in tmp.groupby(cat_var)]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) < 2:
                logger.warning(f"Trop peu de groupes valides pour statistical_tests({cat_var})")
                return None

            stat, p = stats.kruskal(*groups, nan_policy='omit')

            if p < 0.001:
                significance = "Très significatif"
            elif p < 0.01:
                significance = "Significatif"
            elif p < 0.05:
                significance = "Modérément significatif"
            else:
                significance = "Non significatif"

            result = {
                'test': 'Kruskal-Wallis',
                'statistic': float(stat),
                'p_value': float(p),
                'groups_count': len(groups),
                'significance': significance,
                'interpretation': f"Différences entre groupes: {significance.lower()}"
            }

            self.test_results[cat_var] = result
            logger.info(f"Test Kruskal–Wallis pour {cat_var}: p={p:.3g} ({significance})")
            return result

        except Exception as e:
            logger.error(f"Erreur dans statistical_tests pour {cat_var}: {e}")
            return None

    def outlier_detection(self, df: pd.DataFrame, target_var: str) -> Optional[Dict[str, Any]]:
        """Détection d'outliers simple via IQR"""
        try:
            if target_var not in df.columns:
                logger.warning(f"Variable cible {target_var} absente")
                return None

            s = pd.to_numeric(df[target_var], errors='coerce').dropna()
            if s.empty:
                logger.warning("Aucune donnée valide pour outlier_detection")
                return None

            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                logger.warning(f"IQR nul pour {target_var}")
                return {
                    'q1': float(q1), 'q3': float(q3), 'iqr': 0.0,
                    'outlier_rate': 0.0, 'n': int(s.size),
                    'warning': 'Distribution très concentrée'
                }

            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (s < lower) | (s > upper)
            rate = float(outlier_mask.mean())

            summary = {
                'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr),
                'lower': float(lower), 'upper': float(upper),
                'outlier_count': int(outlier_mask.sum()),
                'outlier_rate': rate,
                'n': int(s.size),
                'mean': float(s.mean()),
                'median': float(s.median()),
                'std': float(s.std())
            }

            self.results['outliers_target'] = summary
            logger.info(f"Outliers (IQR) pour {target_var}: {rate:.1%}")
            return summary

        except Exception as e:
            logger.error(f"Erreur dans outlier_detection pour {target_var}: {e}")
            return None

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Résumé complet de toutes les analyses"""
        try:
            summary: Dict[str, Any] = {
                'timestamp': datetime.now().isoformat(),
                'analyses_performed': {
                    'descriptive': 'descriptive' in self.results,
                    'temporal': bool(self.temporal_results),
                    'spatial': bool(self.spatial_results),
                    'correlations': 'correlations' in self.results,
                    'outliers': 'advanced_outliers' in self.results,
                    'outliers_simple': 'outliers_target' in self.results
                },
                'key_findings': {}
            }

            if self.temporal_results:
                temporal = self.temporal_results
                peak_season = None
                if temporal.get('seasonal'):
                    seasonal_data = temporal['seasonal']
                    peak_season = max(seasonal_data.keys(),
                                      key=lambda x: seasonal_data[x].get('mean', 0))
                summary['key_findings']['temporal'] = {
                    'has_significant_trend': temporal.get('trend', {}).get('significant_trend', False),
                    'peak_season': peak_season,
                    'seasonal_amplitude': temporal.get('seasonality', {}).get('seasonal_amplitude', 0)
                }

            if self.spatial_results:
                spatial = self.spatial_results
                top_dept = None
                if 'departements' in spatial and spatial['departements'].get('top_3_montant_moyen'):
                    top_dept = list(spatial['departements']['top_3_montant_moyen'].keys())[0]
                summary['key_findings']['spatial'] = {
                    'top_department': top_dept,
                    'spatial_concentration': spatial.get('departements', {}).get('concentration', {}).get('top_5_pct_volume', 0)
                }

            if 'correlations' in self.results:
                corr_df = self.results['correlations']
                if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                    summary['key_findings']['correlations'] = {
                        'strongest_correlation': {
                            'variable': corr_df.iloc[0]['variable'],
                            'correlation': float(corr_df.iloc[0]['correlation'])
                        },
                        'significant_variables_count': int(len(corr_df)),
                        'high_correlations_count': int(
                            (corr_df['correlation_strength'] == 'Forte').sum()
                        ) if 'correlation_strength' in corr_df.columns else 0
                    }

            if 'advanced_outliers' in self.results:
                outliers = self.results['advanced_outliers']
                summary['key_findings']['outliers'] = {
                    'exceptional_cases_pct': outliers.get('exceptional_analysis', {}).get('exceptional_percentage', 0),
                    'treatment_needed': outliers.get('recommendations', {}).get('action_needed', False),
                    'severity': outliers.get('recommendations', {}).get('severity', 'low')
                }
            elif 'outliers_target' in self.results:
                outliers = self.results['outliers_target']
                summary['key_findings']['outliers'] = {
                    'outlier_rate': outliers.get('outlier_rate', 0),
                    'outlier_count': outliers.get('outlier_count', 0)
                }

            if self.test_results:
                significant_tests = [name for name, result in self.test_results.items()
                                     if result.get('p_value', 1) < 0.05]
                summary['key_findings']['statistical_tests'] = {
                    'total_tests': len(self.test_results),
                    'significant_results': len(significant_tests),
                    'significant_variables': significant_tests
                }

            return summary

        except Exception as e:
            logger.error(f"Erreur dans get_comprehensive_summary: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé simplifié des analyses statistiques"""
        try:
            return {
                'analyses_performed': list(self.results.keys()),
                'tests_performed': list(self.test_results.keys()),
                'total_analyses': len(self.results),
                'total_tests': len(self.test_results),
                'temporal_analysis': bool(self.temporal_results),
                'spatial_analysis': bool(self.spatial_results),
                'has_correlations': 'correlations' in self.results,
                'has_outliers': any(key.startswith('outlier') for key in self.results.keys())
            }
        except Exception as e:
            logger.error(f"Erreur dans get_statistics_summary: {e}")
            return {'error': str(e)}

    def export_results(self, output_dir: str = "outputs") -> Dict[str, str]:
        """Exporte les résultats statistiques"""
        from pathlib import Path
        import json

        export_files: Dict[str, str] = {}
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Corrélations
            if 'correlations' in self.results and isinstance(self.results['correlations'], pd.DataFrame) and not self.results['correlations'].empty:
                corr_file = output_path / "correlations.csv"
                self.results['correlations'].to_csv(corr_file, index=False)
                export_files['correlations'] = str(corr_file)

            # Analyses par groupe
            for key, result in self.results.items():
                if key.startswith('group_analysis_') and isinstance(result, pd.DataFrame):
                    var_name = key.replace('group_analysis_', '')
                    group_file = output_path / f"group_analysis_{var_name}.csv"
                    result.to_csv(group_file, index=False)
                    export_files[f'group_{var_name}'] = str(group_file)

            # Résumé global
            summary_file = output_path / "statistics_summary.json"
            summary = self.get_comprehensive_summary()
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            export_files['summary'] = str(summary_file)

            logger.info(f"Résultats statistiques exportés: {len(export_files)} fichiers")

        except Exception as e:
            logger.error(f"Erreur lors de l'export des résultats: {e}")

        return export_files

    # --------- Helpers tailles d'effet / ordinale --------------------------------
    def _epsilon_squared_kw(self, H: float, k: int, N: int) -> float:
        """Taille d’effet epsilon² pour Kruskal–Wallis"""
        return (H - (k - 1)) / (N - 1) if N > 1 else float("nan")

    def _cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cliff’s delta O(n log n)"""
        x = np.asarray(x)
        y = np.asarray(y)
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        i = j = 0
        gt = lt = 0
        nx, ny = len(x_sorted), len(y_sorted)
        while i < nx and j < ny:
            if x_sorted[i] > y_sorted[j]:
                gt += (nx - i)
                j += 1
            elif x_sorted[i] < y_sorted[j]:
                lt += (ny - j)
                i += 1
            else:
                xi = x_sorted[i]
                yi = y_sorted[j]
                cx = np.searchsorted(x_sorted, xi, side="right") - i
                cy = np.searchsorted(y_sorted, yi, side="right") - j
                i += cx
                j += cy
        return (gt - lt) / (nx * ny) if nx and ny else float("nan")

    # --------- Analyse dédiée RGA → Montant --------------------------------------
    def analyze_rga_vs_amount(
        self,
        df: pd.DataFrame,
        target_var: str = "montant_charge_brut",
        rga_var: str = "risque_rga_gcl",
    ) -> Optional[Dict[str, Any]]:
        """Analyse complète de l’impact du risque RGA (ordonné) sur le montant brut."""
        try:
            if rga_var not in df.columns or target_var not in df.columns:
                logger.warning(f"Colonnes manquantes: {rga_var}/{target_var}")
                return None

            d = df[[rga_var, target_var]].copy()
            d[target_var] = pd.to_numeric(d[target_var], errors="coerce")
            d[rga_var] = d[rga_var].astype(str).str.lower()
            d = d.dropna(subset=[rga_var, target_var])
            d = d[d[target_var] > 0]  # montants strictement positifs

            # Ordre métier
            order = ["faible", "moyen", "fort", "maximal"]
            d[rga_var] = pd.Categorical(d[rga_var], categories=order, ordered=True)
            d = d.dropna(subset=[rga_var])
            if d.empty:
                logger.warning("Aucune donnée valide après nettoyage pour RGA")
                return None

            # Winsorize léger (1–99%) pour stabiliser
            q1, q99 = d[target_var].quantile([0.01, 0.99])
            d[target_var] = d[target_var].clip(q1, q99)

            # Résumé par niveau
            grp = d.groupby(rga_var, observed=True)[target_var]
            summary = pd.DataFrame({
                "n": grp.size(),
                "median": grp.median(),
                "iqr": grp.quantile(0.75) - grp.quantile(0.25),
                "mean_trimmed": grp.apply(
                    lambda s: s.sort_values()
                              .iloc[int(0.05*len(s)) : max(int(0.95*len(s)), 1)]
                              .mean() if len(s) >= 20 else s.mean()
                ),
            }).reset_index()

            # Kruskal–Wallis (global)
            groups = [g.values for _, g in d.groupby(rga_var, observed=True)[target_var]]
            out: Dict[str, Any] = {"summary": summary}
            valid_groups = [g for g in groups if len(g) >= 3]
            if len(valid_groups) >= 2:
                H, p = stats.kruskal(*valid_groups, nan_policy="omit")
                out["kruskal"] = {
                    "H": float(H),
                    "p_value": float(p),
                    "epsilon2": float(self._epsilon_squared_kw(H, len(valid_groups), len(d))),
                }

            # Tendance ordinale (Spearman sur codes ordonnés)
            codes = d[rga_var].cat.codes  # 0..3
            rho, p_s = stats.spearmanr(codes, d[target_var], nan_policy="omit")
            out["trend"] = {
                "spearman_rho": float(rho),
                "p_value": float(p_s),
                "monotone_increase": bool((rho > 0) and (p_s < 0.05)),
            }

            # Post-hoc Dunn (si dispo)
            try:
                import scikit_posthocs as sp
                dunn = sp.posthoc_dunn(groups, p_adjust="holm")
                dunn.index = d[rga_var].cat.categories
                dunn.columns = d[rga_var].cat.categories
                out["dunn_matrix"] = dunn
            except Exception:
                out["dunn_matrix"] = None

            # Tailles d’effet par paire (Cliff’s delta)
            deltas: Dict[str, float] = {}
            cats = d[rga_var].cat.categories
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    a = d.loc[d[rga_var] == cats[i], target_var].values
                    b = d.loc[d[rga_var] == cats[j], target_var].values
                    if len(a) >= 3 and len(b) >= 3:
                        deltas[f"{cats[i]} vs {cats[j]}"] = float(self._cliffs_delta(a, b))
            out["cliffs_delta"] = deltas

            # Régressions quantiles (médiane & Q90), si statsmodels dispo
            try:
                import statsmodels.formula.api as smf
                d = d.copy()
                d["rga_ord"] = d[rga_var].cat.codes  # 0..3
                res50 = smf.quantreg(f"{target_var} ~ rga_ord", d).fit(q=0.5)
                res90 = smf.quantreg(f"{target_var} ~ rga_ord", d).fit(q=0.9)
                out["quantiles"] = {
                    "q50_slope": float(res50.params.get("rga_ord", np.nan)),
                    "q50_p": float(res50.pvalues.get("rga_ord", np.nan)),
                    "q90_slope": float(res90.params.get("rga_ord", np.nan)),
                    "q90_p": float(res90.pvalues.get("rga_ord", np.nan)),
                }
            except Exception:
                out["quantiles"] = None

            self.results["rga_vs_amount"] = out
            logger.info("Analyse RGA vs montant terminée")
            return out

        except Exception as e:
            logger.error(f"Erreur dans analyze_rga_vs_amount: {e}")
            return None

    # --------- Visualisation pratique (optionnelle) ------------------------------
    def plot_rga_vs_amount(self, df: pd.DataFrame,
                           target_var: str = "montant_charge_brut",
                           rga_var: str = "risque_rga_gcl",
                           logy: bool = True) -> Optional[plt.Figure]:
        """Box/violin plot par niveau RGA, prêt à être intégré au rapport."""
        try:
            if rga_var not in df.columns or target_var not in df.columns:
                logger.warning(f"Colonnes manquantes: {rga_var}/{target_var}")
                return None

            d = df[[rga_var, target_var]].copy()
            d[target_var] = pd.to_numeric(d[target_var], errors="coerce")
            d[rga_var] = d[rga_var].astype(str).str.lower()
            d = d.dropna(subset=[rga_var, target_var])
            d = d[d[target_var] > 0]

            order = ["faible", "moyen", "fort", "maximal"]
            d[rga_var] = pd.Categorical(d[rga_var], categories=order, ordered=True)
            d = d.dropna(subset=[rga_var])
            if d.empty:
                return None

            fig, ax = plt.subplots(figsize=(9, 5))
            sns.violinplot(data=d, x=rga_var, y=target_var, inner=None, cut=0, ax=ax)
            sns.boxplot(data=d, x=rga_var, y=target_var, whis=1.5, width=0.2, showcaps=True,
                        boxprops={'zorder': 2}, ax=ax)

            if logy and (d[target_var] > 0).any():
                ax.set_yscale('log')

            ax.set_xlabel("Risque RGA (faible → maximal)")
            ax.set_ylabel("Montant brut (log)" if logy else "Montant brut")
            ax.set_title("Montant brut par niveau de risque RGA")
            ax.grid(True, axis='y', alpha=0.25)

            self.figures['rga_vs_amount'] = fig
            return fig

        except Exception as e:
            logger.error(f"Erreur dans plot_rga_vs_amount: {e}")
            return None
        
    def analyze_binary_flag_vs_amount(
        self,
        df: pd.DataFrame,
        flag_var: str = "risk_retraitGonflementArgile",
        target_var: str = "montant_charge_brut",
    ) -> Optional[Dict[str, Any]]:
        try:
            if flag_var not in df.columns or target_var not in df.columns:
                logger.warning(f"Colonnes manquantes: {flag_var}/{target_var}")
                return None

            d = df[[flag_var, target_var]].copy()
            d[target_var] = pd.to_numeric(d[target_var], errors="coerce")

            def to_bool(x):
                if pd.isna(x): return np.nan
                if isinstance(x, (bool, np.bool_)): return bool(x)
                s = str(x).strip().lower()
                if s in {"1","true","vrai","oui","y","yes"}: return True
                if s in {"0","false","faux","non","n","no"}: return False
                try: return bool(int(s))
                except: return np.nan

            d[flag_var] = d[flag_var].map(to_bool)
            d = d.dropna(subset=[flag_var, target_var])
            d = d[d[target_var] > 0]
            if d.empty:
                logger.warning("Aucune donnée valide après nettoyage (binaire)")
                return None

            # Winsorize léger (1–99%)
            q1, q99 = d[target_var].quantile([0.01, 0.99])
            d[target_var] = d[target_var].clip(q1, q99)

            g0 = d.loc[d[flag_var] == False, target_var].values
            g1 = d.loc[d[flag_var] == True,  target_var].values

            out: Dict[str, Any] = {
                "summary": {
                    "n_false": int(len(g0)),
                    "n_true": int(len(g1)),
                    "median_false": float(np.median(g0)) if len(g0) else None,
                    "median_true": float(np.median(g1)) if len(g1) else None,
                    "iqr_false": float(np.quantile(g0, .75) - np.quantile(g0, .25)) if len(g0) else None,
                    "iqr_true": float(np.quantile(g1, .75) - np.quantile(g1, .25)) if len(g1) else None,
                    "mean_trimmed_false": float(np.mean(np.sort(g0)[int(.05*len(g0)):max(int(.95*len(g0)),1)])) if len(g0) >= 20 else (float(np.mean(g0)) if len(g0) else None),
                    "mean_trimmed_true":  float(np.mean(np.sort(g1)[int(.05*len(g1)):max(int(.95*len(g1)),1)])) if len(g1) >= 20 else (float(np.mean(g1)) if len(g1) else None),
                }
            }

            if len(g0) >= 3 and len(g1) >= 3:
                U, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
                out["mannwhitney"] = {"U": float(U), "p_value": float(p)}
                out["cliffs_delta"] = float(self._cliffs_delta(g1, g0))

                # t-test sur log(1+x) (indicatif)
                lg0, lg1 = np.log1p(g0), np.log1p(g1)
                t, p_t = stats.ttest_ind(lg1, lg0, equal_var=False)
                out["log_ttest"] = {"t": float(t), "p_value": float(p_t)}

                out["median_diff"] = float(np.median(g1) - np.median(g0))
                out["ratio_means"] = float(np.mean(g1) / np.mean(g0)) if np.mean(g0) > 0 else None

            self.results[f"binary_{flag_var}_vs_amount"] = out
            logger.info(f"Analyse binaire {flag_var} vs montant terminée")
            return out

        except Exception as e:
            logger.error(f"Erreur dans analyze_binary_flag_vs_amount: {e}")
            return None


    def plot_binary_flag_vs_amount(
        self,
        df: pd.DataFrame,
        flag_var: str = "risk_retraitGonflementArgile",
        target_var: str = "montant_charge_brut",
        logy: bool = True,
        figure_name: str = "binary_risk_vs_amount",
    ):
        try:
            if flag_var not in df.columns or target_var not in df.columns:
                return None

            d = df[[flag_var, target_var]].copy()
            d[target_var] = pd.to_numeric(d[target_var], errors="coerce")
            d = d.dropna(subset=[flag_var, target_var])
            d = d[d[target_var] > 0]
            if d.empty:
                return None

            # Normaliser l’affichage booléen
            d["_flag"] = d[flag_var].apply(lambda x: "Présent" if bool(x) else "Absent")

            fig, ax = plt.subplots(figsize=(9, 6))
            sns.boxplot(data=d, x="_flag", y=target_var, showfliers=False, ax=ax)
            sns.stripplot(data=d, x="_flag", y=target_var, alpha=0.25, jitter=0.25, dodge=True, ax=ax)
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Risque RGA (binaire)")
            ax.set_ylabel("Montant brut")
            ax.set_title("Montant brut selon présence du risque RGA (binaire)")

            self.figures[figure_name] = fig
            return fig
        except Exception as e:
            logger.error(f"Erreur dans plot_binary_flag_vs_amount: {e}")
            return None

def analyze_floors_vs_amount(
    self,
    df: pd.DataFrame,
    target_var: str = "montant_charge_brut",
    floors_var: str = "nb_etages_batiment",
) -> Optional[Dict[str, Any]]:
    """Analyse du montant vs nombre d’étages (agrégats, tendance, tests, tailles d’effet)."""
    try:
        if floors_var not in df.columns or target_var not in df.columns:
            logger.warning(f"Colonnes manquantes: {floors_var}/{target_var}")
            return None

        d = pd.DataFrame({
            floors_var: pd.to_numeric(df[floors_var], errors="coerce"),
            target_var: pd.to_numeric(df[target_var], errors="coerce"),
        }).dropna()

        # Montants > 0 et filtre simple sur nb d’étages
        d = d[(d[target_var] > 0) & (d[floors_var] >= 0) & (d[floors_var] <= 30)]
        if d.empty:
            logger.warning("Aucune donnée valide après nettoyage pour nb étages")
            return None

        # Arrondir/projeter sur des entiers
        d[floors_var] = d[floors_var].round().astype(int)

        # Winsorize léger (1–99%) pour stabiliser
        q1, q99 = d[target_var].quantile([0.01, 0.99])
        d[target_var] = d[target_var].clip(q1, q99)

        # --- Résumé par nombre d’étages
        def trimmed_mean(s: pd.Series) -> float:
            s = s.dropna().sort_values()
            n = len(s)
            if n < 20:
                return float(s.mean())
            a, b = int(0.05*n), int(0.95*n)
            b = max(b, a+1)
            return float(s.iloc[a:b].mean())

        grp = d.groupby(floors_var, observed=True)[target_var]
        summary = pd.DataFrame({
            "n": grp.size(),
            "median": grp.median(),
            "iqr": grp.quantile(0.75) - grp.quantile(0.25),
            "mean_trimmed": grp.apply(trimmed_mean),
            "mean": grp.mean(),
            "std": grp.std()
        }).reset_index().rename(columns={floors_var: "floors"})

        # --- Test global (Kruskal–Wallis) si assez de données
        groups = [g.values for _, g in d.groupby(floors_var, observed=True)[target_var]]
        out: Dict[str, Any] = {"summary": summary}
        valid_groups = [g for g in groups if len(g) >= 3]
        if len(valid_groups) >= 2:
            H, p = stats.kruskal(*valid_groups, nan_policy="omit")
            out["kruskal"] = {
                "H": float(H),
                "p_value": float(p),
                "epsilon2": float(self._epsilon_squared_kw(H, len(valid_groups), len(d))),
            }

        # --- Tendance monotone (Spearman)
        rho, p_s = stats.spearmanr(d[floors_var].values, d[target_var].values, nan_policy="omit")
        out["trend"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_s),
            "monotone_increase": bool((rho > 0) and (p_s < 0.05)),
        }

        # --- Tailles d’effet (Cliff’s delta) entre étages adjacents
        deltas: Dict[str, float] = {}
        levels = sorted(summary["floors"].unique().tolist())
        for i in range(len(levels) - 1):
            a = d.loc[d[floors_var] == levels[i], target_var].values
            b = d.loc[d[floors_var] == levels[i+1], target_var].values
            if len(a) >= 3 and len(b) >= 3:
                deltas[f"{levels[i]} vs {levels[i+1]}"] = float(self._cliffs_delta(a, b))
        out["cliffs_delta_adjacent"] = deltas

        # --- Régression quantile (médiane & 90e pct) optionnelle
        try:
            import statsmodels.formula.api as smf
            d2 = d.copy()
            res50 = smf.quantreg(f"{target_var} ~ {floors_var}", d2).fit(q=0.5)
            res90 = smf.quantreg(f"{target_var} ~ {floors_var}", d2).fit(q=0.9)
            out["quantiles"] = {
                "q50_slope": float(res50.params.get(floors_var, np.nan)),
                "q50_p": float(res50.pvalues.get(floors_var, np.nan)),
                "q90_slope": float(res90.params.get(floors_var, np.nan)),
                "q90_p": float(res90.pvalues.get(floors_var, np.nan)),
            }
        except Exception:
            out["quantiles"] = None

        self.results["floors_vs_amount"] = out
        logger.info("Analyse nb étages vs montant terminée")
        return out

    except Exception as e:
        logger.error(f"Erreur dans analyze_floors_vs_amount: {e}")
        return None


def plot_floors_vs_amount(
    self,
    df: pd.DataFrame,
    target_var: str = "montant_charge_brut",
    floors_var: str = "nb_etages_batiment",
    figure_name: str = "floors_vs_amount",
    logy: bool = True,
):
    """Boxplot + nuage de points par bin d’étages, avec médianes"""
    try:
        need_cols = {floors_var, target_var}
        if not need_cols.issubset(df.columns):
            return None

        d = df[[floors_var, target_var]].copy()
        d[floors_var] = pd.to_numeric(d[floors_var], errors="coerce")
        d[target_var] = pd.to_numeric(d[target_var], errors="coerce")
        d = d.dropna(subset=[floors_var, target_var])
        d = d[d[target_var] > 0]
        if d.empty:
            return None

        q1, q99 = d[target_var].quantile([0.01, 0.99])
        d[target_var] = d[target_var].clip(q1, q99)

        edges = [ -0.5, 1.5, 2.5, 4.5, 7.5, np.inf ]
        labels = ["1", "2", "3-4", "5-7", "8+"]
        d["floors_bin"] = pd.cut(d[floors_var], bins=edges, labels=labels, right=True)
        if d["floors_bin"].nunique(dropna=True) < 2:
            try:
                d["floors_bin"] = pd.qcut(d[floors_var], q=min(4, d[floors_var].nunique()), duplicates="drop")
            except Exception:
                d["floors_bin"] = d[floors_var].astype(int).astype(str)

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=d, x="floors_bin", y=target_var, showfliers=False, ax=ax)
        sns.stripplot(data=d, x="floors_bin", y=target_var, alpha=0.25, jitter=0.25, dodge=True, ax=ax)

        # Ligne des médianes
        med = d.groupby("floors_bin", observed=True)[target_var].median()
        ax.plot(range(len(med)), med.values, marker="o")

        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Nombre d’étages (bins)")
        ax.set_ylabel("Montant brut")
        ax.set_title("Montant brut vs nombre d’étages (bins)")

        self.figures[figure_name] = fig
        return fig
    except Exception as e:
        logger.error(f"Erreur dans plot_floors_vs_amount: {e}")
        return None
