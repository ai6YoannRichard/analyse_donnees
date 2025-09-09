from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64
from io import BytesIO
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .SynthesisManager import SynthesisManager
from .AdvancedReportGenerator import AdvancedReportGenerator

logger = logging.getLogger(__name__)

class IntegratedSynthesisManager(SynthesisManager):
    """Gestionnaire de synthèse intégré avec rapports avancés"""

    def __init__(self, output_dir: str = "./outputs"):
        super().__init__(output_dir)
        self.advanced_generator = AdvancedReportGenerator(output_dir)

    # ---------- Utils internes -------------------------------------------------
    def _extract_month_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Renvoie une Series (index de df) avec mois 1..12 selon les colonnes disponibles.
        Ordre de priorité: 'mois_exercice' (numérique), 'mois', 'date_exercice', 'date_sinistre'.
        """
        try:
            # Déjà un mois numérique 1..12 ?
            for c in ["mois_exercice", "mois"]:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce")
                    s = s.where((s >= 1) & (s <= 12))
                    if s.notna().any():
                        return s

            # Sinon, on essaie de déduire depuis une date
            for c in ["date_exercice", "date_sinistre"]:
                if c in df.columns:
                    s = pd.to_datetime(df[c], errors="coerce").dt.month
                    if s.notna().any():
                        return s

        except Exception as e:
            logger.error(f"_extract_month_series: {e}")

        return None

    # --- VISU cible RGA & NB_ETAGES -------------------------------------
    def _add_visuals_rga_etages(
        self,
        df: pd.DataFrame,
        target_var: str = "montant_charge_brut",
        rga_var: str = "risque_rga_gcl",
        etages_var: str = "nb_etages_batiment",
    ) -> None:
        sns.set_style("whitegrid")

        # A) Boxplot : montant vs RGA
        if rga_var in df.columns and target_var in df.columns:
            try:
                d = pd.DataFrame({
                    rga_var: df[rga_var].astype(str).str.lower(),
                    target_var: pd.to_numeric(df[target_var], errors="coerce")
                }).dropna()
                d = d[d[target_var] > 0]

                order = ["faible", "moyen", "fort", "maximal"]
                mapping = {
                    "faible": "faible", "bas": "faible",
                    "moyen": "moyen", "modere": "moyen", "modéré": "moyen",
                    "fort": "fort", "eleve": "fort", "élevé": "fort",
                    "tres eleve": "maximal", "très élevé": "maximal", "maximal": "maximal"
                }
                d[rga_var] = d[rga_var].map(mapping)
                d = d.dropna(subset=[rga_var])
                d[rga_var] = pd.Categorical(d[rga_var], categories=order, ordered=True)

                if not d.empty:
                    q1, q99 = d[target_var].quantile([0.01, 0.99])
                    d[target_var] = d[target_var].clip(q1, q99)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=d, x=rga_var, y=target_var, order=order, showfliers=False, ax=ax)
                    ax.set_title("Montant brut par niveau de risque RGA")
                    ax.set_xlabel("Risque RGA (GCL)")
                    ax.set_ylabel("Montant brut (€)")
                    self.advanced_generator.add_figure(
                        "boxplot_rga_vs_montant",
                        fig,
                        "Distribution des montants par niveau RGA (valeurs extrêmes tronquées au 1–99%)."
                    )
            except Exception as e:
                logger.error(f"Visu RGA vs montant - erreur: {e}")

        # B) Courbe médiane : montant vs nb étages
        if etages_var in df.columns and target_var in df.columns:
            try:
                d2 = pd.DataFrame({
                    etages_var: pd.to_numeric(df[etages_var], errors="coerce"),
                    target_var: pd.to_numeric(df[target_var], errors="coerce"),
                }).dropna()
                d2 = d2[(d2[target_var] > 0) & (d2[etages_var] >= 0) & (d2[etages_var] <= 20)]
                if not d2.empty:
                    q1, q99 = d2[target_var].quantile([0.01, 0.99])
                    d2[target_var] = d2[target_var].clip(q1, q99)

                    agg = (d2.groupby(etages_var, observed=True)[target_var]
                           .agg(count="count", median="median", mean="mean")
                           .reset_index())
                    agg = agg[agg["count"] >= 10].sort_values(etages_var)

                    if not agg.empty:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.lineplot(data=agg, x=etages_var, y="median", marker="o", ax=ax2)
                        ax2.set_title("Montant brut médian selon le nombre d’étages")
                        ax2.set_xlabel("Nombre d’étages du bâtiment")
                        ax2.set_ylabel("Montant brut médian (€)")

                        ax3 = ax2.twinx()
                        ax3.bar(agg[etages_var], agg["count"], alpha=0.25)
                        ax3.set_ylabel("N (observations)")

                        self.advanced_generator.add_figure(
                            "median_vs_nb_etages",
                            fig2,
                            "Médiane des montants par nombre d’étages (barres = effectifs, classes N≥10)."
                        )
            except Exception as e:
                logger.error(f"Visu nb_etages vs montant - erreur: {e}")

    # --- NOUVEAU : VISU montants vs mois d’exercice ---------------------
    def _add_visuals_montants_par_mois_exercice(
        self,
        df: pd.DataFrame,
        target_var: str = "montant_charge_brut",
        min_count: int = 12,
        fiscal_start_month: int = 1 
    ) -> None:
        """
        Produit 1–2 figures:
          - Ligne de la médiane (ou moyenne) par mois + barres des effectifs (axe droit).
          - (optionnel) Boxplots par mois si suffisamment d’observations.
        Et tente de pousser des tests via stats_manager si disponible dans le pipeline appelant.
        """
        # ... extrait 'mois_exercice_tmp' (1..12) comme avant ...
        # ordre fiscal (ici identique à 1..12)
        order = list(range(fiscal_start_month, 13)) + list(range(1, fiscal_start_month))
        agg = agg.set_index("mois_exercice_tmp").reindex(order).reset_index().rename(columns={"index":"mois_exercice_tmp"})
        try:
            mois = self._extract_month_series(df)
            if mois is None or target_var not in df.columns:
                return

            d = pd.DataFrame({
                "mois_exercice_tmp": pd.to_numeric(mois, errors="coerce"),
                target_var: pd.to_numeric(df[target_var], errors="coerce"),
            }).dropna()
            d = d[(d["mois_exercice_tmp"] >= 1) & (d["mois_exercice_tmp"] <= 12)]
            d = d[d[target_var] > 0]
            if d.empty:
                return

            # Winsorize léger
            q1, q99 = d[target_var].quantile([0.01, 0.99])
            d[target_var] = d[target_var].clip(q1, q99)

            # Agrégats par mois
            agg = (d.groupby("mois_exercice_tmp", observed=True)[target_var]
                   .agg(count="count", median="median", mean="mean", sum="sum")
                   .reset_index()
                   .sort_values("mois_exercice_tmp"))
            agg = agg[agg["count"] >= min_count]
            if agg.empty:
                return

            # Figure 1 : médiane par mois + effectifs
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=agg, x="mois_exercice_tmp", y="median", marker="o", ax=ax)
            ax.set_xticks(range(1, 13))
            ax.set_xlabel("Mois d’exercice")
            ax.set_ylabel("Montant brut médian (€)")
            ax.set_title("Montant brut par mois d’exercice (médiane)")

            ax2 = ax.twinx()
            ax2.bar(agg["mois_exercice_tmp"], agg["count"], alpha=0.25)
            ax2.set_ylabel("N (observations)")

            self.advanced_generator.add_figure(
                "montant_par_mois_exercice",
                fig,
                "Médiane des montants par mois d’exercice (barres = effectifs, N≥{min_count})."
            )

            # Figure 2 (optionnelle) : boxplots par mois si dispersion utile
            if d["mois_exercice_tmp"].nunique() >= 3:
                fig2, axb = plt.subplots(figsize=(11, 6))
                d["mois_exercice_tmp"] = d["mois_exercice_tmp"].astype(int)
                sns.boxplot(
                    data=d,
                    x="mois_exercice_tmp",
                    y=target_var,
                    showfliers=False,
                    ax=axb
                )
                axb.set_xticks(range(0, 12), [str(i) for i in range(1, 13)])
                axb.set_xlabel("Mois d’exercice")
                axb.set_ylabel("Montant brut (€)")
                axb.set_title("Distribution des montants par mois d’exercice")
                self.advanced_generator.add_figure(
                    "boxplot_mois_exercice_vs_montant",
                    fig2,
                    "Distribution (boxplots) des montants par mois d’exercice (valeurs extrêmes tronquées au 1–99%)."
                )

        except Exception as e:
            logger.error(f"Visu montants par mois exercice - erreur: {e}")

    # --- ANALYSE COMPLÈTE ------------------------------------------------
    def create_comprehensive_analysis(
        self,
        df: pd.DataFrame,
        target_variable: str,
        stats_manager,
        data_filter=None,
        preprocessor=None
    ):
        print("🔍 Démarrage de l'analyse complète...")

        # 1) Rapports de base
        df_arg = data_filter if data_filter is not None else {'filters_detail': []}
        pp_arg = preprocessor if preprocessor is not None else {'preprocessing_summary': {}}

        data_quality_report = self.generate_data_quality_report(
            data_loader={"data_summary": self._get_enhanced_data_summary(df)},
            data_filter=df_arg,
            preprocessor=pp_arg
        )
        statistical_report = self.generate_statistical_report(stats_manager, target_variable)
        business_insights = self.generate_business_insights(
            df, target_variable, statistical_report.get('key_findings', {})
        )

        # 2) Analyses ciblées (RGA / binaire / étages)
        rga_candidates = ['risque_rga_gcl', 'risque_rga_gcl_detail', 'rga', 'risque_rga']
        rga_var = next((c for c in rga_candidates if c in df.columns), None)

        if rga_var and target_variable in df.columns and hasattr(stats_manager, 'analyze_rga_vs_amount'):
            rga_results = stats_manager.analyze_rga_vs_amount(
                df, target_var=target_variable, rga_var=rga_var
            )
            if hasattr(stats_manager, 'plot_rga_vs_amount'):
                stats_manager.plot_rga_vs_amount(df, target_var=target_variable, rga_var=rga_var, logy=True)
            if rga_results:
                kr = rga_results.get('kruskal')
                tr = rga_results.get('trend')
                tests_patch = {}
                if kr:
                    tests_patch['RGA vs montant (Kruskal)'] = (
                        f"{'Significatif' if kr['p_value'] < 0.05 else 'Non significatif'} "
                        f"(p={kr['p_value']:.4f}, ε²={kr['epsilon2']:.3f})"
                    )
                if tr:
                    tests_patch['Tendance ordinale (Spearman)'] = (
                        f"rho={tr['spearman_rho']:.3f}, p={tr['p_value']:.4f}"
                    )
                stats_manager.results['rga_tests_patch'] = tests_patch

        if 'risk_retraitGonflementArgile' in df.columns and target_variable in df.columns:
            if hasattr(stats_manager, 'analyze_binary_flag_vs_amount'):
                bin_res = stats_manager.analyze_binary_flag_vs_amount(
                    df, flag_var='risk_retraitGonflementArgile', target_var=target_variable
                )
                if hasattr(stats_manager, 'plot_binary_flag_vs_amount'):
                    stats_manager.plot_binary_flag_vs_amount(
                        df, flag_var='risk_retraitGonflementArgile', target_var=target_variable, logy=True
                    )
                if bin_res:
                    patch = {}
                    mw = bin_res.get('mannwhitney')
                    if mw:
                        cd = bin_res.get('cliffs_delta', np.nan)
                        sig = "Significatif" if mw['p_value'] < 0.05 else "Non significatif"
                        patch['Risque RGA (binaire) vs montant (Mann–Whitney)'] = (
                            f"{sig} (p={mw['p_value']:.4f}, Δ_cliff={cd:.3f})"
                        )
                    stats_manager.results['binary_risk_tests_patch'] = patch

        if 'nb_etages_batiment' in df.columns and target_variable in df.columns:
            if hasattr(stats_manager, 'analyze_floors_vs_amount'):
                floors_res = stats_manager.analyze_floors_vs_amount(
                    df, target_var=target_variable, floors_var='nb_etages_batiment'
                )
            else:
                floors_res = None
            if hasattr(stats_manager, 'plot_floors_vs_amount'):
                stats_manager.plot_floors_vs_amount(
                    df, target_var=target_variable, floors_var='nb_etages_batiment', logy=True
                )
            if floors_res:
                patch = {}
                kr = floors_res.get('kruskal')
                tr = floors_res.get('trend')
                if kr:
                    sig = "Significatif" if kr['p_value'] < 0.05 else "Non significatif"
                    patch['Étages vs montant (Kruskal–Wallis)'] = (
                        f"{sig} (p={kr['p_value']:.4f}, ε²={kr['epsilon2']:.3f})"
                    )
                if tr:
                    m = "↑ monotone" if tr.get('monotone_increase') else "pas de tendance monotone"
                    patch['Tendance ordinale (Spearman)'] = (
                        f"rho={tr['spearman_rho']:.3f}, p={tr['p_value']:.4f} ({m})"
                    )
                stats_manager.results['floors_tests_patch'] = patch

        # 2.bis) Analyse montants vs mois d’exercice (générique + figure)
        try:
            mois = self._extract_month_series(df)
            if mois is not None and target_variable in df.columns:
                # On fabrique un petit DF pour l’analyse par groupes (1..12)
                df_m = pd.DataFrame({
                    "mois_exercice_tmp": pd.to_numeric(mois, errors="coerce"),
                    target_variable: pd.to_numeric(df[target_variable], errors="coerce")
                }).dropna()
                df_m = df_m[(df_m["mois_exercice_tmp"] >= 1) & (df_m["mois_exercice_tmp"] <= 12)]

                if hasattr(stats_manager, "analyze_distribution_by_groups") and not df_m.empty:
                    dist_mois = stats_manager.analyze_distribution_by_groups(
                        df=df_m,
                        target_var=target_variable,
                        group_var="mois_exercice_tmp",
                        order=list(range(1, 13)),
                        min_count=10,
                        positive_only=True,
                        winsor=(0.01, 0.99),
                        coerce_group_numeric=True
                    )
                    # On peut tenter un test global (Kruskal) sur les 12 mois si dispo
                    if hasattr(stats_manager, "statistical_tests") and not dist_mois.empty:
                        test_res = stats_manager.statistical_tests(
                            df=df_m.assign(mois_exercice_tmp=df_m["mois_exercice_tmp"].astype(int).astype(str)),
                            cat_var="mois_exercice_tmp",
                            target_var=target_variable
                        )
                        if test_res:
                            stats_manager.results['month_tests_patch'] = {
                                "Mois d’exercice vs montant (Kruskal–Wallis)":
                                    f"{'Significatif' if test_res.get('p_value', 1) < 0.05 else 'Non significatif'} "
                                    f"(p={test_res.get('p_value', 1):.4f})"
                            }

                # Figures dédiées
                self._add_visuals_montants_par_mois_exercice(df, target_variable, min_count=10)

        except Exception as e:
            logger.error(f"Analyse mois d’exercice - erreur: {e}")

        # 3) Préparer données pour le rapport
        self._prepare_advanced_report_data(df, target_variable, stats_manager)

        # Injecter les patches de tests éventuels dans la section “tests”
        stats_data_tests = {}
        for k in ["binary_risk_tests_patch", "rga_tests_patch", "floors_tests_patch", "month_tests_patch"]:
            if k in stats_manager.results:
                stats_data_tests.update(stats_manager.results[k])
        if stats_data_tests:
            # self._prepare_advanced_report_data s’occupe déjà des tests,
            # mais on renforce l’inclusion si besoin :
            stats = self.advanced_generator.analysis_results.get("statistics", {})
            tests = (stats.get("tests") or {}) | stats_data_tests
            self.advanced_generator.analysis_results["statistics"] = {**stats, "tests": tests}

        # 4) Ajouter les figures (si le stats_manager en a produit)
        self._add_statistical_figures(stats_manager)

        # 5) Visualisations ciblées (RGA & NB_ETAGES)
        self._add_visuals_rga_etages(df, target_variable)

        # 6) Export multi-format + index
        report_paths = self.advanced_generator.export_all_formats(
            f"Analyse Complète - {target_variable}"
        )
        index_path = self._create_reports_index(report_paths)

        return {
            'reports': report_paths,
            'index': index_path,
            'data_quality': data_quality_report,
            'statistical': statistical_report,
            'business': business_insights
        }
