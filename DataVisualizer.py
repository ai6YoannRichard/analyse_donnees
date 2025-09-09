# sinistres_analysis/core/DataVisualizer.py
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class DataVisualizer:
    """Gestionnaire des visualisations"""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        # Style Matplotlib/Seaborn avec fallback si indisponible
        try:
            plt.style.use(style)
        except Exception:
            logger.warning(f"Style '{style}' indisponible. Fallback sur 'default'.")
            plt.style.use("default")

        sns.set_theme(style="whitegrid")
        self.figsize = figsize
        self.plots_created: List[Dict[str, Any]] = []
        try:
            self.color_palette = sns.color_palette("husl", 10)
        except Exception:
            self.color_palette = None  # Matplotlib choisira les couleurs par défaut

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Colonnes manquantes: {missing}")

    @staticmethod
    def _maybe_save(fig: plt.Figure, save_path: Optional[str]) -> None:
        if not save_path: 
            save_path = "./outputs/"
            
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    def _log_plot(self, plot_description: str) -> None:
        self.plots_created.append({"plot": plot_description, "timestamp": datetime.now()})
        logger.info(f"Graphique créé: {plot_description}")

    # ------------------------------------------------------------------ #
    # 1) Distribution
    # ------------------------------------------------------------------ #
    def plot_distribution(
        self,
        df: pd.DataFrame,
        variable: str,
        log_transform: bool = False,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualise la distribution d'une variable (hist + boxplot)."""
        self._ensure_columns(df, [variable])

        series = pd.to_numeric(df[variable], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Aucune valeur numérique exploitable pour '{variable}'.")

        title_suffix = ""
        if log_transform and (series > 0).all():
            series = np.log(series)
            title_suffix = " (échelle log)"

        bins = min(50, max(10, int(series.nunique() ** 0.5) * 5))
        fig, axes = plt.subplots(1, 2, figsize=(max(self.figsize[0], 15), 6))

        # Histogramme
        axes[0].hist(series, bins=bins, alpha=0.8)
        axes[0].set_title(f"Distribution de {variable}{title_suffix}")
        axes[0].set_xlabel(variable)
        axes[0].set_ylabel("Fréquence")

        # Box plot
        axes[1].boxplot(series, vert=True, showfliers=True)
        axes[1].set_title(f"Box plot de {variable}{title_suffix}")
        axes[1].set_ylabel(variable)

        fig.tight_layout()
        self._maybe_save(fig, save_path)
        self._log_plot(f"Distribution: {variable}{title_suffix}")
        return fig

    # ------------------------------------------------------------------ #
    # 2) Matrice de corrélation
    # ------------------------------------------------------------------ #
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Visualise la matrice de corrélation (triangulaire inférieure masquée)."""
        if variables is None:
            variables = df.select_dtypes(include=[np.number]).columns.tolist()

        if not variables:
            raise ValueError("Aucune variable numérique fournie pour la matrice de corrélation.")

        # Conserver uniquement colonnes numériques parmi 'variables'
        num_vars = [v for v in variables if pd.api.types.is_numeric_dtype(df[v])]
        if len(num_vars) < 2:
            raise ValueError("Il faut au moins 2 variables numériques pour corréler.")

        corr = df[num_vars].corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Matrice de corrélation")
        fig.tight_layout()

        self._maybe_save(fig, save_path)
        self._log_plot("Matrice de corrélation")
        return fig

    # ------------------------------------------------------------------ #
    # 3) Comparaison par groupes
    # ------------------------------------------------------------------ #
    def plot_group_comparison(
        self,
        df: pd.DataFrame,
        group_variable: str,
        target_variable: str,
        plot_type: str = "boxplot",  # "boxplot" | "violin" | "bar"
        save_path: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> plt.Figure:
        """Compare la distribution de la cible par groupe."""
        self._ensure_columns(df, [group_variable, target_variable])

        # Nettoyage minimal
        y = pd.to_numeric(df[target_variable], errors="coerce")
        x = df[group_variable].astype("string")
        data = pd.DataFrame({group_variable: x, target_variable: y}).dropna()

        if data.empty:
            raise ValueError("Pas de données valides après nettoyage pour la comparaison par groupes.")

        # Option: limiter aux top_n groupes (par effectif)
        if top_n and top_n > 0:
            top_levels = data[group_variable].value_counts().nlargest(top_n).index
            data = data[data[group_variable].isin(top_levels)]

        fig, ax = plt.subplots(figsize=self.figsize)

        if plot_type == "boxplot":
            sns.boxplot(data=data, x=group_variable, y=target_variable, ax=ax)
        elif plot_type == "violin":
            sns.violinplot(data=data, x=group_variable, y=target_variable, ax=ax, cut=0)
        elif plot_type == "bar":
            group_means = data.groupby(group_variable, observed=True)[target_variable].mean().sort_values(ascending=False)
            group_means.plot(kind="bar", ax=ax)
            ax.set_ylabel(f"Moyenne de {target_variable}")
        else:
            raise ValueError("plot_type doit être 'boxplot', 'violin' ou 'bar'.")

        ax.set_title(f"{target_variable} par {group_variable}")
        ax.set_xlabel(group_variable)
        ax.set_ylabel(target_variable if plot_type != "bar" else ax.get_ylabel())
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        self._maybe_save(fig, save_path)
        self._log_plot(f"Comparaison: {group_variable} vs {target_variable} ({plot_type})")
        return fig

    # ------------------------------------------------------------------ #
    # 4) Nuage de points avec régression
    # ------------------------------------------------------------------ #
    def plot_scatter_with_regression(
        self,
        df: pd.DataFrame,
        x_variable: str,
        y_variable: str,
        hue_variable: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Nuage de points + droite de régression (linéaire)."""
        cols = [x_variable, y_variable] + ([hue_variable] if hue_variable else [])
        self._ensure_columns(df, [c for c in cols if c])

        x = pd.to_numeric(df[x_variable], errors="coerce")
        y = pd.to_numeric(df[y_variable], errors="coerce")
        data = pd.DataFrame({x_variable: x, y_variable: y})

        if hue_variable:
            data[hue_variable] = df[hue_variable].astype("string")

        data = data.dropna(subset=[x_variable, y_variable])
        if len(data) < 3:
            raise ValueError("Trop peu de points valides pour tracer une régression.")

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.scatterplot(data=data, x=x_variable, y=y_variable, hue=hue_variable, alpha=0.6, ax=ax)
        sns.regplot(data=data, x=x_variable, y=y_variable, scatter=False, color="red", ax=ax)

        ax.set_title(f"Relation {x_variable} - {y_variable}")
        fig.tight_layout()

        self._maybe_save(fig, save_path)
        self._log_plot(f"Scatter: {x_variable} vs {y_variable}")
        return fig

    # ------------------------------------------------------------------ #
    # 5) Dashboard
    # ------------------------------------------------------------------ #
    def create_dashboard(
        self,
        df: pd.DataFrame,
        target_variable: str,
        key_variables: List[str],
        save_path: Optional[str] = "./outputs/",
    ) -> plt.Figure:
        """Crée un dashboard simple (distribution cible + cartes par variables clés)."""
        self._ensure_columns(df, [target_variable])
        key_variables = [v for v in key_variables if v in df.columns]

        n_plots = 1 + len(key_variables)  # 1 pour la distribution de la cible
        n_cols = 2
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        # 1) Distribution de la cible
        tgt = pd.to_numeric(df[target_variable], errors="coerce").dropna()
        axes[0].hist(tgt, bins=min(50, max(10, int(tgt.nunique() ** 0.5) * 5)), alpha=0.8)
        axes[0].set_title(f"Distribution de {target_variable}")
        axes[0].set_xlabel(target_variable)
        axes[0].set_ylabel("Fréquence")

        # 2) Boucle sur variables clés
        for i, var in enumerate(key_variables, start=1):
            ax = axes[i]
            col = df[var]

            if pd.api.types.is_numeric_dtype(col):
                x = pd.to_numeric(col, errors="coerce")
                y = pd.to_numeric(df[target_variable], errors="coerce")
                tmp = pd.DataFrame({var: x, target_variable: y}).dropna()
                if tmp.empty:
                    ax.set_visible(False)
                    continue
                ax.scatter(tmp[var], tmp[target_variable], alpha=0.5)
                ax.set_title(f"{target_variable} vs {var}")
                ax.set_xlabel(var)
                ax.set_ylabel(target_variable)
            else:
                # Categorical → bar des moyennes
                y = pd.to_numeric(df[target_variable], errors="coerce")
                x = col.astype("string")
                tmp = pd.DataFrame({var: x, target_variable: y}).dropna()
                if tmp.empty:
                    ax.set_visible(False)
                    continue
                means = tmp.groupby(var, observed=True)[target_variable].mean().sort_values(ascending=False)
                means.plot(kind="bar", ax=ax)
                ax.set_title(f"{target_variable} moyen par {var}")
                ax.set_xlabel(var)
                ax.set_ylabel(f"Moyenne de {target_variable}")
                ax.tick_params(axis="x", rotation=45)

        # Masquer axes inutilisés
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        self._maybe_save(fig, save_path)
        self._log_plot("Dashboard complet")
        return fig

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des visualisations créées."""
        return {
            "plots_created": len(self.plots_created),
            "plot_types": [p["plot"] for p in self.plots_created],
        }

    def reset(self) -> None:
        """Réinitialise l'historique des graphiques."""
        self.plots_created.clear()
