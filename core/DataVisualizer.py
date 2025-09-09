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

    def plot_monthly_amounts_mmaaaa(
        self,
        df: pd.DataFrame,
        target: str = "montant_charge_brut",
        date_col: Optional[str] = "date_sinistre",
        month_col: Optional[str] = None,
        year_col: Optional[str] = None,
        agg: str = "sum",               # "sum" | "mean" | "median"
        min_count: int = 1,
        show_counts: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Trace l’évolution mensuelle sur plusieurs années au format étiquettes 'mmaaaa'.
        - Si `date_col` est fourni: construit la période mensuelle à partir de la date.
        - Sinon, utilise `month_col` (1..12) + `year_col` pour construire les périodes.
        """
        self._ensure_columns(df, [target])

        # 1) Construire la série mensuelle (PeriodIndex) puis l’étiquette 'mmaaaa'
        if date_col and date_col in df.columns:
            d = df[[date_col, target]].copy()
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            d[target] = pd.to_numeric(d[target], errors="coerce")
            d = d.dropna()
            d["periode"] = d[date_col].dt.to_period("M")
        else:
            if month_col is None or year_col is None:
                raise KeyError("Fournir soit `date_col`, soit `month_col` + `year_col`.")
            self._ensure_columns(df, [month_col, year_col])
            d = df[[month_col, year_col, target]].copy()
            d[month_col] = pd.to_numeric(d[month_col], errors="coerce")
            d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
            d[target] = pd.to_numeric(d[target], errors="coerce")
            d = d.dropna()
            d = d[(d[month_col] >= 1) & (d[month_col] <= 12)]
            # construit une date arbitraire au 1er du mois:
            d["periode"] = pd.PeriodIndex(pd.to_datetime(
                dict(year=d[year_col].astype(int), month=d[month_col].astype(int), day=1)
            ), freq="M")

        if d.empty:
            raise ValueError("Pas de données valides pour la série mensuelle.")

        d = d[d[target] > 0]

        # 2) Agrégats par mois
        aggfun = {"sum": "sum", "mean": "mean", "median": "median"}[agg]
        monthly = d.groupby("periode", observed=True)[target].agg([aggfun, "count"])
        monthly = monthly.rename(columns={aggfun: agg})
        monthly = monthly.sort_index()
        monthly["mmaaaa"] = monthly.index.to_timestamp().strftime("%m%Y")

        # Filtre d’effectif minimal
        monthly = monthly[monthly["count"] >= min_count]
        if monthly.empty:
            raise ValueError("Aucun mois ne respecte min_count.")

        # 3) Plot : ligne de l’agrégat + (option) barres d’effectifs
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(monthly["mmaaaa"], monthly[agg], marker="o", linewidth=2)
        ax.set_title(f"{target} ({agg}) par mois (format mmaaaa)")
        ax.set_xlabel("Mois (mmaaaa)")
        ax.set_ylabel(f"{agg} de {target}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        if show_counts:
            ax2 = ax.twinx()
            ax2.bar(monthly["mmaaaa"], monthly["count"], alpha=0.25)
            ax2.set_ylabel("N (observations)")

        fig.tight_layout()
        self._maybe_save(fig, save_path)
        self._log_plot(f"Série mensuelle {target} ({agg})")
        return fig

    # --- Série mensuelle agrégée (étiquettes mmAAAA) ---
    def plot_monthly_series(
        df: pd.DataFrame,
        date_col: str = "date_sinistre",
        target_col: str = "montant_charge_brut",
        agg: str = "sum",          # "sum", "mean", "median", ...
        rotate: int = 90,
        show_every: Optional[int] = None,  # ex: 2 pour afficher 1 tick sur 2
        title: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt

        d = df[[date_col, target_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d[target_col] = pd.to_numeric(d[target_col], errors="coerce")
        d = d.dropna(subset=[date_col, target_col])

        # Index mensuel (chronologique), puis agrégation
        d["ym"] = d[date_col].dt.to_period("M")
        monthly = getattr(d.groupby("ym")[target_col], agg)()

        # Réindexe pour inclure tous les mois entre min et max
        if len(monthly) > 0:
            full_idx = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
            monthly = monthly.reindex(full_idx, fill_value=0)

        # Etiquettes mmAAAA
        x_labels = monthly.index.strftime("%m%Y") if len(monthly) else []

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x_labels, monthly.values, marker="o")

        ax.set_xlabel("Mois (mmAAAA)")
        ax.set_ylabel(f"{target_col} ({agg})")
        ax.set_title(title or f"Montants mensuels – {target_col} ({agg})")
        ax.tick_params(axis="x", rotation=rotate)

        # Alléger les ticks si nécessaire (vise ~24 ticks max par défaut)
        step = max(1, int(show_every) if show_every else (len(x_labels) // 24 or 1))
        for i, lbl in enumerate(ax.get_xticklabels()):
            lbl.set_visible(i % step == 0)

        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        monthly_df = monthly.reset_index(name=target_col).rename(columns={"ym": "period"})
        monthly_df["mmAAAA"] = monthly_df["period"].dt.strftime("%m%Y")
        return fig, monthly_df[["period", "mmAAAA", target_col]]