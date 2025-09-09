# sinistres_analysis/core/DataPreprocessor.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class DataPreprocessor:
    """Gestionnaire du nettoyage et de la préparation des données"""

    def __init__(self):
        self.transformations_applied: List[Dict[str, Any]] = []
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    # --------------------------------------------------------------------- #
    # Entrée principale
    # --------------------------------------------------------------------- #
    def clean_data(
        self,
        df: pd.DataFrame,
        *,
        outlier_strategy: str = "clip",            # "clip" | "nan" | "none"
        outlier_q: Tuple[float, float] = (0.01, 0.99),
        impute_numeric: str = "median",            # "median" | "mean" | "none"
    ) -> pd.DataFrame:
        """Pipeline de nettoyage principal."""
        cleaned_df = df.copy()

        cleaned_df = self._clean_dates(cleaned_df)
        cleaned_df = self._clean_numeric_variables(
            cleaned_df,
            outlier_strategy=outlier_strategy,
            outlier_q=outlier_q,
            impute_numeric=impute_numeric,
        )
        cleaned_df = self._clean_categorical_variables(cleaned_df)
        cleaned_df = self._create_derived_variables(cleaned_df)

        return cleaned_df

    # --------------------------------------------------------------------- #
    # DATES
    # --------------------------------------------------------------------- #
    def _clean_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les variables de date, en les convertissant en datetime (NaT si invalide)."""
        date_columns = ["date_sinistre", "date_cloture_sinistre"]

        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                self._log_transformation(f"Conversion date: {col}")

        return df

    # --------------------------------------------------------------------- #
    # NUMÉRIQUES
    # --------------------------------------------------------------------- #
    def _clean_numeric_variables(
        self,
        df: pd.DataFrame,
        *,
        outlier_strategy: str = "clip",            # "clip" | "nan" | "none"
        outlier_q: Tuple[float, float] = (0.01, 0.99),
        impute_numeric: str = "median",            # "median" | "mean" | "none"
    ) -> pd.DataFrame:
        """Nettoie les variables numériques: coercition, gestion des outliers, imputation."""
        # Coercition douce: convertit les colonnes totalement numériques en nombre
        df = df.copy()
        df = df.apply(lambda s: pd.to_numeric(s, errors="ignore") if s.dtype == "object" else s)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        q_low, q_high = outlier_q
        for col in numeric_cols:
            s = df[col]

            # Si trop peu de valeurs non-nulles, on saute
            if s.notna().sum() < 5:
                continue

            # Gestion des outliers
            if outlier_strategy in {"clip", "nan"}:
                q1 = s.quantile(q_low)
                q3 = s.quantile(q_high)
                if pd.notna(q1) and pd.notna(q3) and q3 >= q1:
                    if outlier_strategy == "clip":
                        df[col] = s.clip(lower=q1, upper=q3)
                        self._log_transformation(f"Outliers clip [{q_low:.2%},{q_high:.2%}]: {col}")
                    else:  # "nan"
                        mask = (s < q1) | (s > q3)
                        if mask.any():
                            df.loc[mask, col] = np.nan
                            self._log_transformation(f"Outliers -> NaN [{q_low:.2%},{q_high:.2%}]: {col}")

            # Imputation
            if impute_numeric in {"median", "mean"} and df[col].isna().any():
                if impute_numeric == "median":
                    val = df[col].median()
                    strat = "médiane"
                else:
                    val = df[col].mean()
                    strat = "moyenne"
                df[col] = df[col].fillna(val)
                self._log_transformation(f"Imputation {strat}: {col}")

        return df

    # --------------------------------------------------------------------- #
    # CATÉGORIELLES
    # --------------------------------------------------------------------- #
    def _clean_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les variables catégorielles: NA -> 'Inconnu', trim + uppercase."""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            df[col] = df[col].astype("string")
            df[col] = df[col].fillna("Inconnu")
            df[col] = df[col].str.strip().str.upper()
            self._log_transformation(f"Nettoyage texte: {col}")

        return df

    # --------------------------------------------------------------------- #
    # FEATURES DÉRIVÉES
    # --------------------------------------------------------------------- #
    def _create_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des variables dérivées utiles à l’analyse."""
        df = df.copy()
        now_year = datetime.now().year

        # Âge du bâtiment
        if "annee_construction_batiment" in df.columns:
            years = pd.to_numeric(df["annee_construction_batiment"], errors="coerce")
            age = (now_year - years).clip(lower=0)
            df["age_batiment"] = age
            self._log_transformation("Création: age_batiment")

        # Délai de clôture (jours)
        if {"date_sinistre", "date_cloture_sinistre"}.issubset(df.columns):
            delay = (df["date_cloture_sinistre"] - df["date_sinistre"]).dt.days
            df["delai_cloture_jours"] = delay
            self._log_transformation("Création: delai_cloture_jours")

        # Année et saison du sinistre
        if "date_sinistre" in df.columns:
            df["annee_sinistre"] = df["date_sinistre"].dt.year
            month = df["date_sinistre"].dt.month
            season = pd.Series(index=df.index, dtype="string")
            season[(month >= 3) & (month <= 5)] = "PRINTEMPS"
            season[(month >= 6) & (month <= 8)] = "ETE"
            season[(month >= 9) & (month <= 11)] = "AUTOMNE"
            season[(month == 12) | (month <= 2)] = "HIVER"
            df["saison_sinistre"] = season
            self._log_transformation("Création: annee_sinistre, saison_sinistre")

        # Classes d'âge
        if "age_batiment" in df.columns:
            df["classe_age"] = pd.cut(
                df["age_batiment"],
                bins=[0, 20, 40, 60, np.inf],
                labels=["RÉCENT", "MODERNE", "ANCIEN", "TRÈS ANCIEN"],
                right=True,
                include_lowest=True,
            )
            self._log_transformation("Création: classe_age")

        # Classes de surface
        if "surface_police" in df.columns:
            surf = pd.to_numeric(df["surface_police"], errors="coerce")
            df["classe_surface"] = pd.cut(
                surf,
                bins=[0, 80, 120, 180, np.inf],
                labels=["PETITE", "MOYENNE", "GRANDE", "TRÈS GRANDE"],
                right=True,
                include_lowest=True,
            )
            self._log_transformation("Création: classe_surface")

        # Code département (à partir du code postal)
        if "cpostal" in df.columns:
            cpos = df["cpostal"].astype("string").str.strip()
            df["region_code"] = cpos.str.zfill(5).str[:2]
            self._log_transformation("Création: region_code")

        return df

    # --------------------------------------------------------------------- #
    # ENCODAGE
    # --------------------------------------------------------------------- #
    def encode_categorical_variables(
        self,
        df: pd.DataFrame,
        *,
        method: str = "label",          # "label" | "onehot"
        drop_original: bool = False,
        onehot_min_freq: Optional[int] = None,
    ) -> pd.DataFrame:
        """Encode les variables catégorielles (label par défaut, ou one-hot)."""
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        encoded_df = df.copy()
        cat_cols = encoded_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if not cat_cols:
            return encoded_df

        if method == "label":
            # OrdinalEncoder gère les inconnus: unknown_value = -1
            if "label" not in self.encoders:
                self.encoders["label"] = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
                )
                self.encoders["label_cols"] = cat_cols.copy()
                encoded_vals = self.encoders["label"].fit_transform(encoded_df[cat_cols].astype("string"))
                self._log_transformation(f"Fit OrdinalEncoder sur {len(cat_cols)} colonnes")
            else:
                # Si on a déjà fit, on restreint aux colonnes apprises
                cat_cols = [c for c in cat_cols if c in self.encoders.get("label_cols", [])]
                encoded_vals = self.encoders["label"].transform(encoded_df[cat_cols].astype("string"))
                self._log_transformation(f"Transform OrdinalEncoder sur {len(cat_cols)} colonnes")

            # Colonnes encodées en suffixe _enc
            for i, col in enumerate(cat_cols):
                try:
                    encoded_df[f"{col}_encoded"] = encoded_vals[:, i].astype("Int64")
                except TypeError:
                    # Fallback pour les anciennes versions de pandas
                    encoded_df[f"{col}_encoded"] = encoded_vals[:, i].astype(int)

            if drop_original:
                encoded_df.drop(columns=cat_cols, inplace=True)

            return encoded_df

        elif method == "onehot":
            # handle_unknown="ignore" => colonnes inconnues ignorées au transform
            if "onehot" not in self.encoders:
                kwargs = dict(handle_unknown="ignore", sparse=False)
                # compatibilité scikit-learn >=1.2
                try:
                    self.encoders["onehot"] = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                except TypeError:
                    self.encoders["onehot"] = OneHotEncoder(**kwargs)

                if onehot_min_freq is not None:
                    # Paramètre disponible sur versions récentes
                    try:
                        self.encoders["onehot"].min_frequency = onehot_min_freq  # type: ignore
                    except Exception:
                        pass

                self.encoders["onehot_cols"] = cat_cols.copy()
                arr = self.encoders["onehot"].fit_transform(encoded_df[cat_cols].astype("string"))
                self._log_transformation(f"Fit OneHotEncoder sur {len(cat_cols)} colonnes")
            else:
                cat_cols = [c for c in cat_cols if c in self.encoders.get("onehot_cols", [])]
                arr = self.encoders["onehot"].transform(encoded_df[cat_cols].astype("string"))
                self._log_transformation(f"Transform OneHotEncoder sur {len(cat_cols)} colonnes")

            # Construit un DataFrame pour les colonnes OHE
            try:
                ohe_cols = self.encoders["onehot"].get_feature_names_out(cat_cols)
            except Exception:
                ohe_cols = self.encoders["onehot"].get_feature_names(cat_cols)
            ohe_df = pd.DataFrame(arr, columns=ohe_cols, index=encoded_df.index)

            if drop_original:
                encoded_df = pd.concat([encoded_df.drop(columns=cat_cols), ohe_df], axis=1)
            else:
                encoded_df = pd.concat([encoded_df, ohe_df], axis=1)

            return encoded_df

        else:
            raise ValueError("method doit être 'label' ou 'onehot'")

    # --------------------------------------------------------------------- #
    # SCALING (optionnel)
    # --------------------------------------------------------------------- #
    def scale_numeric_variables(self, df: pd.DataFrame, *, method: str = "standard") -> pd.DataFrame:
        """Normalise les variables numériques: 'standard' (z-score) ou 'minmax'."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        scaled_df = df.copy()
        num_cols = scaled_df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return scaled_df

        if method == "standard":
            scaler = StandardScaler()
            name = "standard"
        elif method == "minmax":
            scaler = MinMaxScaler()
            name = "minmax"
        else:
            raise ValueError("method doit être 'standard' ou 'minmax'.")

        if name not in self.scalers:
            self.scalers[name] = scaler.fit(scaled_df[num_cols])
            self._log_transformation(f"Fit scaler ({name}) sur {len(num_cols)} colonnes")
        else:
            scaler = self.scalers[name]
            self._log_transformation(f"Transform scaler ({name}) sur {len(num_cols)} colonnes")

        scaled_df[num_cols] = scaler.transform(scaled_df[num_cols])
        return scaled_df

    # --------------------------------------------------------------------- #
    # LOG & SUMMARY
    # --------------------------------------------------------------------- #
    def _log_transformation(self, transformation: str) -> None:
        """Log une transformation appliquée."""
        self.transformations_applied.append(
            {"transformation": transformation, "timestamp": datetime.now()}
        )
        logger.info(f"Transformation: {transformation}")

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Résumé du préprocessing."""
        print("inside get_preprocessing_summary")
        return {
            "transformations_count": len(self.transformations_applied),
            "transformations": [t["transformation"] for t in self.transformations_applied],
            "encoders_created": [k for k in self.encoders.keys() if not k.endswith("_cols")],
            "scalers_created": list(self.scalers.keys()),
        }
