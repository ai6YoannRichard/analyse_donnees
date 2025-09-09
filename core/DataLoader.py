# sinistres_analysis/core/DataLoader.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class DataLoader:
    """Gestionnaire de chargement des données depuis diverses sources"""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config: Dict[str, Any] = self._load_config(config_file) if config_file else {}
        self.raw_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    # Utils
    def _load_config(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Charge la configuration depuis un fichier JSON (optionnel)."""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Config non chargée ({config_file}): {e}")
            return {}

    @staticmethod
    def _detect_encoding(path: Path, sample_bytes: int = 2048) -> str:
        """Détection simple d'encodage: UTF-8, sinon Latin-1, sinon CP1252."""
        try:
            with path.open("rb") as f:
                raw = f.read(sample_bytes)
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    raw.decode(enc)
                    return enc
                except Exception:
                    continue
        except Exception:
            pass
        return "utf-8"

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie légèrement les noms de colonnes (espaces/retours)."""
        df = df.copy()
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        return df

    def _extract_metadata(self) -> None:
        """Extrait les métadonnées du dataset (sûres pour JSON)."""
        if self.raw_data is None:
            return
        dtypes_map = {str(k): str(v) for k, v in self.raw_data.dtypes.items()}
        self.metadata = {
            "nb_rows": int(len(self.raw_data)),
            "nb_columns": int(len(self.raw_data.columns)),
            "columns": list(map(str, self.raw_data.columns)),
            "dtypes": dtypes_map,
            "memory_usage": int(self.raw_data.memory_usage(deep=True).sum()),
            "load_timestamp": datetime.now().isoformat(),
        }

    # Loaders
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Charge un fichier CSV ou Excel selon l'extension.
        Exemples:
            load("data.csv", parse_dates=["date"])
            load("data.xlsx", sheet_name=0, parse_dates=["date"])
        """
        path = Path(file_path)
        print(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path.resolve()}")

        ext = path.suffix.lower()
        print(ext)
        if ext == ".csv":
            return self.load_from_csv(path, **kwargs)
        elif ext in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
            return self.load_from_excel(path, **kwargs)
        else:
            raise ValueError(
                f"Extension non supportée: {ext}. "
                "Formats acceptés: .csv, .xls, .xlsx, .xlsm, .xlsb"
            )
        
    #  CSV 
    def load_from_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge les données depuis un CSV avec options robustes."""
        path = Path(file_path)
        logger.info(f"Chargement des données depuis {path}")

        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path.resolve()}")

        # Paramètres par défaut pour CSV
        csv_defaults = {
            "encoding": self._detect_encoding(path),
            "sep": None,
            "engine": "python", 
            "na_values": ["", "NA", "N/A", "NaN", "NULL", "null"],
            "compression": "infer",
        }
        
        # Fusion avec config et kwargs
        read_kwargs = csv_defaults.copy()
        read_kwargs.update(self.config.get("csv_read_params", {}))
        read_kwargs.update(kwargs)

        try:
            df = pd.read_csv(path, **read_kwargs)
            df = self._normalize_columns(df)
            
            # Parse dates si spécifié
            if "parse_dates" in read_kwargs and read_kwargs["parse_dates"]:
                for col in read_kwargs["parse_dates"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

            self.raw_data = df
            self._extract_metadata()
            logger.info(f"✅ {len(df)} lignes chargées")
            return df

        except Exception as e:
            logger.error(f"❌ Erreur CSV: {e}")
            raise

    def load_from_excel(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Charge les données depuis Excel."""
        path = Path(file_path)
        logger.info(f"Chargement Excel depuis {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {path.resolve()}")

        # Paramètres par défaut pour Excel
        excel_defaults = {
            "sheet_name": 0,
            "na_values": ["", "NA", "N/A", "NaN", "NULL", "null"],
        }
        
        # Fusion avec config et kwargs
        read_kwargs = excel_defaults.copy()
        read_kwargs.update(self.config.get("excel_read_params", {}))
        read_kwargs.update(kwargs)

        try:
            df = pd.read_excel(path, **read_kwargs)
            df = self._normalize_columns(df)
            
            # Parse dates si spécifié
            if "parse_dates" in read_kwargs and read_kwargs["parse_dates"]:
                for col in read_kwargs["parse_dates"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

            self.raw_data = df
            self._extract_metadata()
            logger.info(f"✅ {len(df)} lignes chargées")
            return df

        except Exception as e:
            logger.error(f"❌ Erreur Excel: {e}")
            raise

    #  Database (optionnel)
    def load_from_database(self, connection_string: str, query: str, **kwargs) -> pd.DataFrame:
        """Charge les données depuis une base SQL via SQLAlchemy (optionnel)."""
        try:
            from sqlalchemy import create_engine  # type: ignore
        except Exception as e:
            raise ImportError(
                "SQLAlchemy est requis pour load_from_database. Installe-le: pip install SQLAlchemy"
            ) from e

        logger.info("Chargement des données depuis la base de données…")
        try:
            engine = create_engine(connection_string)
            df = pd.read_sql(query, engine, **kwargs)
            df = self._normalize_columns(df)
            self.raw_data = df
            self._extract_metadata()
            logger.info(f"✅ {len(self.raw_data)} lignes chargées depuis la base")
            return self.raw_data
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement DB: {e}")
            raise

    # --------------------------------------------------------------- Summary
    def get_data_summary(self) -> Dict[str, Any]:
        """Résumé synthétique des données chargées (toujours JSON-safe)."""
        if self.raw_data is None:
            return {"status": "No data loaded"}

        types_count = self.raw_data.dtypes.value_counts()
        data_types = {str(k): int(v) for k, v in types_count.items()}

        return {
            "shape": tuple(map(int, self.raw_data.shape)),
            "columns": int(len(self.raw_data.columns)),
            "memory_mb": round(self.metadata.get("memory_usage", 0) / 1024**2, 2),
            "missing_values": int(self.raw_data.isnull().sum().sum()),
            "data_types": data_types,
        }
