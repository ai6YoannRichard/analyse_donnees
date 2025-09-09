from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

class SynthesisManager:
    """Gestionnaire de synthèse et d'export des résultats"""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Créer les sous-dossiers
        for subdir in ['exports', 'reports', 'visualizations']:
            (self.output_dir / subdir).mkdir(exist_ok=True)
            
        self.reports: Dict[str, Any] = {}

    def generate_data_quality_report(self, data_loader, data_filter, preprocessor) -> Dict[str, Any]:
        """Génère le rapport de qualité des données"""
        from datetime import datetime

        def _safe_loader_summary(x):
            if hasattr(x, "get_data_summary"):
                return x.get_data_summary()
            if isinstance(x, dict):
                return x.get("data_summary", {})
            return {}

        def _safe_loader_metadata(x):
            if hasattr(x, "metadata"):
                return getattr(x, "metadata", {})
            if isinstance(x, dict):
                return x.get("metadata", {})
            return {}

        def _safe_filter_summary(x):
            if hasattr(x, "get_filter_summary"):
                return x.get_filter_summary()
            if isinstance(x, dict):
                # compat: accepte un dict déjà résumé
                return x.get("filters_detail", x)
            return {}

        def _safe_preproc_summary(x):
            if hasattr(x, "get_preprocessing_summary"):
                return x.get_preprocessing_summary()
            if isinstance(x, dict):
                return x.get("preprocessing_summary", x)
            return {}

        report = {
            "timestamp": datetime.now().isoformat(),
            "data_loading": {
                "summary": _safe_loader_summary(data_loader),
                "metadata": _safe_loader_metadata(data_loader),
            },
            "data_filtering": _safe_filter_summary(data_filter),
            "preprocessing": _safe_preproc_summary(preprocessor),
        }

        self.reports["data_quality"] = report
        logger.info("Rapport de qualité des données généré")
        return report

    def generate_statistical_report(self, stats_manager, target_variable: str) -> Dict[str, Any]:
        """Génère le rapport statistique"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_variable': target_variable,
            'summary': stats_manager.get_statistics_summary() if hasattr(stats_manager, 'get_statistics_summary') else {},
            'key_findings': {
                'top_correlations': [],
                'significant_tests': [],
                'outlier_summary': {}
            }
        }
        
        # Top corrélations
        if 'correlations' in stats_manager.results:
            corr_df = stats_manager.results['correlations']
            if not corr_df.empty:
                report['key_findings']['top_correlations'] = corr_df.head(5).to_dict('records')
        
        # Tests significatifs
        significant_tests = [
            test_name for test_name, result in stats_manager.test_results.items()
            if result.get('significant', False)
        ]
        report['key_findings']['significant_tests'] = significant_tests
        
        # Résumé outliers
        outlier_keys = [k for k in stats_manager.results.keys() if k.startswith('outliers_')]
        if outlier_keys:
            outlier_key = outlier_keys[0]
            report['key_findings']['outlier_summary'] = stats_manager.results[outlier_key]
        
        self.reports['statistical'] = report
        logger.info("Rapport statistique généré")
        return report

    def generate_business_insights(self, df: pd.DataFrame, target_variable: str, 
                                 key_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Génère les insights métier"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'dataset_overview': {
                'total_claims': len(df),
                'target_variable': target_variable,
                'time_period': self._get_time_period(df),
                'average_amount': float(pd.to_numeric(df[target_variable], errors='coerce').mean()) if target_variable in df.columns else 0
            },
            'risk_factors': key_findings.get('top_risk_factors', []),
            'recommendations': self._generate_recommendations(df, target_variable, key_findings)
        }
        
        self.reports['business_insights'] = insights
        logger.info("Insights métier générés")
        return insights

    def _get_time_period(self, df: pd.DataFrame) -> Dict[str, str]:
        """Extrait la période temporelle des données"""
        if 'date_sinistre' in df.columns:
            dates = pd.to_datetime(df['date_sinistre'], errors='coerce').dropna()
            if not dates.empty:
                return {
                    'start_date': dates.min().strftime('%Y-%m-%d'),
                    'end_date': dates.max().strftime('%Y-%m-%d'),
                    'duration_days': (dates.max() - dates.min()).days
                }
        return {}

    def _generate_recommendations(self, df: pd.DataFrame, target_variable: str, 
                                key_findings: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        # Recommandations basées sur les corrélations
        top_corr = key_findings.get('top_risk_factors', [])
        if top_corr:
            high_corr_vars = [item.get('variable', '') for item in top_corr[:3] 
                            if item.get('correlation', 0) > 0.3]
            if high_corr_vars:
                recommendations.append(
                    f"Surveiller particulièrement les variables {', '.join(high_corr_vars)} "
                    "qui montrent une forte corrélation avec les montants de sinistres."
                )
        
        # Recommandations sur les outliers
        if 'outliers_percent' in str(key_findings):
            recommendations.append(
                "Mettre en place un système de détection automatique des sinistres atypiques "
                "pour un traitement prioritaire."
            )
        
        # Recommandations temporelles
        if 'saison_sinistre' in df.columns:
            seasonal_impact = df.groupby('saison_sinistre')[target_variable].mean() if target_variable in df.columns else None
            if seasonal_impact is not None and not seasonal_impact.empty:
                peak_season = seasonal_impact.idxmax()
                recommendations.append(
                    f"Renforcer les mesures préventives durant la saison {peak_season} "
                    "où les montants de sinistres sont les plus élevés."
                )
        
        if not recommendations:
            recommendations.append("Continuer le monitoring des données pour identifier de nouveaux patterns.")
        
        return recommendations

    def export_to_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Exporte les données en CSV"""
        filepath = self.output_dir / "exports" / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        data.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Export CSV: {filepath}")
        return str(filepath)

    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename: str) -> str:
        """Exporte multiple DataFrames en Excel multi-onglets"""
        filepath = self.output_dir / "exports" / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                # Limiter le nom de l'onglet à 31 caractères (limite Excel)
                safe_sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        
        logger.info(f"Export Excel: {filepath}")
        return str(filepath)

    def export_synthesis_report(self) -> str:
        """Exporte le rapport de synthèse en JSON"""
        filepath = self.output_dir / "reports" / f"synthesis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.reports, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Rapport de synthèse exporté: {filepath}")
        return str(filepath)

    def generate_html_report(self) -> str:
        """Génère un rapport HTML basique"""
        filepath = self.output_dir / "reports" / f"html_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = self._create_html_content()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML généré: {filepath}")
        return str(filepath)

    def _create_html_content(self) -> str:
        """Crée le contenu HTML du rapport"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse des Sinistres</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Rapport d'Analyse des Sinistres</h1>
        """
        
        # Ajouter les sections du rapport
        for report_type, content in self.reports.items():
            html += f'<div class="section"><h2>{report_type.replace("_", " ").title()}</h2>'
            if isinstance(content, dict):
                for key, value in content.items():
                    if not isinstance(value, (dict, list)):
                        html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
            html += '</div>'
        
        html += """
        </body>
        </html>
        """
        
        return html