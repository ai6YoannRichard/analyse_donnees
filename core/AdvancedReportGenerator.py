from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)

class AdvancedReportGenerator:
    """Générateur de rapports avancés avec export multi-format"""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stockage des données et figures
        self.figures: Dict[str, plt.Figure] = {}
        self.figure_descriptions: Dict[str, str] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info(f"AdvancedReportGenerator initialisé - dossier: {self.output_dir}")
    
    def add_figure(self, name: str, figure: plt.Figure, description: str = ""):
        """Ajoute une figure au rapport"""
        self.figures[name] = figure
        self.figure_descriptions[name] = description
        logger.debug(f"Figure ajoutée: {name}")
    
    def add_analysis_results(self, section: str, data: Dict[str, Any]):
        """Ajoute des résultats d'analyse"""
        self.analysis_results[section] = data
        logger.debug(f"Résultats ajoutés pour la section: {section}")
    
    def _figure_to_base64(self, figure: plt.Figure) -> str:
        """Convertit une figure matplotlib en base64"""
        buffer = BytesIO()
        canvas = FigureCanvasAgg(figure)
        canvas.print_png(buffer)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        return image_base64
    
    def _save_figure_as_png(self, figure: plt.Figure, filepath: Path) -> str:
        """Sauvegarde une figure en PNG"""
        figure.savefig(filepath, dpi=300, bbox_inches='tight')
        return str(filepath)
    
    def export_html_report(self, title: str = "Rapport d'Analyse Avancé") -> str:
        """Exporte un rapport HTML complet"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5; 
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 0 10px rgba(0,0,0,0.1); 
                }}
                h1 {{ 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 10px; 
                }}
                h2 {{ 
                    color: #34495e; 
                    margin-top: 30px; 
                    border-left: 4px solid #3498db; 
                    padding-left: 15px; 
                }}
                .summary-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0; 
                }}
                .metric-card {{ 
                    background: #ecf0f1; 
                    padding: 20px; 
                    border-radius: 8px; 
                    text-align: center; 
                    border-left: 4px solid #3498db; 
                }}
                .metric-value {{ 
                    font-size: 2em; 
                    font-weight: bold; 
                    color: #2c3e50; 
                }}
                .metric-label {{ 
                    color: #7f8c8d; 
                    margin-top: 5px; 
                }}
                .figure-container {{ 
                    margin: 20px 0; 
                    text-align: center; 
                    background: white; 
                    padding: 15px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }}
                .figure-description {{ 
                    font-style: italic; 
                    color: #7f8c8d; 
                    margin-top: 10px; 
                }}
                .stats-table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                    font-size: 14px; 
                }}
                .stats-table th, .stats-table td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }}
                .stats-table th {{ 
                    background-color: #3498db; 
                    color: white; 
                }}
                .stats-table tr:hover {{ 
                    background-color: #f1f1f1; 
                }}
                .insights-list {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    border-left: 4px solid #28a745; 
                }}
                .insights-list li {{ 
                    margin: 10px 0; 
                    padding-left: 10px; 
                }}
                .timestamp {{ 
                    color: #6c757d; 
                    font-size: 0.9em; 
                    text-align: right; 
                    margin-top: 30px; 
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div class="timestamp">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</div>
        """
        
        # Section résumé
        if 'summary' in self.analysis_results:
            summary = self.analysis_results['summary']
            html_content += """
                <h2>📊 Résumé Exécutif</h2>
                <div class="summary-grid">
            """
            
            metrics = [
                ('total_records', 'Nombre de sinistres', ''),
                ('average_amount', 'Montant moyen', '€'),
                ('total_amount', 'Montant total', '€'),
            ]
            
            for key, label, unit in metrics:
                if key in summary:
                    value = f"{summary[key]:,}".replace(',', ' ')
                    html_content += f"""
                        <div class="metric-card">
                            <div class="metric-value">{value} {unit}</div>
                            <div class="metric-label">{label}</div>
                        </div>
                    """
            
            html_content += "</div>"
            
            # Points clés
            if 'key_points' in summary:
                html_content += """
                    <div class="insights-list">
                        <h3>🎯 Points Clés</h3>
                        <ul>
                """
                for point in summary['key_points']:
                    html_content += f"<li>{point}</li>"
                html_content += "</ul></div>"
        
        # Section figures
        if self.figures:
            html_content += "<h2>📈 Visualisations</h2>"
            for fig_name, figure in self.figures.items():
                img_base64 = self._figure_to_base64(figure)
                description = self.figure_descriptions.get(fig_name, "")
                
                html_content += f"""
                    <div class="figure-container">
                        <h3>{fig_name.replace('_', ' ').title()}</h3>
                        <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;">
                        <div class="figure-description">{description}</div>
                    </div>
                """
        
        # Section statistiques
        if 'statistics' in self.analysis_results:
            stats = self.analysis_results['statistics']
            html_content += "<h2>📋 Statistiques Détaillées</h2>"
            
            # Corrélations
            if 'correlations' in stats:
                html_content += """
                    <h3>🔗 Principales Corrélations</h3>
                    <table class="stats-table">
                        <tr><th>Variable</th><th>Corrélation</th></tr>
                """
                for var, corr in stats['correlations'].items():
                    html_content += f"<tr><td>{var}</td><td>{corr}</td></tr>"
                html_content += "</table>"
            
            # Tests statistiques
            if 'tests' in stats:
                html_content += """
                    <h3>🧪 Tests Statistiques</h3>
                    <table class="stats-table">
                        <tr><th>Test</th><th>Résultat</th></tr>
                """
                for test, result in stats['tests'].items():
                    html_content += f"<tr><td>{test}</td><td>{result}</td></tr>"
                html_content += "</table>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Sauvegarder le fichier
        html_path = self.output_dir / f"{title.replace(' ', '_').lower()}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML généré: {html_path}")
        return str(html_path)
    
    def export_json_report(self, title: str = "Rapport JSON") -> str:
        """Exporte un rapport JSON structuré"""
        report_data = {
            'metadata': {
                'title': title,
                'generated_at': datetime.now().isoformat(),
                'generator': 'AdvancedReportGenerator'
            },
            'analysis_results': self.analysis_results,
            'figures': {
                name: {
                    'description': self.figure_descriptions.get(name, ''),
                    'base64': self._figure_to_base64(fig)
                }
                for name, fig in self.figures.items()
            }
        }
        
        json_path = self.output_dir / f"{title.replace(' ', '_').lower()}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Rapport JSON généré: {json_path}")
        return str(json_path)
    
    def export_pdf_summary(self, title: str = "Résumé PDF") -> str:
        """Exporte un résumé PDF simple"""
        # Note: Pour une vraie génération PDF, il faudrait utiliser reportlab ou weasyprint
        # Ici, on génère un fichier texte comme placeholder
        
        content = f"{title}\n{'='*len(title)}\n\n"
        content += f"Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M')}\n\n"
        
        if 'summary' in self.analysis_results:
            summary = self.analysis_results['summary']
            content += "RÉSUMÉ EXÉCUTIF\n" + "-"*15 + "\n"
            
            for key, value in summary.items():
                if key != 'key_points':
                    content += f"{key.replace('_', ' ').title()}: {value}\n"
            
            if 'key_points' in summary:
                content += "\nPoints clés:\n"
                for i, point in enumerate(summary['key_points'], 1):
                    content += f"{i}. {point}\n"
        
        content += f"\nNombre de visualisations générées: {len(self.figures)}\n"
        
        txt_path = self.output_dir / f"{title.replace(' ', '_').lower()}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Résumé textuel généré: {txt_path}")
        return str(txt_path)
    
    def export_figures_separately(self) -> Dict[str, str]:
        """Exporte toutes les figures en fichiers PNG séparés"""
        figure_paths = {}
        
        for fig_name, figure in self.figures.items():
            png_path = self.output_dir / f"{fig_name}.png"
            self._save_figure_as_png(figure, png_path)
            figure_paths[fig_name] = str(png_path)
        
        logger.info(f"Figures exportées: {len(figure_paths)} fichiers PNG")
        return figure_paths
    
    def export_all_formats(self, base_title: str = "Rapport Complet") -> Dict[str, str]:
        """Exporte le rapport dans tous les formats disponibles"""
        export_paths = {}
        
        try:
            export_paths['html_report'] = self.export_html_report(f"{base_title} - HTML")
        except Exception as e:
            logger.error(f"Erreur export HTML: {e}")
        
        try:
            export_paths['json_report'] = self.export_json_report(f"{base_title} - JSON")
        except Exception as e:
            logger.error(f"Erreur export JSON: {e}")
        
        try:
            export_paths['summary_text'] = self.export_pdf_summary(f"{base_title} - Résumé")
        except Exception as e:
            logger.error(f"Erreur export résumé: {e}")
        
        try:
            figure_paths = self.export_figures_separately()
            export_paths.update(figure_paths)
        except Exception as e:
            logger.error(f"Erreur export figures: {e}")
        
        logger.info(f"Export multi-format terminé: {len(export_paths)} fichiers générés")
        return export_paths
    
    def clear_data(self):
        """Nettoie toutes les données stockées"""
        self.figures.clear()
        self.figure_descriptions.clear()
        self.analysis_results.clear()
        logger.info("Données du générateur nettoyées")
    
    def get_summary(self) -> Dict[str, int]:
        """Retourne un résumé de l'état du générateur"""
        return {
            'figures_count': len(self.figures),
            'analysis_sections': len(self.analysis_results),
            'output_dir': str(self.output_dir)
        }