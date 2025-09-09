from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

import pandas as pd

import sys
import os

# Ajouter le répertoire parent si nécessaire
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Imports des modules du projet
try:
    from core.DataLoader import DataLoader
    from core.DataFilter import DataFilter
    from core.DataPreprocessor import DataPreprocessor
    from core.DataVisualizer import DataVisualizer
    from core.VariableManager import VariableManager
    from core.ModelingManager import ModelingManager
    from core.StatisticsManager import StatisticsManager
    from core.SynthesisManager import SynthesisManager
    from core.AdvancedReportGenerator import AdvancedReportGenerator
    from core.IntegratedSynthesisManager import IntegratedSynthesisManager
except ImportError:
    # Fallback pour les imports relatifs si les absolus échouent
    from .DataLoader import DataLoader
    from .DataFilter import DataFilter
    from .DataPreprocessor import DataPreprocessor
    from .DataVisualizer import DataVisualizer
    from .VariableManager import VariableManager
    from .ModelingManager import ModelingManager
    from .StatisticsManager import StatisticsManager
    from .SynthesisManager import SynthesisManager
    from .AdvancedReportGenerator import AdvancedReportGenerator
    from .IntegratedSynthesisManager import IntegratedSynthesisManager

logger = logging.getLogger(__name__)

class SinistreAnalysisPipeline:
    """Pipeline principal d'analyse des sinistres avec rapports avancés"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        
        # Initialisation des composants EXISTANTS
        self.data_loader = DataLoader()
        self.data_filter = DataFilter(config.get('filter_params', {}))
        self.preprocessor = DataPreprocessor()
        self.variable_manager = VariableManager()        
        self.stats_manager = StatisticsManager(config.get('stats_params', {}))
        self.visualizer = DataVisualizer()        
        
        output_dir = config.get('output_dir', './outputs')
        self.synthesis_manager = IntegratedSynthesisManager(output_dir)        
        
        self.generate_advanced_reports = config.get('generate_advanced_reports', True)
        
        logger.info("Pipeline d'analyse initialisé avec rapports avancés")
    
    # ====================================================================
    # GARDEZ TOUTES VOS MÉTHODES EXISTANTES SANS MODIFICATION
    # ====================================================================
    
    def load_data(self, file_path: str, **kwargs) -> 'SinistreAnalysisPipeline':
        """Charge les données"""
        logger.info("Étape 1: Chargement des données")
        self.raw_data = self.data_loader.load(file_path, **kwargs)
        logger.info(f"Données chargées: {self.raw_data.shape}")
        return self
    
    def filter_data(self, apply_quality: bool = True, 
                   apply_business: bool = False, **kwargs) -> 'SinistreAnalysisPipeline':
        """Filtre les données"""
        logger.info("Étape 2: Filtrage des données")
        
        if self.raw_data is None:
            raise ValueError("Aucune donnée chargée. Exécuter load_data() d'abord.")
        
        filtered_data = self.raw_data.copy()
        
        if apply_quality:
            filtered_data = self.data_filter.apply_quality_filters(filtered_data)
        
        if apply_business:
            filtered_data = self.data_filter.apply_business_filters(filtered_data, **kwargs)
        
        self.processed_data = filtered_data
        logger.info(f"Données après filtrage: {self.processed_data.shape}")
        return self     
    
    def setup_variables(self, target_variable: str) -> 'SinistreAnalysisPipeline':
        """Configure les variables d'analyse"""
        logger.info("Étape 3: Configuration des variables")
        
        self.variable_manager.define_target(target_variable)
        self.variable_manager.define_feature_groups()
        
        if self.processed_data is not None and target_variable not in self.processed_data.columns:
            raise KeyError(f"La variable cible '{target_variable}' est absente des données.")
        
        return self
    
    def preprocess_data(self, apply_encoding: bool = True) -> 'SinistreAnalysisPipeline':
        """Préprocesse les données : Nettoyage & encodage"""
        logger.info("Étape 4: Préprocessing des données :Nettoyage & encodage")
        
        if self.processed_data is None:
            raise ValueError("Aucune donnée filtrée. Exécuter filter_data() d'abord.")
        
        self.processed_data = self.preprocessor.clean_data(self.processed_data)
        
        if apply_encoding:
            self.processed_data = self.preprocessor.encode_categorical_variables(self.processed_data)
        
        return self
    
    def analyze_statistics(self, correlation_threshold: float = 0.1) -> 'SinistreAnalysisPipeline':
        """Effectue les analyses statistiques"""
        logger.info("Étape 5: Analyses statistiques")
        
        target_var = self.variable_manager.target_variable['name']
        
        # Statistiques descriptives
        self.stats_manager.descriptive_statistics(self.processed_data)
        
        # Analyse des corrélations
        self.stats_manager.correlation_analysis(self.processed_data, target_var)
        
        # Analyses par groupe pour variables catégorielles importantes
        categorical_vars = ['risque_rga_gcl', 'classe_age', 'classe_surface']
        for var in categorical_vars:
            if var in self.processed_data.columns:
                self.stats_manager.group_analysis(self.processed_data, var, target_var)
                self.stats_manager.statistical_tests(self.processed_data, var, target_var)
        
        # Détection d'outliers
        self.stats_manager.outlier_detection(self.processed_data, target_var)
        
        return self    
    
    def create_monthly_visualization(self) -> 'SinistreAnalysisPipeline':
        """Crée le graphique mensuel des montants"""
        if self.processed_data is not None:
            try:
                self.visualizer.plot_monthly_amounts_mmaaaa(
                    df=self.processed_data,
                    target=self.config.get("target_variable", "montant_charge_brut"),
                    date_col="date_sinistre",        # ou month_col="mois_exercice", year_col="annee_exercice"
                    agg="sum",                       # ou "mean"/"median"
                    min_count=5,
                    show_counts=True,
                    save_path="./outputs/fig_monthly_mmaaaa.png",
                )
                logger.info("Graphique mensuel créé avec succès")
            except Exception as e:
                logger.warning(f"Impossible de créer le graphique mensuel: {e}")
        else:
            logger.warning("Données non disponibles pour le graphique mensuel")
        
        return self


    def create_visualizations(self, key_variables: List[str] = None) -> 'SinistreAnalysisPipeline':
        """Crée les visualisations"""
        logger.info("Étape 6: Création des visualisations")
        
        target_var = self.variable_manager.target_variable['name']
        
        if key_variables is None:
            # Sélectionner automatiquement les variables les plus corrélées
            if 'correlations' in self.stats_manager.results:
                key_variables = self.stats_manager.results['correlations'].head(5)['variable'].tolist()
            else:
                key_variables = ['surface_police', 'age_batiment'] if 'age_batiment' in self.processed_data.columns else ['surface_police']
        
        # Distribution de la variable cible
        self.visualizer.plot_distribution(self.processed_data, target_var, log_transform=True)
        
        # Matrice de corrélation
        numeric_vars = [target_var] + [var for var in key_variables if var in self.processed_data.columns]
        self.visualizer.plot_correlation_matrix(self.processed_data, numeric_vars)
        
        # Comparaisons par groupe
        categorical_vars = ['risque_rga_gcl', 'classe_age', 'classe_surface']
        for var in categorical_vars:
            if var in self.processed_data.columns:
                self.visualizer.plot_group_comparison(self.processed_data, var, target_var)
        
        # Dashboard complet
        self.visualizer.create_dashboard(self.processed_data, target_var, key_variables)
        
        # NOUVEAU - Graphique mensuel ajouté ici après le chargement des données
        self.create_monthly_visualization()
        
        # Intégrer les figures dans le générateur de rapports avancés
        if self.generate_advanced_reports:
            self._integrate_visualizations_with_advanced_reports()
        
        return self
    # ====================================================================
    # NOUVELLES MÉTHODES POUR LES RAPPORTS AVANCÉS
    # ====================================================================
    
    def _integrate_visualizations_with_advanced_reports(self):
        """Intègre les visualisations existantes dans le générateur de rapports avancés"""
        
        # Récupérer les figures du visualizer si elles existent
        if hasattr(self.visualizer, 'figures') and self.visualizer.figures:
            for fig_name, fig in self.visualizer.figures.items():
                description = self._get_figure_description(fig_name)
                self.synthesis_manager.advanced_generator.add_figure(fig_name, fig, description)
                logger.info(f"Figure '{fig_name}' ajoutée aux rapports avancés")
    
    def _get_figure_description(self, fig_name: str) -> str:
        """Génère une description pour une figure"""
        descriptions = {
            'distribution': 'Distribution de la variable cible avec transformation logarithmique',
            'correlation_matrix': 'Matrice de corrélation des principales variables numériques',
            'group_comparison': 'Comparaison des groupes pour les variables catégorielles',
            'dashboard': 'Dashboard complet avec vue d\'ensemble des analyses'
        }
        
        for key, desc in descriptions.items():
            if key.lower() in fig_name.lower():
                return desc
        
        return f"Visualisation de l'analyse: {fig_name.replace('_', ' ').title()}"
    
    def _prepare_advanced_report_data(self, target_variable: str):
        """Prépare les données pour les rapports avancés"""
        print("inside _prepare_advanced_report_data")
        # Résumé général
        summary_data = {
            'total_records': len(self.processed_data),
            'average_amount': round(self.processed_data[target_variable].mean(), 2),
            'total_amount': round(self.processed_data[target_variable].sum(), 2),
            'period_coverage': self._get_period_coverage(),
            'key_points': self._extract_key_insights(target_variable)
        }
        
        # Statistiques détaillées
        statistics_data = {
            'descriptive': self._format_descriptive_stats(),
            'correlations': self._format_correlation_data(),
            'tests': self._format_statistical_tests()
        }
        
        # Ajouter aux résultats du générateur avancé
        self.synthesis_manager.advanced_generator.add_analysis_results("summary", summary_data)
        self.synthesis_manager.advanced_generator.add_analysis_results("statistics", statistics_data)
    
    def _get_period_coverage(self) -> str:
        """Détermine la période couverte par les données"""
        date_columns = ['date_sinistre', 'date_declaration', 'date_effet_police']
        
        for col in date_columns:
            if col in self.processed_data.columns:
                try:
                    dates = pd.to_datetime(self.processed_data[col], errors='coerce').dropna()
                    if not dates.empty:
                        start_date = dates.min().strftime('%m/%Y')
                        end_date = dates.max().strftime('%m/%Y')
                        return f"{start_date} - {end_date}"
                except:
                    continue
        return "Période non déterminée"
    
    def _extract_key_insights(self, target_variable: str) -> List[str]:
        """Extrait les insights clés de l'analyse"""
        insights = []
        
        # Insight sur la taille des données
        insights.append(f"Analyse de {len(self.processed_data):,} sinistres après filtrage et nettoyage")
        
        # Insights sur les corrélations
        if 'correlations' in self.stats_manager.results:
            corr_df = self.stats_manager.results['correlations']
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
                # Choisir la colonne numérique adéquate
                col = (
                    'correlation_abs'
                    if 'correlation_abs' in corr_df.columns else
                    ('correlation' if 'correlation' in corr_df.columns else None)
                )
                if col:
                    s = pd.to_numeric(corr_df[col], errors='coerce').abs()
                    strong_corr = corr_df.loc[s >= 0.3]
                    insights.append(
                        f"Identification de {len(strong_corr)} variables fortement corrélées avec {target_variable}"
                    )
        
        # Insights sur les tests statistiques
        significant_tests = [test for test, result in self.stats_manager.test_results.items() 
                           if result.get('significant', False)]
        if significant_tests:
            insights.append(f"Tests statistiques significatifs: {len(significant_tests)} détectés")
        
        # Insight sur les variables catégorielles
        categorical_vars = ['risque_rga_gcl', 'classe_age', 'classe_surface']
        analyzed_cats = [var for var in categorical_vars if var in self.processed_data.columns]
        if analyzed_cats:
            insights.append(f"Analyse de segmentation sur {len(analyzed_cats)} dimensions métier")
        
        return insights
    
    def _format_descriptive_stats(self) -> Dict[str, Dict[str, float]]:
        """Formate les statistiques descriptives pour les rapports avancés"""
        if 'descriptive' not in self.stats_manager.results:
            return {}
        
        formatted_stats = {}
        descriptive_data = self.stats_manager.results['descriptive']
        
        if isinstance(descriptive_data, pd.DataFrame):
            for col in descriptive_data.columns:
                formatted_stats[col] = {
                    'mean': round(descriptive_data[col].mean(), 2),
                    'median': round(descriptive_data[col].median(), 2),
                    'std': round(descriptive_data[col].std(), 2),
                    'min': round(descriptive_data[col].min(), 2),
                    'max': round(descriptive_data[col].max(), 2),
                    'q25': round(descriptive_data[col].quantile(0.25), 2),
                    'q75': round(descriptive_data[col].quantile(0.75), 2)
                }
        
        return formatted_stats
    
    def _format_correlation_data(self) -> Dict[str, float]:
        """Formate les corrélations pour les rapports avancés (robuste)."""
        corrs = self.stats_manager.results.get('correlations')
        if not isinstance(corrs, pd.DataFrame) or corrs.empty:
            return {}

        # Choix des colonnes disponibles
        strength_col = 'correlation_abs' if 'correlation_abs' in corrs.columns else (
            'correlation' if 'correlation' in corrs.columns else None
        )
        value_col = 'correlation' if 'correlation' in corrs.columns else strength_col
        var_col = 'variable' if 'variable' in corrs.columns else None

        if strength_col is None or value_col is None:
            return {}

        corrs_sorted = corrs.sort_values(by=strength_col, ascending=False).head(10)
        out: Dict[str, float] = {}

        for _, row in corrs_sorted.iterrows():
            var_name = str(row[var_col]) if var_col and var_col in row else str(getattr(row, "name", "var"))
            try:
                out[var_name] = round(float(row[value_col]), 3)
            except Exception:
                continue

        return out
    
    def _format_statistical_tests(self) -> Dict[str, str]:
        """Formate les résultats des tests statistiques"""
        tests = {}
        
        for test_name, result in self.stats_manager.test_results.items():
            if isinstance(result, dict):
                p_value = result.get('p_value', 1.0)
                significant = result.get('significant', False)
                
                if significant:
                    tests[test_name] = f"Significatif (p={p_value:.4f})"
                else:
                    tests[test_name] = f"Non significatif (p={p_value:.4f})"
        
        return tests
    
    # ====================================================================
    # MÉTHODE GENERATE_SYNTHESIS MODIFIÉE
    # ====================================================================
    def _ensure_business_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        S’assure que les colonnes métier attendues existent pour la synthèse.
        Crée la colonne manquante avec une valeur par défaut non vide.
        """
        expected_defaults = {
            "risque_rga_gcl": "INCONNU",
            "classe_age": "INCONNU",
            "classe_surface": "INCONNU",
        }
        safe = df.copy()
        for col, default in expected_defaults.items():
            if col not in safe.columns:
                safe[col] = default
        return safe
    
    def generate_synthesis(self) -> 'SinistreAnalysisPipeline':
        """Génère la synthèse finale avec rapports avancés (robuste aux colonnes manquantes)."""
        logger.info("Étape 7: Génération de la synthèse")
        
        if self.processed_data is None:
            raise ValueError("Aucune donnée préprocessée. Exécuter preprocess_data() d'abord.")
            
        
        target_var = self.variable_manager.target_variable['name']

        print(target_var)
        
        # CORRECTION PRINCIPALE: S'assurer que les colonnes métier existent
        # Si elles n'existent pas, les créer avec des valeurs par défaut
        required_columns = ['risque_rga_gcl', 'annee_construction_batiment', 'surface_batiment']
        for col in required_columns:
            if col not in self.processed_data.columns:
                logger.warning(f"Colonne '{col}' manquante - création avec valeur par défaut")
                self.processed_data[col] = 'INCONNU'
        
        # Générer les rapports de base
        self.synthesis_manager.generate_data_quality_report(
            self.data_loader, self.data_filter, self.preprocessor
        )
        print("after generate_data_quality_report")
        self.synthesis_manager.generate_statistical_report(self.stats_manager, target_var)
        
        # Préparer les key findings
        key_findings = {}
        
        # Gestion robuste des corrélations
        if 'correlations' in self.stats_manager.results:
            corr_data = self.stats_manager.results['correlations']
            if isinstance(corr_data, pd.DataFrame) and not corr_data.empty:
                # Identifier la colonne de corrélation
                corr_col = None
                if 'correlation_abs' in corr_data.columns:
                    corr_col = 'correlation_abs'
                elif 'correlation' in corr_data.columns:
                    corr_col = 'correlation'
                
                if corr_col:
                    top_correlations = corr_data.nlargest(3, corr_col)
                    key_findings['top_risk_factors'] = top_correlations.to_dict('records')
                else:
                    key_findings['top_risk_factors'] = []
            else:
                key_findings['top_risk_factors'] = []
        else:
            key_findings['top_risk_factors'] = []
        
        # Tests statistiques
        key_findings['statistical_significance'] = list(self.stats_manager.test_results.keys())
        
        # Business insights
        self.synthesis_manager.generate_business_insights(
            self.processed_data, target_var, key_findings
        )
        
        # Rapports avancés si activés
        if self.generate_advanced_reports:
            logger.info("Génération des rapports HTML avancés...")
            
            # Préparer les données pour les rapports avancés
            self._prepare_advanced_report_data(target_var)
            
            # Générer tous les formats de rapports avancés
            advanced_results = self.synthesis_manager.create_comprehensive_analysis(
                df=self.processed_data,
                target_variable=target_var,
                stats_manager=self.stats_manager,
                data_filter=self.data_filter,
                preprocessor=self.preprocessor
            )
            
            self.results['advanced_reports'] = advanced_results
            logger.info("Rapports avancés générés avec succès")
        
        return self
    
    # ====================================================================
    # MÉTHODE EXPORT_RESULTS ENRICHIE
    # ====================================================================
    
    def export_results(self, export_format: str = 'all') -> Dict[str, str]:
        """Exporte les résultats avec rapports avancés"""
        logger.info("Étape 8: Export des résultats")
        
        export_paths = {}
        
        # EXPORTS EXISTANTS (gardez tel quel)
        if export_format in ['csv', 'all']:
            csv_path = self.synthesis_manager.export_to_csv(
                self.processed_data, "donnees_processees"
            )
            export_paths['processed_data'] = csv_path
        
        if export_format in ['excel', 'all']:
            excel_data = {}
            
            if 'descriptive' in self.stats_manager.results:
                excel_data['Statistiques'] = self.stats_manager.results['descriptive']
            
            if 'correlations' in self.stats_manager.results:
                excel_data['Correlations'] = self.stats_manager.results['correlations']
            
            for key, result in self.stats_manager.results.items():
                if 'group_analysis' in key and isinstance(result, pd.DataFrame):
                    sheet_name = key.replace('group_analysis_', '').title()
                    excel_data[sheet_name] = result
            
            if excel_data:
                excel_path = self.synthesis_manager.export_to_excel(excel_data, "analyse_statistique")
                export_paths['statistical_analysis'] = excel_path
        
        if export_format in ['json', 'all']:
            json_path = self.synthesis_manager.export_synthesis_report()
            export_paths['synthesis_report'] = json_path
        
        if export_format in ['html', 'all']:
            html_path = self.synthesis_manager.generate_html_report()
            export_paths['html_report'] = html_path
        
        # NOUVEAU - Exports des rapports avancés
        if self.generate_advanced_reports and 'advanced_reports' in self.results:
            advanced_reports = self.results['advanced_reports']['reports']
            export_paths.update(advanced_reports)
            
            # Index principal des rapports avancés
            export_paths['reports_index'] = self.results['advanced_reports']['index']
            
            logger.info(f"Rapports avancés exportés: {len(advanced_reports)} formats")
        
        return export_paths
    
    # ====================================================================
    # MÉTHODE RUN_FULL_ANALYSIS ENRICHIE
    # ====================================================================
    
    def run_full_analysis(self, file_path: str, 
                         target_variable: str = 'montant_charge_brut',
                         **kwargs) -> Dict[str, str]:
        """Exécute l'analyse complète avec rapports avancés"""
        logger.info("DÉBUT DE L'ANALYSE COMPLÈTE AVEC RAPPORTS AVANCÉS")
        
        try:
            # Pipeline complet (identique à votre version)
            export_paths = (self
                           .load_data(file_path, **kwargs.get('common_load_params', {}))
                           .filter_data(**kwargs.get('filter_params', {}))
                           .setup_variables(target_variable)
                           .preprocess_data(**kwargs.get('preprocess_params', {}))
                           .analyze_statistics(**kwargs.get('stats_params', {}))
                           .create_visualizations(kwargs.get('key_variables'))
                           .generate_synthesis()
                           .export_results(kwargs.get('export_format', 'all')))
            
            # NOUVEAU - Affichage des résultats enrichis
            logger.info("ANALYSE TERMINÉE AVEC SUCCÈS")
            
            if self.generate_advanced_reports and 'reports_index' in export_paths:
                logger.info(f"📊 Index des rapports avancés: {export_paths['reports_index']}")
                
                advanced_count = len([k for k in export_paths.keys() 
                                    if k.startswith(('html_report', 'dashboard', 'summary_slide'))])
                logger.info(f"📋 {advanced_count} rapports visuels générés")
            
            return export_paths
            
        except Exception as e:
            logger.error(f"ERREUR DANS L'ANALYSE: {e}")
            raise
    
    # ====================================================================
    # NOUVELLES MÉTHODES UTILITAIRES
    # ====================================================================
    
    def disable_advanced_reports(self) -> 'SinistreAnalysisPipeline':
        """Désactive la génération des rapports avancés"""
        self.generate_advanced_reports = False
        logger.info("Rapports avancés désactivés")
        return self
    
    def enable_advanced_reports(self) -> 'SinistreAnalysisPipeline':
        """Active la génération des rapports avancés"""
        self.generate_advanced_reports = True
        logger.info("Rapports avancés activés")
        return self
    
    def get_advanced_reports_status(self) -> Dict[str, Any]:
        """Retourne le statut des rapports avancés"""
        status = {
            'enabled': self.generate_advanced_reports,
            'reports_generated': 'advanced_reports' in self.results,
            'available_formats': []
        }
        
        if 'advanced_reports' in self.results:
            status['available_formats'] = list(self.results['advanced_reports']['reports'].keys())
            status['index_path'] = self.results['advanced_reports'].get('index', '')
        
        return status
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Retourne un résumé complet du pipeline enrichi"""
        base_summary = {
            'data_summary': self.data_loader.get_data_summary() if self.data_loader else {},
            'filter_summary': self.data_filter.get_filter_summary() if self.data_filter else {},
            'preprocessing_summary': self.preprocessor.get_preprocessing_summary() if self.preprocessor else {},
            'statistics_summary': self.stats_manager.get_statistics_summary() if self.stats_manager else {},
            'visualization_summary': self.visualizer.get_visualization_summary() if self.visualizer else {},
            'target_variable': self.variable_manager.target_variable if self.variable_manager else None
        }
        
        # NOUVEAU - Ajout du statut des rapports avancés
        base_summary['advanced_reports_status'] = self.get_advanced_reports_status()
        
        return base_summary
    
    def check_available_columns(self) -> Dict[str, List[str]]:
        """Vérifie et liste les colonnes disponibles dans les données"""
        if self.processed_data is None:
            return {"error": "Aucune donnée chargée"}
        
        columns_info = {
            "total_columns": len(self.processed_data.columns),
            "all_columns": list(self.processed_data.columns),
            "numeric_columns": list(self.processed_data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.processed_data.select_dtypes(include=['object', 'category']).columns),
            "missing_expected": []
        }
        
        # Vérifier les colonnes attendues
        expected_columns = [
            'risque_rga_gcl', 'annee_construction_batiment', 'surface_batiment',
            'date_sinistre', 'surface_police', 'montant_charge_brut'
        ]
        
        for col in expected_columns:
            if col not in self.processed_data.columns:
                columns_info["missing_expected"].append(col)
        
        return columns_info