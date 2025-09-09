#!/usr/bin/env python3
"""
Script d'intégration pour le générateur de rapports HTML avancés
Ce script montre comment intégrer le générateur avec vos analyses existantes
"""

from pathlib import Path
import sys
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rapport_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReportingPipeline:
    """Pipeline complet pour la génération de rapports"""
    
    def __init__(self, output_dir: str = "./rapports_generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialiser le générateur de rapport (à importer depuis votre code)
        # from your_module import IntegratedSynthesisManager
        # self.synthesis_manager = IntegratedSynthesisManager(str(self.output_dir))
        
    def run_complete_analysis(self, 
                            data_path: str, 
                            target_column: str,
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Lance une analyse complète et génère tous les rapports
        
        Args:
            data_path: Chemin vers le fichier de données (CSV, Excel, etc.)
            target_column: Nom de la colonne cible à analyser
            config: Configuration optionnelle pour l'analyse
            
        Returns:
            Dictionary avec les chemins des rapports générés
        """
        
        try:
            logger.info(f"🚀 Début de l'analyse complète - Fichier: {data_path}")
            
            # 1. Chargement des données
            df = self._load_data(data_path)
            logger.info(f"✅ Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
            
            # 2. Validation et préparation
            df_clean = self._prepare_data(df, target_column, config)
            logger.info(f"✅ Données préparées: {len(df_clean)} lignes après nettoyage")
            
            # 3. Analyse statistique
            stats_results = self._run_statistical_analysis(df_clean, target_column)
            logger.info("✅ Analyse statistique terminée")
            
            # 4. Génération des visualisations
            figures = self._create_visualizations(df_clean, target_column)
            logger.info(f"✅ {len(figures)} visualisations créées")
            
            # 5. Génération des rapports
            reports = self._generate_all_reports(df_clean, target_column, stats_results, figures)
            logger.info(f"✅ {len(reports)} rapports générés")
            
            # 6. Création de l'index final
            index_path = self._create_final_index(reports, df_clean, target_column)
            
            logger.info("🎉 Analyse complète terminée avec succès!")
            return {**reports, 'index': index_path}
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse: {str(e)}")
            raise
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Charge les données depuis différents formats"""
        
        file_path = Path(data_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {data_path}")
        
        # Détecter le format et charger
        if file_path.suffix.lower() == '.csv':
            # Essayer différents encodages
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"Fichier CSV chargé avec l'encodage {encoding}")
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError("Impossible de décoder le fichier CSV")
            
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            logger.info("Fichier Excel chargé")
            return df
            
        elif file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
            logger.info("Fichier Parquet chargé")
            return df
            
        else:
            raise ValueError(f"Format de fichier non supporté: {file_path.suffix}")
    
    def _prepare_data(self, df: pd.DataFrame, target_column: str, config: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Prépare et nettoie les données"""
        
        df_clean = df.copy()
        
        # Vérifier que la colonne cible existe
        if target_column not in df_clean.columns:
            raise ValueError(f"Colonne cible '{target_column}' non trouvée dans les données")
        
        # Configuration par défaut
        if config is None:
            config = {}
        
        # Nettoyage des valeurs manquantes
        missing_threshold = config.get('missing_threshold', 0.5)
        
        # Supprimer les colonnes avec trop de valeurs manquantes
        missing_ratio = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Suppression de {len(cols_to_drop)} colonnes avec >50% de valeurs manquantes")
            df_clean = df_clean.drop(columns=cols_to_drop)
        
        # Supprimer les lignes avec valeur cible manquante
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=[target_column])
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            logger.info(f"Suppression de {dropped_rows} lignes avec valeur cible manquante")
        
        # Conversion des types de données
        df_clean = self._convert_data_types(df_clean, config)
        
        # Détection et traitement des outliers si demandé
        if config.get('remove_outliers', False):
            df_clean = self._handle_outliers(df_clean, target_column)
        
        return df_clean
    
    def _convert_data_types(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Convertit automatiquement les types de données"""
        
        df_converted = df.copy()
        
        # Tentative de conversion automatique des dates
        date_patterns = ['date', 'time', 'timestamp']
        
        for col in df_converted.columns:
            if any(pattern in col.lower() for pattern in date_patterns):
                try:
                    df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                    logger.info(f"Colonne '{col}' convertie en datetime")
                except:
                    pass
        
        # Conversion des colonnes numériques
        for col in df_converted.select_dtypes(include=['object']).columns:
            # Essayer de convertir en numérique si possible
            numeric_col = pd.to_numeric(df_converted[col], errors='coerce')
            
            # Si plus de 80% des valeurs sont converties, garder la conversion
            if numeric_col.notna().sum() / len(df_converted) > 0.8:
                df_converted[col] = numeric_col
                logger.info(f"Colonne '{col}' convertie en numérique")
        
        return df_converted
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Gère les valeurs aberrantes"""
        
        if target_column not in df.columns:
            return df
        
        # Méthode IQR pour détecter les outliers
        Q1 = df[target_column].quantile(0.25)
        Q3 = df[target_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Compter les outliers
        outliers_mask = (df[target_column] < lower_bound) | (df[target_column] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        if n_outliers > 0:
            logger.info(f"Détection de {n_outliers} valeurs aberrantes ({n_outliers/len(df)*100:.1f}%)")
            
            # Option: supprimer les outliers extrêmes seulement
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR
            
            extreme_mask = (df[target_column] >= extreme_lower) & (df[target_column] <= extreme_upper)
            df_clean = df[extreme_mask].copy()
            
            removed = len(df) - len(df_clean)
            if removed > 0:
                logger.info(f"Suppression de {removed} outliers extrêmes")
            
            return df_clean
        
        return df
    
    def _run_statistical_analysis(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Lance l'analyse statistique complète"""
        
        results = {
            'descriptive_stats': {},
            'correlations': {},
            'categorical_analysis': {},
            'temporal_analysis': {},
            'outliers_detection': {}
        }
        
        # 1. Statistiques descriptives
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            results['descriptive_stats'][col] = {
                'count': int(df[col].count()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        # 2. Matrice de corrélation
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            # Extraire les corrélations avec la variable cible
            if target_column in corr_matrix.columns:
                target_correlations = corr_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)
                
                results['correlations'] = {
                    'target_correlations': target_correlations.to_dict(),
                    'correlation_matrix': corr_matrix.to_dict()
                }
        
        # 3. Analyse des variables catégorielles
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:10]:  # Limiter à 10 colonnes
            if df[col].nunique() < 50:  # Éviter les colonnes avec trop de catégories
                value_counts = df[col].value_counts()
                
                # Calculer la moyenne de la cible par catégorie
                if target_column in df.columns:
                    target_by_category = df.groupby(col)[target_column].agg(['mean', 'count']).round(2)
                    
                    results['categorical_analysis'][col] = {
                        'value_counts': value_counts.to_dict(),
                        'target_by_category': target_by_category.to_dict()
                    }
        
        # 4. Analyse temporelle
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        for col in date_cols:
            if df[col].notna().sum() > 0:
                # Extraire les composantes temporelles
                df_temp = df.copy()
                df_temp['year'] = df_temp[col].dt.year
                df_temp['month'] = df_temp[col].dt.month
                df_temp['quarter'] = df_temp[col].dt.quarter
                df_temp['weekday'] = df_temp[col].dt.day_name()
                
                temporal_analysis = {}
                for time_comp in ['year', 'month', 'quarter', 'weekday']:
                    if target_column in df.columns:
                        temporal_stats = df_temp.groupby(time_comp)[target_column].agg(['mean', 'count']).round(2)
                        temporal_analysis[time_comp] = temporal_stats.to_dict()
                
                results['temporal_analysis'][col] = temporal_analysis
        
        # 5. Détection d'outliers
        if target_column in df.columns:
            Q1 = df[target_column].quantile(0.25)
            Q3 = df[target_column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[target_column] < lower_bound) | (df[target_column] > upper_bound)
            n_outliers = outliers_mask.sum()
            
            results['outliers_detection'] = {
                'total_outliers': int(n_outliers),
                'outliers_percentage': float(n_outliers / len(df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR)
            }
        
        return results
    
    def _create_visualizations(self, df: pd.DataFrame, target_column: str) -> Dict[str, plt.Figure]:
        """Crée toutes les visualisations nécessaires"""
        
        figures = {}
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        try:
            # 1. Distribution de la variable cible
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            
            if target_column in df.columns:
                # Histogramme et courbe de densité
                df[target_column].hist(bins=50, alpha=0.7, ax=ax1, density=True, color='skyblue')
                
                # Ajouter la courbe de densité
                from scipy import stats
                df_clean = df[target_column].dropna()
                if len(df_clean) > 0:
                    density = stats.gaussian_kde(df_clean)
                    xs = np.linspace(df_clean.min(), df_clean.max(), 200)
                    density_curve = density(xs)
                    ax1.plot(xs, density_curve, 'r-', linewidth=2, label='Courbe de densité')
                
                ax1.set_title(f'Distribution de {target_column}', fontsize=14, fontweight='bold')
                ax1.set_xlabel(target_column, fontsize=12)
                ax1.set_ylabel('Densité', fontsize=12)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                figures['distribution'] = fig1
            
            # 2. Boxplot pour détecter les outliers
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            if target_column in df.columns:
                box_data = [df[target_column].dropna()]
                bp = ax2.boxplot(box_data, patch_artist=True, labels=[target_column])
                
                # Personnaliser le boxplot
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_alpha(0.7)
                
                ax2.set_title(f'Boxplot - {target_column}', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Valeurs', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                figures['boxplot'] = fig2
            
            # 3. Matrice de corrélation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                fig3, ax3 = plt.subplots(figsize=(12, 8))
                
                corr_matrix = df[numeric_cols].corr()
                
                # Heatmap
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                           square=True, ax=ax3, fmt='.2f')
                
                ax3.set_title('Matrice de Corrélation', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                figures['correlation_heatmap'] = fig3
            
            # 4. Top corrélations avec la variable cible
            if target_column in df.columns and len(numeric_cols) > 1:
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                
                corr_with_target = df[numeric_cols].corrwith(df[target_column]).abs()
                corr_with_target = corr_with_target.drop(target_column, errors='ignore')
                top_corr = corr_with_target.nlargest(10)
                
                if not top_corr.empty:
                    top_corr.plot(kind='barh', ax=ax4, color='steelblue')
                    ax4.set_title(f'Top 10 - Corrélations avec {target_column}', fontsize=14, fontweight='bold')
                    ax4.set_xlabel('Corrélation absolue', fontsize=12)
                    ax4.grid(True, alpha=0.3)
                    
                    figures['top_correlations'] = fig4
            
            # 5. Analyse temporelle si applicable
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            if date_cols and target_column in df.columns:
                date_col = date_cols[0]  # Prendre la première colonne date
                
                fig5, ((ax5a, ax5b), (ax5c, ax5d)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Préparer les données temporelles
                df_temp = df.copy()
                df_temp['year'] = df_temp[date_col].dt.year
                df_temp['month'] = df_temp[date_col].dt.month
                df_temp['quarter'] = df_temp[date_col].dt.quarter
                df_temp['weekday'] = df_temp[date_col].dt.day_name()
                
                # Évolution par année
                yearly_avg = df_temp.groupby('year')[target_column].mean()
                yearly_avg.plot(kind='line', ax=ax5a, marker='o', linewidth=2)
                ax5a.set_title('Évolution annuelle', fontsize=12, fontweight='bold')
                ax5a.grid(True, alpha=0.3)
                
                # Évolution par mois
                monthly_avg = df_temp.groupby('month')[target_column].mean()
                monthly_avg.plot(kind='bar', ax=ax5b, color='orange', alpha=0.7)
                ax5b.set_title('Variation mensuelle', fontsize=12, fontweight='bold')
                ax5b.set_xlabel('Mois')
                ax5b.grid(True, alpha=0.3)
                
                # Évolution par trimestre
                quarterly_avg = df_temp.groupby('quarter')[target_column].mean()
                quarterly_avg.plot(kind='bar', ax=ax5c, color='green', alpha=0.7)
                ax5c.set_title('Variation trimestrielle', fontsize=12, fontweight='bold')
                ax5c.set_xlabel('Trimestre')
                ax5c.grid(True, alpha=0.3)
                
                # Évolution par jour de la semaine
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_avg = df_temp.groupby('weekday')[target_column].mean().reindex(weekday_order)
                weekday_avg.plot(kind='bar', ax=ax5d, color='purple', alpha=0.7)
                ax5d.set_title('Variation par jour de la semaine', fontsize=12, fontweight='bold')
                ax5d.set_xlabel('Jour de la semaine')
                ax5d.tick_params(axis='x', rotation=45)
                ax5d.grid(True, alpha=0.3)
                
                plt.suptitle('Analyse Temporelle', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                figures['temporal_analysis'] = fig5
            
            # 6. Analyse des variables catégorielles les plus importantes
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols and target_column in df.columns:
                # Prendre les 4 premières variables catégorielles avec peu de catégories
                cat_cols_filtered = [col for col in categorical_cols if df[col].nunique() <= 10][:4]
                
                if cat_cols_filtered:
                    n_cols = len(cat_cols_filtered)
                    n_rows = (n_cols + 1) // 2
                    
                    fig6, axes = plt.subplots(n_rows, 2, figsize=(16, 6*n_rows))
                    if n_cols == 1:
                        axes = [axes]
                    elif n_rows == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(cat_cols_filtered):
                        ax = axes[i]
                        
                        # Moyenne de la variable cible par catégorie
                        cat_means = df.groupby(col)[target_column].mean().sort_values(ascending=False)
                        
                        cat_means.plot(kind='bar', ax=ax, color='coral', alpha=0.7)
                        ax.set_title(f'Moyenne de {target_column} par {col}', fontsize=12, fontweight='bold')
                        ax.set_ylabel(f'Moyenne {target_column}')
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3)
                    
                    # Cacher les axes non utilisés
                    for i in range(len(cat_cols_filtered), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.suptitle('Analyse des Variables Catégorielles', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    
                    figures['categorical_analysis'] = fig6
            
            logger.info(f"✅ {len(figures)} visualisations créées avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création des visualisations: {str(e)}")
        
        return figures
    
    def _generate_all_reports(self, df: pd.DataFrame, target_column: str, 
                            stats_results: Dict[str, Any], figures: Dict[str, plt.Figure]) -> Dict[str, str]:
        """Génère tous les types de rapports"""
        
        reports = {}
        
        try:
            # Ici vous intégreriez votre AdvancedReportGenerator
            # Pour la démo, nous créons des rapports simples
            
            # 1. Rapport HTML simple
            html_report = self._create_simple_html_report(df, target_column, stats_results, figures)
            reports['html_simple'] = html_report
            
            # 2. Rapport CSV avec résumé statistique
            csv_report = self._create_csv_report(df, target_column, stats_results)
            reports['csv_summary'] = csv_report
            
            # 3. Rapport JSON avec tous les résultats
            json_report = self._create_json_report(df, target_column, stats_results)
            reports['json_complete'] = json_report
            
            # Si vous avez le générateur avancé:
            # synthesis_manager = IntegratedSynthesisManager(str(self.output_dir))
            # advanced_reports = synthesis_manager.create_comprehensive_analysis(df, target_column, stats_results)
            # reports.update(advanced_reports['reports'])
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération des rapports: {str(e)}")
        
        return reports
    
    def _create_simple_html_report(self, df: pd.DataFrame, target_column: str, 
                                  stats_results: Dict[str, Any], figures: Dict[str, plt.Figure]) -> str:
        """Crée un rapport HTML simple"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        html_path = self.output_dir / f"rapport_simple_{timestamp}.html"
        
        # Sauvegarder les figures comme images
        figures_html = ""
        for fig_name, fig in figures.items():
            img_path = self.output_dir / f"{fig_name}_{timestamp}.png"
            fig.savefig(img_path, dpi=300, bbox_inches='tight')
            figures_html += f"""
            <div class="figure">
                <h3>{fig_name.replace('_', ' ').title()}</h3>
                <img src="{img_path.name}" alt="{fig_name}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        # Créer le HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse - {target_column}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>📊 Rapport d'Analyse - {target_column}</h1>
            <p>Généré le {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}</p>
            
            <div class="section">
                <h2>Résumé des données</h2>
                <p><strong>Nombre total d'enregistrements:</strong> {len(df):,}</p>
                <p><strong>Nombre de colonnes:</strong> {len(df.columns)}</p>
                <p><strong>Variable analysée:</strong> {target_column}</p>
            </div>
            
            <div class="section">
                <h2>Statistiques descriptives</h2>
                {self._create_stats_table_html(stats_results.get('descriptive_stats', {}))}
            </div>
            
            <div class="section">
                <h2>Visualisations</h2>
                {figures_html}
            </div>
            
            <div class="section">
                <h2>Insights principaux</h2>
                {self._create_insights_html(df, target_column, stats_results)}
            </div>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML simple créé: {html_path}")
        return str(html_path)
    
    def _create_stats_table_html(self, descriptive_stats: Dict[str, Any]) -> str:
        """Crée le tableau HTML des statistiques"""
        
        if not descriptive_stats:
            return "<p>Aucune statistique disponible</p>"
        
        html = """
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Moyenne</th>
                    <th>Médiane</th>
                    <th>Écart-type</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for var, stats in descriptive_stats.items():
            html += f"""
            <tr>
                <td>{var}</td>
                <td>{stats.get('mean', 'N/A'):.2f}</td>
                <td>{stats.get('median', 'N/A'):.2f}</td>
                <td>{stats.get('std', 'N/A'):.2f}</td>
                <td>{stats.get('min', 'N/A'):.2f}</td>
                <td>{stats.get('max', 'N/A'):.2f}</td>
            </tr>
            """
        
        html += "</tbody></table>"
        return html
    
    def _create_insights_html(self, df: pd.DataFrame, target_column: str, stats_results: Dict[str, Any]) -> str:
        """Crée la section insights en HTML"""
        
        insights = []
        
        # Insight sur la distribution
        if 'descriptive_stats' in stats_results and target_column in stats_results['descriptive_stats']:
            stats = stats_results['descriptive_stats'][target_column]
            mean_val = stats.get('mean', 0)
            median_val = stats.get('median', 0)
            
            if mean_val > median_val * 1.1:
                insights.append("La distribution est asymétrique avec une queue vers les valeurs élevées")
            elif median_val > mean_val * 1.1:
                insights.append("La distribution est asymétrique avec une queue vers les valeurs faibles")
            else:
                insights.append("La distribution est relativement symétrique")
        
        # Insight sur les corrélations
        if 'correlations' in stats_results and 'target_correlations' in stats_results['correlations']:
            corr_dict = stats_results['correlations']['target_correlations']
            strong_corr = {k: v for k, v in corr_dict.items() if abs(v) > 0.5}
            
            if strong_corr:
                top_var = max(strong_corr, key=lambda k: abs(strong_corr[k]))
                insights.append(f"Variable la plus corrélée: {top_var} (corrélation: {strong_corr[top_var]:.3f})")
        
        # Insight sur les outliers
        if 'outliers_detection' in stats_results:
            outliers_pct = stats_results['outliers_detection'].get('outliers_percentage', 0)
            if outliers_pct > 5:
                insights.append(f"Attention: {outliers_pct:.1f}% de valeurs aberrantes détectées")
        
        if not insights:
            insights.append("Analyse complétée - voir les visualisations pour plus de détails")
        
        html = "<ul>"
        for insight in insights:
            html += f"<li>{insight}</li>"
        html += "</ul>"
        
        return html
    
    def _create_csv_report(self, df: pd.DataFrame, target_column: str, stats_results: Dict[str, Any]) -> str:
        """Crée un rapport CSV avec les statistiques"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f"resume_statistiques_{timestamp}.csv"
        
        # Préparer les données pour le CSV
        summary_data = []
        
        if 'descriptive_stats' in stats_results:
            for var, stats in stats_results['descriptive_stats'].items():
                summary_data.append({
                    'Variable': var,
                    'Moyenne': stats.get('mean', ''),
                    'Mediane': stats.get('median', ''),
                    'EcartType': stats.get('std', ''),
                    'Minimum': stats.get('min', ''),
                    'Maximum': stats.get('max', ''),
                    'Q25': stats.get('q25', ''),
                    'Q75': stats.get('q75', '')
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"Rapport CSV créé: {csv_path}")
        return str(csv_path)
    
    def _create_json_report(self, df: pd.DataFrame, target_column: str, stats_results: Dict[str, Any]) -> str:
        """Crée un rapport JSON complet"""
        
        import json
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"analyse_complete_{timestamp}.json"
        
        # Préparer le rapport complet
        complete_report = {
            'metadata': {
                'generated_at': pd.Timestamp.now().isoformat(),
                'target_variable': target_column,
                'total_records': len(df),
                'total_columns': len(df.columns)
            },
            'data_summary': {
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'datetime_columns': len(df.select_dtypes(include=['datetime64']).columns),
                'missing_values': df.isnull().sum().sum()
            },
            'analysis_results': stats_results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(complete_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Rapport JSON créé: {json_path}")
        return str(json_path)
    
    def _create_final_index(self, reports: Dict[str, str], df: pd.DataFrame, target_column: str) -> str:
        """Crée la page d'index finale"""
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        index_path = self.output_dir / f"index_{timestamp}.html"
        
        # Créer la liste des rapports
        reports_list = ""
        for report_type, path in reports.items():
            filename = Path(path).name
            reports_list += f"""
            <li>
                <strong>{report_type.replace('_', ' ').title()}:</strong> 
                <a href="{filename}" target="_blank">{filename}</a>
            </li>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Index - Analyse {target_column}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }}
                h1 {{ color: #333; text-align: center; }}
                .summary {{ background: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                ul {{ line-height: 2; }}
                a {{ color: #007bff; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Index des Rapports d'Analyse</h1>
                
                <div class="summary">
                    <h3>Résumé de l'analyse</h3>
                    <p><strong>Variable analysée:</strong> {target_column}</p>
                    <p><strong>Nombre d'enregistrements:</strong> {len(df):,}</p>
                    <p><strong>Date de génération:</strong> {pd.Timestamp.now().strftime('%d/%m/%Y à %H:%M')}</p>
                    <p><strong>Nombre de rapports générés:</strong> {len(reports)}</p>
                </div>
                
                <h3>📋 Rapports disponibles:</h3>
                <ul>
                    {reports_list}
                </ul>
                
                <p style="text-align: center; color: #666; margin-top: 40px;">
                    Tous les fichiers sont disponibles dans le dossier: {self.output_dir}
                </p>
            </div>
        </body>
        </html>
        """
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Index final créé: {index_path}")
        return str(index_path)


def main():
    """Fonction principale pour lancer l'analyse"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Générateur de rapports d\'analyse automatisé')
    parser.add_argument('--data', required=True, help='Chemin vers le fichier de données')
    parser.add_argument('--target', required=True, help='Nom de la colonne cible à analyser')
    parser.add_argument('--output', default='./rapports_output', help='Dossier de sortie')
    parser.add_argument('--remove-outliers', action='store_true', help='Supprimer les outliers extrêmes')
    parser.add_argument('--missing-threshold', type=float, default=0.5, help='Seuil pour supprimer les colonnes avec trop de valeurs manquantes')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'remove_outliers': args.remove_outliers,
        'missing_threshold': args.missing_threshold
    }
    
    # Créer le pipeline
    pipeline = ReportingPipeline(args.output)
    
    try:
        # Lancer l'analyse complète
        results = pipeline.run_complete_analysis(args.data, args.target, config)
        
        print("\n" + "="*60)
        print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("="*60)
        print(f"📁 Dossier de sortie: {args.output}")
        print(f"🎯 Page d'accueil: {results['index']}")
        print("\n📊 Rapports générés:")
        
        for report_type, path in results.items():
            if report_type != 'index':
                print(f"   - {report_type}: {Path(path).name}")
        
        print("\n💡 Ouvrez le fichier index dans votre navigateur pour accéder à tous les rapports!")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        logger.error(f"Erreur principale: {str(e)}", exc_info=True)
        return 1
    
    return 0


# Exemple d'utilisation programmatique
def example_usage():
    """Exemple d'utilisation du pipeline en mode programmatique"""
    
    # Créer des données d'exemple
    np.random.seed(42)
    n_samples = 2000
    
    # Simuler des données réalistes d'assurance
    df_example = pd.DataFrame({
        'montant_sinistre': np.random.lognormal(mean=8.2, sigma=1.1, size=n_samples),
        'age_conducteur': np.random.randint(18, 80, size=n_samples),
        'anciennete_permis': np.random.randint(0, 60, size=n_samples),
        'bonus_malus': np.random.uniform(0.5, 3.5, size=n_samples),
        'puissance_vehicule': np.random.randint(4, 20, size=n_samples),
        'age_vehicule': np.random.randint(0, 25, size=n_samples),
        'kilometrage_annuel': np.random.randint(5000, 50000, size=n_samples),
        'type_vehicule': np.random.choice(['citadine', 'berline', 'suv', 'sportive', 'utilitaire'], 
                                        size=n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'zone_geographique': np.random.choice(['urbain', 'rural', 'periurbain'], 
                                            size=n_samples, p=[0.45, 0.25, 0.3]),
        'profession': np.random.choice(['employe', 'cadre', 'ouvrier', 'retraite', 'etudiant'], 
                                     size=n_samples, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
        'date_sinistre': pd.date_range('2022-01-01', periods=n_samples, freq='H')[:n_samples]
    })
    
    # Ajouter quelques corrélations réalistes
    df_example['montant_sinistre'] = (
        df_example['montant_sinistre'] * 
        (1 + 0.3 * (df_example['puissance_vehicule'] / 20)) *  # Plus de puissance = plus cher
        (1 + 0.2 * (df_example['age_vehicule'] / 25)) *        # Plus vieux = plus cher
        (1 - 0.1 * (df_example['bonus_malus'] - 1))           # Bonus réduit le coût
    )
    
    # Ajouter quelques valeurs manquantes de manière réaliste
    df_example.loc[np.random.choice(df_example.index, size=50), 'profession'] = np.nan
    df_example.loc[np.random.choice(df_example.index, size=30), 'kilometrage_annuel'] = np.nan
    
    # Sauvegarder les données d'exemple
    example_data_path = "./exemple_donnees_sinistres.csv"
    df_example.to_csv(example_data_path, index=False, encoding='utf-8')
    
    print(f"📁 Données d'exemple créées: {example_data_path}")
    print(f"📊 {len(df_example)} lignes, {len(df_example.columns)} colonnes")
    
    # Créer le pipeline et lancer l'analyse
    pipeline = ReportingPipeline("./exemple_rapports")
    
    config = {
        'remove_outliers': True,
        'missing_threshold': 0.3
    }
    
    results = pipeline.run_complete_analysis(
        data_path=example_data_path,
        target_column='montant_sinistre',
        config=config
    )
    
    print("\n🎯 Exemple d'analyse terminé!")
    print(f"📋 Consultez: {results['index']}")
    
    return results


# Utilitaires supplémentaires
class ReportValidator:
    """Classe pour valider et vérifier les rapports générés"""
    
    @staticmethod
    def validate_data_file(file_path: str) -> Dict[str, Any]:
        """Valide un fichier de données avant analyse"""
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            file_path = Path(file_path)
            
            # Vérifier que le fichier existe
            if not file_path.exists():
                validation_result['valid'] = False
                validation_result['errors'].append(f"Fichier non trouvé: {file_path}")
                return validation_result
            
            # Vérifier la taille du fichier
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            validation_result['info']['file_size_mb'] = round(file_size_mb, 2)
            
            if file_size_mb > 500:  # Plus de 500 MB
                validation_result['warnings'].append(f"Fichier volumineux ({file_size_mb:.1f} MB) - le traitement peut être lent")
            
            # Essayer de charger un échantillon
            if file_path.suffix.lower() == '.csv':
                sample_df = pd.read_csv(file_path, nrows=1000)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                sample_df = pd.read_excel(file_path, nrows=1000)
            else:
                validation_result['errors'].append(f"Format de fichier non supporté: {file_path.suffix}")
                validation_result['valid'] = False
                return validation_result
            
            # Informations sur l'échantillon
            validation_result['info']['columns'] = list(sample_df.columns)
            validation_result['info']['n_columns'] = len(sample_df.columns)
            validation_result['info']['sample_rows'] = len(sample_df)
            
            # Vérifications de qualité
            if len(sample_df.columns) > 100:
                validation_result['warnings'].append(f"Beaucoup de colonnes ({len(sample_df.columns)}) - considérer une sélection")
            
            # Vérifier les types de données
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            validation_result['info']['numeric_columns'] = len(numeric_cols)
            
            if len(numeric_cols) == 0:
                validation_result['warnings'].append("Aucune colonne numérique détectée")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Erreur lors de la validation: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def check_target_column(file_path: str, target_column: str) -> Dict[str, Any]:
        """Vérifie la validité de la colonne cible"""
        
        check_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Charger un échantillon
            if Path(file_path).suffix.lower() == '.csv':
                sample_df = pd.read_csv(file_path, nrows=1000)
            else:
                sample_df = pd.read_excel(file_path, nrows=1000)
            
            # Vérifier que la colonne existe
            if target_column not in sample_df.columns:
                check_result['valid'] = False
                check_result['errors'].append(f"Colonne '{target_column}' non trouvée")
                check_result['info']['available_columns'] = list(sample_df.columns)
                return check_result
            
            # Analyser la colonne cible
            target_series = sample_df[target_column]
            
            check_result['info']['data_type'] = str(target_series.dtype)
            check_result['info']['non_null_count'] = target_series.notna().sum()
            check_result['info']['null_count'] = target_series.isna().sum()
            check_result['info']['unique_values'] = target_series.nunique()
            
            # Vérifications
            if target_series.isna().sum() > len(target_series) * 0.5:
                check_result['warnings'].append(f"Plus de 50% de valeurs manquantes dans '{target_column}'")
            
            if not pd.api.types.is_numeric_dtype(target_series):
                check_result['warnings'].append(f"Colonne '{target_column}' n'est pas numérique")
            
            if target_series.nunique() < 10:
                check_result['info']['sample_values'] = target_series.value_counts().head().to_dict()
            else:
                check_result['info']['min_value'] = float(target_series.min()) if pd.api.types.is_numeric_dtype(target_series) else None
                check_result['info']['max_value'] = float(target_series.max()) if pd.api.types.is_numeric_dtype(target_series) else None
                check_result['info']['mean_value'] = float(target_series.mean()) if pd.api.types.is_numeric_dtype(target_series) else None
        
        except Exception as e:
            check_result['valid'] = False
            check_result['errors'].append(f"Erreur lors de la vérification: {str(e)}")
        
        return check_result


def interactive_setup():
    """Configuration interactive pour les utilisateurs non-techniques"""
    
    print("🔧 CONFIGURATION INTERACTIVE")
    print("=" * 50)
    
    # Demander le fichier de données
    while True:
        data_file = input("\n📁 Chemin vers votre fichier de données: ").strip()
        
        if not data_file:
            print("❌ Veuillez spécifier un fichier")
            continue
            
        # Valider le fichier
        validation = ReportValidator.validate_data_file(data_file)
        
        if not validation['valid']:
            print("❌ Fichier invalide:")
            for error in validation['errors']:
                print(f"   - {error}")
            continue
        
        # Afficher les informations
        print(f"✅ Fichier valide ({validation['info']['file_size_mb']} MB)")
        print(f"📊 {validation['info']['n_columns']} colonnes détectées")
        
        # Afficher les avertissements
        for warning in validation['warnings']:
            print(f"⚠️  {warning}")
        
        break
    
    # Afficher les colonnes disponibles
    print(f"\n📋 Colonnes disponibles:")
    for i, col in enumerate(validation['info']['columns'], 1):
        print(f"   {i:2d}. {col}")
    
    # Demander la colonne cible
    while True:
        target_input = input(f"\n🎯 Colonne cible (nom ou numéro): ").strip()
        
        if target_input.isdigit():
            col_index = int(target_input) - 1
            if 0 <= col_index < len(validation['info']['columns']):
                target_column = validation['info']['columns'][col_index]
            else:
                print("❌ Numéro de colonne invalide")
                continue
        else:
            target_column = target_input
        
        # Valider la colonne cible
        target_check = ReportValidator.check_target_column(data_file, target_column)
        
        if not target_check['valid']:
            print("❌ Colonne cible invalide:")
            for error in target_check['errors']:
                print(f"   - {error}")
            continue
        
        print(f"✅ Colonne cible: {target_column}")
        print(f"📊 Type: {target_check['info']['data_type']}")
        print(f"📈 Valeurs non-nulles: {target_check['info']['non_null_count']}")
        
        for warning in target_check['warnings']:
            print(f"⚠️  {warning}")
        
        break
    
    # Options avancées
    print(f"\n🔧 Options avancées (Entrée pour valeur par défaut):")
    
    output_dir = input("📁 Dossier de sortie [./rapports_auto]: ").strip()
    if not output_dir:
        output_dir = "./rapports_auto"
    
    remove_outliers = input("🚫 Supprimer les outliers extrêmes? [o/N]: ").strip().lower()
    remove_outliers = remove_outliers in ['o', 'oui', 'y', 'yes']
    
    # Lancer l'analyse
    print(f"\n🚀 LANCEMENT DE L'ANALYSE")
    print("=" * 50)
    
    config = {
        'remove_outliers': remove_outliers,
        'missing_threshold': 0.5
    }
    
    pipeline = ReportingPipeline(output_dir)
    
    try:
        results = pipeline.run_complete_analysis(data_file, target_column, config)
        
        print(f"\n🎉 ANALYSE TERMINÉE!")
        print(f"📁 Dossier: {output_dir}")
        print(f"🌐 Page d'accueil: {results['index']}")
        
        # Proposer d'ouvrir le navigateur
        open_browser = input("\n🌐 Ouvrir dans le navigateur? [O/n]: ").strip().lower()
        if open_browser not in ['n', 'non', 'no']:
            try:
                import webbrowser
                webbrowser.open(f"file://{Path(results['index']).absolute()}")
                print("🌐 Navigateur ouvert!")
            except:
                print("❌ Impossible d'ouvrir automatiquement le navigateur")
                print(f"📋 Ouvrez manuellement: {results['index']}")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # Vérifier les arguments
    if len(sys.argv) == 1:
        # Mode interactif
        sys.exit(interactive_setup())
    elif '--example' in sys.argv:
        # Mode exemple
        print("🧪 GÉNÉRATION D'UN EXEMPLE COMPLET")
        print("=" * 50)
        example_usage()
    else:
        # Mode ligne de commande
        sys.exit(main())