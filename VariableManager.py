from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class VariableImportance(Enum):
    """Niveaux d'importance des variables"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class VariableMetadata:
    """Métadonnées enrichies pour les variables"""
    name: str
    description: str
    var_type: str  # 'numeric', 'categorical', 'boolean', 'datetime'
    importance: VariableImportance
    expected_correlation: str  # 'positive', 'negative', 'mixed', 'none'
    business_meaning: str

class VariableManager:
    """Gestionnaire des variables d'analyse spécialisé sinistres sécheresse"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.target_variable: Optional[Dict[str, Any]] = None # Variable cible ACTIVE
        self.feature_groups: Dict[str, List[str]] = {}
        self.variable_metadata: Dict[str, VariableMetadata] = {}
        self.analysis_results: Dict[str, Any] = {}
        
        # Initialiser avec les groupes spécialisés sécheresse
        self._initialize_secheresse_groups()

    def _initialize_secheresse_groups(self) -> None:
        """Initialise les groupes de variables spécialisés pour les sinistres sécheresse"""
        
        # Groupes principaux selon votre classification
        self.feature_groups = {
            # Variables de base - forte corrélation attendue
            'base_critical': [
                'prime', 'surface_police', 'nb_pieces_police', 'engagements'
            ],
            
            'risk_critical': [
                'risque_rga_gcl', 'risque_rga_gcl_detail', 'risk_retraitGonflementArgile'
            ],
            
            # Variables bâtiment - impact modéré
            'building_structure': [
                'surface_batiment', 'hauteur_batiment', 'nb_etages_batiment',
                'annee_construction_batiment', 'type_mur_batiment', 'type_toit_batiment'
            ],
            
            'building_bdtopo': [
                'bdtopo_surface', 'bdtopo_hauteur', 'bdtopo_nb_etages',
                'bdtopo_leger_bool', 'bdtopo_matmurs', 'bdtopo_usage1'
            ],
            
            # Variables environnementales - impact spécifique sécheresse
            'soil_composition': [
                'compo_sol_has_argile', 'compo_sol_has_sable', 'compo_sol_has_calcaire',
                'compo_sol_has_gravier', 'compo_sol_has_roche'
            ],
            
            'topography': [
                'dem_pente_maximale', 'dem_altitude', 'dem_pente_est', 'dem_pente_nord'
            ],
            
            'weather_critical': [
                'weather_precip_mean', 'weather_precip_std', 'weather_precip_min', 'weather_precip_max',
                'weather_temp_mean', 'weather_temp_std', 'weather_humidity_mean'
            ],
            
            'sma_critical': [  # Très pertinent pour sécheresse
                'sma_mean', 'sma_std', 'sma_min', 'sma_max',
                'sma_poly3_coeff0', 'sma_poly3_coeff1', 'sma_fft_first_coeff_energy'
            ],
            
            'egms_terrain': [  # Déplacements terrain
                'egms_ts_slope', 'egms_ts_std', 'egms_ts_range', 'egms_ts_velocity_max_abs',
                'egms_ts_acceleration_max_abs'
            ],
            
            'environment_trees': [
                'tree_dist_3', 'tree_dist_5', 'tree_dist_8', 'tree_dist_12'
            ],
            
            # Variables contextuelles
            'geographic': [
                'cpostal', 'ville', 'latitude', 'longitude'
            ],
            
            'temporal': [
                'date_sinistre', 'exercice'
            ],
            
            # Variables dérivées (anciennes + nouvelles)
            'derived': [
                'age_batiment', 'classe_age', 'classe_surface', 'region_code', 
                'saison_sinistre', 'delai_cloture_jours'
            ]
        }
        
        # Définir les métadonnées pour les variables principales
        self._define_variable_metadata()
        
        logger.info(f"Groupes de variables sécheresse initialisés: {list(self.feature_groups.keys())}")

    def _define_variable_metadata(self) -> None:
        """Définit les métadonnées des variables principales"""
        
        # Variables critiques
        critical_vars = {
            'prime': VariableMetadata(
                'prime', 'Prime d\'assurance', 'numeric', VariableImportance.CRITICAL, 
                'positive', 'Montant assuré - proxy de la valeur du bien'
            ),
            'risque_rga_gcl': VariableMetadata(
                'risque_rga_gcl', 'Niveau de risque RGA', 'categorical', VariableImportance.CRITICAL,
                'positive', 'Classification du risque retrait-gonflement des argiles'
            ),
            'sma_mean': VariableMetadata(
                'sma_mean', 'Anomalie humidité sol moyenne', 'numeric', VariableImportance.CRITICAL,
                'negative', 'Indicateur direct de sécheresse des sols'
            ),
            'compo_sol_has_argile': VariableMetadata(
                'compo_sol_has_argile', 'Présence d\'argile', 'boolean', VariableImportance.CRITICAL,
                'positive', 'Argile = sensibilité au retrait-gonflement'
            )
        }
        
        # Variables importantes
        high_vars = {
            'surface_police': VariableMetadata(
                'surface_police', 'Surface déclarée au contrat', 'numeric', VariableImportance.HIGH,
                'positive', 'Taille du bien assuré'
            ),
            'weather_precip_mean': VariableMetadata(
                'weather_precip_mean', 'Précipitations moyennes', 'numeric', VariableImportance.HIGH,
                'negative', 'Moins de pluie = plus de risque sécheresse'
            ),
            'egms_ts_slope': VariableMetadata(
                'egms_ts_slope', 'Tendance déplacement terrain', 'numeric', VariableImportance.HIGH,
                'positive', 'Déformation progressive du terrain'
            )
        }
        
        self.variable_metadata.update(critical_vars)
        self.variable_metadata.update(high_vars)

    def define_target(self, variable_name: str, description: str = "") -> None:
        """Définit la variable cible avec métadonnées enrichies"""
        
        if not description and variable_name == 'montant_charge_brut':
            description = "Montant de charge brut du sinistre - coût total pour l'assureur"
        
        self.target_variable = {
            'name': variable_name,
            'description': description or f"Variable cible: {variable_name}",
            'type': 'target',
            'business_impact': 'direct',  # Impact métier
            'analysis_type': 'regression'  # Type d'analyse prévu
        }
        
        logger.info(f"Variable cible définie: {variable_name}")

    def analyze_variable_availability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse la disponibilité des variables par groupe"""
        availability_report = {}
        
        for group_name, variables in self.feature_groups.items():
            available_vars = [var for var in variables if var in df.columns]
            missing_vars = [var for var in variables if var not in df.columns]
            
            # Statistiques de complétude pour les variables disponibles
            completeness_stats = {}
            if available_vars:
                for var in available_vars:
                    non_null = df[var].count()
                    completeness_stats[var] = {
                        'non_null_count': non_null,
                        'completeness_pct': (non_null / len(df)) * 100,
                        'missing_count': len(df) - non_null
                    }
            
            # Importance moyenne du groupe
            group_importance = self._get_group_importance(group_name)
            
            availability_report[group_name] = {
                'total_vars': len(variables),
                'available_vars': len(available_vars),
                'missing_vars': len(missing_vars),
                'availability_rate': (len(available_vars) / len(variables)) * 100 if variables else 0,
                'group_importance': group_importance,
                'available_vars_list': available_vars,
                'missing_vars_list': missing_vars,
                'completeness_stats': completeness_stats
            }
        
        # Résumé global
        total_vars = sum(len(vars) for vars in self.feature_groups.values())
        total_available = sum(report['available_vars'] for report in availability_report.values())
        
        summary = {
            'total_defined_vars': total_vars,
            'total_available_vars': total_available,
            'overall_availability_pct': (total_available / total_vars) * 100 if total_vars > 0 else 0,
            'critical_groups_ready': [
                name for name, report in availability_report.items()
                if report['group_importance'] in ['critical', 'high'] and report['availability_rate'] > 50
            ]
        }
        
        availability_report['summary'] = summary
        self.analysis_results['availability'] = availability_report
        
        logger.info(f"Analyse disponibilité: {total_available}/{total_vars} variables ({summary['overall_availability_pct']:.1f}%)")
        return availability_report

    def _get_group_importance(self, group_name: str) -> str:
        """Détermine l'importance d'un groupe de variables"""
        if 'critical' in group_name:
            return 'critical'
        elif group_name in ['building_structure', 'weather_critical', 'soil_composition', 'egms_terrain']:
            return 'high'
        elif group_name in ['building_bdtopo', 'topography', 'environment_trees']:
            return 'medium'
        else:
            return 'low'

    def compute_variable_importance(self, df: pd.DataFrame, target_var: str = None) -> Dict[str, Any]:
        """Calcule l'importance des variables par rapport à la cible"""
        
        if target_var is None:
            if self.target_variable:
                target_var = self.target_variable['name']
            else:
                target_var = 'montant_charge_brut'  # Default
        
        if target_var not in df.columns:
            logger.error(f"Variable cible {target_var} non trouvée")
            return {}
        
        target_data = pd.to_numeric(df[target_var], errors='coerce')
        importance_by_group = {}
        
        for group_name, variables in self.feature_groups.items():
            available_vars = [var for var in variables if var in df.columns]
            if not available_vars:
                continue
            
            group_scores = {}
            
            for var in available_vars:
                var_data = df[var]
                
                # Calculer le score d'importance selon le type de variable
                if pd.api.types.is_numeric_dtype(var_data):
                    # Corrélation pour variables numériques
                    correlation = var_data.corr(target_data)
                    score = abs(correlation) if not pd.isna(correlation) else 0
                    
                elif pd.api.types.is_bool_dtype(var_data) or var_data.dtype == 'object':
                    # Test ANOVA pour variables catégorielles
                    try:
                        if var_data.dtype == 'object':
                            unique_vals = var_data.nunique()
                            if unique_vals > 10:  # Trop de catégories
                                score = 0
                            else:
                                # Grouper et tester
                                groups = [target_data[var_data == val].dropna() 
                                         for val in var_data.unique() 
                                         if not pd.isna(val)]
                                groups = [g for g in groups if len(g) >= 3]
                                
                                if len(groups) >= 2:
                                    f_stat, p_val = stats.f_oneway(*groups)
                                    score = f_stat if not np.isnan(f_stat) and p_val < 0.05 else 0
                                else:
                                    score = 0
                        else:
                            # Variable booléenne
                            true_vals = target_data[var_data == True].dropna()
                            false_vals = target_data[var_data == False].dropna()
                            
                            if len(true_vals) >= 3 and len(false_vals) >= 3:
                                t_stat, p_val = stats.ttest_ind(true_vals, false_vals)
                                score = abs(t_stat) if not np.isnan(t_stat) and p_val < 0.05 else 0
                            else:
                                score = 0
                    except:
                        score = 0
                else:
                    score = 0
                
                group_scores[var] = {
                    'importance_score': score,
                    'data_type': str(var_data.dtype),
                    'completeness': (var_data.count() / len(df)) * 100,
                    'unique_values': var_data.nunique()
                }
            
            if group_scores:
                # Statistiques du groupe
                scores = [s['importance_score'] for s in group_scores.values()]
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                
                # Top variables du groupe
                top_vars = sorted(group_scores.items(), key=lambda x: x[1]['importance_score'], reverse=True)
                
                importance_by_group[group_name] = {
                    'group_importance': self._get_group_importance(group_name),
                    'variable_count': len(group_scores),
                    'avg_importance_score': avg_score,
                    'max_importance_score': max_score,
                    'top_3_variables': [(var, data['importance_score']) for var, data in top_vars[:3]],
                    'all_scores': group_scores
                }
        
        # Classement des groupes
        group_ranking = sorted(
            [(name, data['avg_importance_score']) for name, data in importance_by_group.items()],
            key=lambda x: x[1], reverse=True
        )
        
        importance_summary = {
            'target_variable': target_var,
            'groups_analyzed': len(importance_by_group),
            'group_ranking': group_ranking,
            'most_important_group': group_ranking[0][0] if group_ranking else None,
            'by_group': importance_by_group
        }
        
        self.analysis_results['importance'] = importance_summary
        logger.info(f"Importance calculée pour {len(importance_by_group)} groupes")
        
        return importance_summary

    def select_top_variables(self, max_per_group: int = 3, min_score: float = 0.05) -> List[str]:
        """Sélectionne les meilleures variables par groupe"""
        
        if 'importance' not in self.analysis_results:
            logger.warning("Calcul de l'importance requis avant sélection")
            return []
        
        selected_vars = []
        selection_details = {}
        
        importance_data = self.analysis_results['importance']['by_group']
        
        for group_name, group_data in importance_data.items():
            group_importance = group_data['group_importance']
            
            # Adapter le max selon l'importance
            if group_importance == 'critical':
                max_vars = max_per_group + 2
            elif group_importance == 'high':
                max_vars = max_per_group + 1
            else:
                max_vars = max_per_group
            
            # Sélectionner les variables
            selected_from_group = []
            for var_name, score in group_data['top_3_variables']:
                if len(selected_from_group) < max_vars and score >= min_score:
                    selected_from_group.append(var_name)
            
            if selected_from_group:
                selected_vars.extend(selected_from_group)
                selection_details[group_name] = {
                    'selected_count': len(selected_from_group),
                    'variables': selected_from_group
                }
        
        self.analysis_results['selection'] = {
            'total_selected': len(selected_vars),
            'selection_criteria': {'max_per_group': max_per_group, 'min_score': min_score},
            'by_group': selection_details,
            'final_variables': selected_vars
        }
        
        logger.info(f"Sélection terminée: {len(selected_vars)} variables retenues")
        return selected_vars

    # Méthodes originales conservées et améliorées
    def define_feature_groups(self, custom_groups: Dict[str, List[str]] = None) -> None:
        """Définit ou met à jour les groupes de variables"""
        if custom_groups:
            self.feature_groups.update(custom_groups)
            logger.info(f"Groupes personnalisés ajoutés: {list(custom_groups.keys())}")
        else:
            logger.info("Utilisation des groupes par défaut (déjà initialisés)")

    def get_variables_by_group(self, group_name: str) -> List[str]:
        """Retourne les variables d'un groupe donné"""
        return self.feature_groups.get(group_name, [])

    def get_numeric_variables(self, df: pd.DataFrame, exclude_target: bool = True) -> List[str]:
        """Retourne la liste des variables numériques"""
        numeric_vars = df.select_dtypes(include=['number']).columns.tolist()
        
        if exclude_target and self.target_variable:
            target_name = self.target_variable['name']
            if target_name in numeric_vars:
                numeric_vars.remove(target_name)
        
        return numeric_vars

    def get_categorical_variables(self, df: pd.DataFrame) -> List[str]:
        """Retourne la liste des variables catégorielles"""
        return df.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()

    def get_analysis_summary(self) -> str:
        """Génère un résumé textuel des analyses"""
        
        if not self.analysis_results:
            return "Aucune analyse effectuée."
        
        lines = ["=== RÉSUMÉ D'ANALYSE DES VARIABLES ===\n"]
        
        if 'availability' in self.analysis_results:
            avail = self.analysis_results['availability']['summary']
            lines.append(f"📊 DISPONIBILITÉ: {avail['total_available_vars']}/{avail['total_defined_vars']} variables ({avail['overall_availability_pct']:.1f}%)")
            lines.append(f"   Groupes critiques prêts: {len(avail['critical_groups_ready'])}")
            lines.append("")
        
        if 'importance' in self.analysis_results:
            imp = self.analysis_results['importance']
            lines.append("🎯 TOP 5 GROUPES PAR IMPORTANCE:")
            for i, (group, score) in enumerate(imp['group_ranking'][:5], 1):
                lines.append(f"   {i}. {group}: {score:.3f}")
            lines.append("")
        
        if 'selection' in self.analysis_results:
            sel = self.analysis_results['selection']
            lines.append(f"✅ SÉLECTION FINALE: {sel['total_selected']} variables")
            lines.append("   Variables retenues par groupe:")
            for group, details in sel['by_group'].items():
                lines.append(f"   • {group}: {details['selected_count']} variables")
        
        return "\n".join(lines)