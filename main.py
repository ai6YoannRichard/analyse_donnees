from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# --- Imports robustes : module OU script ---
if __package__ is None or __package__ == "":
    # Exécution directe: ajouter le dossier courant au PYTHONPATH
    THIS_DIR = Path(__file__).resolve().parent          # .../analyse_donnees
    sys.path.insert(0, str(THIS_DIR))

    # Imports directs depuis les modules à la racine
    from core.pipeline import SinistreAnalysisPipeline
    from config.settings import DEFAULT_CONFIG, PROJECT_ROOT
else:
    # Exécution comme module: imports relatifs classiques
    from .core.pipeline import SinistreAnalysisPipeline
    from .config.settings import DEFAULT_CONFIG, PROJECT_ROOT

def find_data_file(data_dir: Path) -> Optional[Path]:
    """Recherche intelligente du fichier de données"""
    # Ordre de priorité : Excel > CSV, par ordre d'importance
    search_patterns = [
        "run_04_dataset_imputed.xlsx",
        "run_04_dataset_imputed.csv",
        "*imputed*.xlsx",
        "*imputed*.csv", 
        "sinistres*.xlsx",
        "sinistres*.csv",
        "dataset*.xlsx",
        "dataset*.csv",
        "*.xlsx",
        "*.csv"
    ]
    
    for pattern in search_patterns:
        files = list(data_dir.glob(pattern))
        if files:
            # Prendre le premier fichier trouvé pour ce pattern
            selected_file = files[0]
            print(f"Fichier trouvé: {selected_file.name}")
            if len(files) > 1:
                print(f"  (Note: {len(files)} fichiers correspondent à '{pattern}', utilisation du premier)")
            return selected_file
    
    return None

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Valide et corrige la configuration"""
    validated_config = config.copy()
    
    # S'assurer que les clés essentielles existent
    if 'filter_params' not in validated_config:
        validated_config['filter_params'] = {}
    
    if 'export_params' not in validated_config:
        validated_config['export_params'] = {'format': 'all'}
    
    # Valider les paramètres de filtre
    filter_params = validated_config['filter_params']
    if 'date_range' in filter_params:
        try:
            start_date, end_date = filter_params['date_range']
            # Validation basique du format de date
            if not isinstance(start_date, str) or not isinstance(end_date, str):
                print("Avertissement: Format de date invalide dans date_range")
                del filter_params['date_range']
        except (ValueError, TypeError):
            print("Avertissement: date_range mal formaté, suppression du paramètre")
            del filter_params['date_range']
    
    return validated_config

def display_analysis_summary(pipeline: SinistreAnalysisPipeline) -> None:
    """Affiche un résumé détaillé de l'analyse"""
    try:
        summary = pipeline.get_pipeline_summary()
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'ANALYSE")
        print("="*60)
        
        # État du pipeline
        status = summary.get('pipeline_status', {})
        completed = summary.get('completed_steps', 0)
        total = summary.get('total_steps', 0)
        
        print(f"\nÉtat du pipeline: {completed}/{total} étapes complétées")
        for step, success in status.items():
            status_icon = "✓" if success else "✗"
            step_name = step.replace('_', ' ').title()
            print(f"  {status_icon} {step_name}")
        
        # Informations sur les données
        data_shapes = summary.get('data_shapes', {})
        if data_shapes.get('raw_data'):
            print(f"\nDonnées brutes: {data_shapes['raw_data'][0]:,} × {data_shapes['raw_data'][1]} colonnes")
        if data_shapes.get('processed_data'):
            print(f"Données traitées: {data_shapes['processed_data'][0]:,} × {data_shapes['processed_data'][1]} colonnes")
        
        # Variable cible
        target_var = summary.get('target_variable')
        if target_var:
            print(f"\nVariable cible: {target_var['name']}")
            print(f"Description: {target_var.get('description', 'N/A')}")
        
        # Analyse des variables
        var_analysis = summary.get('variable_analysis_summary', {})
        if var_analysis:
            print(f"\nAnalyse des variables:")
            print(f"  • Groupes définis: {var_analysis.get('groups_defined', 0)}")
            print(f"  • Analyses effectuées: {len(var_analysis.get('analysis_results', []))}")
            print(f"  • Métadonnées définies: {var_analysis.get('metadata_defined', 0)}")
        
        # Diagnostic de santé
        if hasattr(pipeline, 'get_pipeline_health_check'):
            health = pipeline.get_pipeline_health_check()
            print(f"\nSanté du pipeline: {health['overall_status'].upper()}")
            
            if health.get('issues'):
                print("  Problèmes détectés:")
                for issue in health['issues']:
                    print(f"    • {issue}")
            
            if health.get('warnings'):
                print("  Avertissements:")
                for warning in health['warnings']:
                    print(f"    • {warning}")
            
            if health.get('recommendations'):
                print("  Recommandations:")
                for rec in health['recommendations']:
                    print(f"    • {rec}")
        
    except Exception as e:
        print(f"Erreur lors de l'affichage du résumé: {e}")

def main() -> None:
    """Fonction principale d'exécution de l'analyse"""
    
    print("="*60)
    print("ANALYSE DES SINISTRES SÉCHERESSE")
    print("="*60)
    
    # Configuration des chemins
    try:
        # Si PROJECT_ROOT n'est pas défini, utiliser le répertoire du script
        if 'PROJECT_ROOT' in globals():
            project_root = PROJECT_ROOT
        else:
            project_root = Path(__file__).resolve().parent
            
        data_dir = project_root / "data"
        output_dir = project_root / "outputs"
        
        # Créer le dossier de sortie s'il n'existe pas
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dossier de données: {data_dir}")
        print(f"Dossier de sortie: {output_dir}")
        
    except Exception as e:
        print(f"Erreur lors de la configuration des chemins: {e}")
        return
    
    # Recherche du fichier de données
    print(f"\nRecherche du fichier de données dans: {data_dir}")
    data_path = find_data_file(data_dir)
    
    if data_path is None:
        print(f"\nAucun fichier de données trouvé dans: {data_dir}")
        print("\nFichiers supportés:")
        print("  • Excel: *.xlsx")
        print("  • CSV: *.csv")
        print("\nSuggestions:")
        print("  1. Placez votre fichier dans le dossier 'data/'")
        print("  2. Vérifiez que le fichier n'est pas corrompu")
        print("  3. Nommez votre fichier 'run_04_dataset_imputed.xlsx' pour une détection automatique")
        return
    
    print(f"Fichier sélectionné: {data_path}")
    print(f"Taille du fichier: {data_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Configuration personnalisée
    try:
        # Configuration par défaut si DEFAULT_CONFIG n'est pas disponible
        if 'DEFAULT_CONFIG' in globals():
            config = validate_config(DEFAULT_CONFIG.copy())
        else:
            config = {
                "filter_params": {},
                "export_params": {"format": "all"},
                "stats_params": {"correlation_threshold": 0.1}
            }
        
        # Configuration spécialisée sécheresse
        config.update({
            "filter_params": {
                **config.get("filter_params", {}),
                "apply_quality": True,
                "apply_business": True,
                # Dates par défaut pour sécheresse récente
                "date_range": ("2021-01-01", "2024-12-31"),
            },
            "export_params": {
                **config.get("export_params", {}),
                "format": "all"
            },
            "stats_params": {
                "correlation_threshold": 0.1
            },
            # Variables clés spécialisées sécheresse
            "key_variables": [
                "surface_police", "prime", "risque_rga_gcl", 
                "sma_mean", "compo_sol_has_argile", "weather_precip_mean",
                "tree_dist_5", "age_batiment"
            ],
        })
        
        print(f"\nConfiguration validée: {len(config)} paramètres")
        
    except Exception as e:
        print(f"Erreur lors de la configuration: {e}")
        print("Utilisation de la configuration par défaut...")
        config = {"filter_params": {}, "export_params": {"format": "all"}}
    
    # Exécution de l'analyse
    try:
        print("\n" + "-"*60)
        print("DÉMARRAGE DE L'ANALYSE")
        print("-"*60)
        
        # Initialisation du pipeline
        print("Initialisation du pipeline d'analyse...")
        pipeline = SinistreAnalysisPipeline(config)
        
        # Paramètres d'analyse
        analysis_params = {
            "load_params": {},  # Paramètres pour le chargement
            "filter_params": config.get("filter_params", {}),
            "preprocess_params": {"apply_encoding": True},
            "stats_params": config.get("stats_params", {}),
            "key_variables": config.get("key_variables"),
            "export_format": config.get("export_params", {}).get("format", "all"),
            "output_dir": str(output_dir)
        }
        
        print(f"Démarrage de l'analyse: {data_path.name}")
        
        # Exécution de l'analyse complète
        export_paths = pipeline.run_full_analysis(
            file_path=str(data_path),
            target_variable="montant_charge_brut",
            **analysis_params
        )
        
        # Affichage des résultats
        print("\n" + "="*60)
        print("ANALYSE TERMINÉE AVEC SUCCÈS")
        print("="*60)
        
        # Fichiers générés
        if export_paths:
            print(f"\nFichiers générés ({len(export_paths)}):")
            for export_type, file_path in export_paths.items():
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    file_size = file_path_obj.stat().st_size / 1024  # KB
                    print(f"  • {export_type.replace('_', ' ').title()}: {file_path_obj.name} ({file_size:.1f} KB)")
                else:
                    print(f"  • {export_type.replace('_', ' ').title()}: {file_path} (fichier manquant)")
        else:
            print("\nAucun fichier généré (vérifiez les logs pour les erreurs)")
        
        # Résumé détaillé de l'analyse
        display_analysis_summary(pipeline)
        
    except FileNotFoundError as e:
        print(f"\nFICHIER NON TROUVÉ: {e}")
        print("Vérifiez le chemin du fichier et son existence.")
        
    except ImportError as e:
        print(f"\nERREUR D'IMPORT: {e}")
        print("Vérifiez que toutes les dépendances sont installées:")
        print("  pip install pandas numpy scipy scikit-learn openpyxl xlrd matplotlib seaborn")
        
    except KeyError as e:
        print(f"\nCOLONNE MANQUANTE: {e}")
        print("\nVérifications suggérées:")
        print("1. Variable cible présente:")
        print("   - 'montant_charge_brut' (recommandé)")
        print("   - ou 'montant_sinistre', 'cout_total', etc.")
        print("\n2. Colonnes essentielles pour sécheresse:")
        print("   - 'date_sinistre' (dates des sinistres)")
        print("   - 'surface_police' (surface du bien)")
        print("   - 'risque_rga_gcl' (risque retrait-gonflement)")
        print("   - Variables météo/sol (sma_*, weather_*, compo_sol_*)")
        
    except MemoryError:
        print("\nERREUR MÉMOIRE: Jeu de données trop volumineux")
        print("Solutions suggérées:")
        print("1. Filtrer les données par période plus restreinte")
        print("2. Échantillonner les données")
        print("3. Augmenter la mémoire disponible")
        
    except Exception as e:
        print(f"\nERREUR INATTENDUE: {e}")
        print(f"\nType d'erreur: {type(e).__name__}")
        
        print("\nVérifications suggérées:")
        print(f"1. Format du fichier: {data_path.suffix if 'data_path' in locals() else 'N/A'} est-il supporté?")
        print("2. Fichier non corrompu et lisible?")
        print("3. Encodage correct (UTF-8 recommandé pour CSV)?")
        print("4. Permissions d'écriture dans le dossier outputs/?")
        print("5. Espace disque suffisant?")
        
        print(f"\nTrace détaillée de l'erreur:")
        traceback.print_exc()
        
        # Essayer d'obtenir plus d'informations si le pipeline existe
        try:
            if 'pipeline' in locals():
                health = pipeline.get_pipeline_health_check()
                print(f"\nÉtat du pipeline: {health['overall_status']}")
                if health.get('issues'):
                    print("Problèmes détectés:")
                    for issue in health['issues']:
                        print(f"  - {issue}")
        except:
            pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalyse interrompue par l'utilisateur")
    except Exception as e:
        print(f"\nERREUR CRITIQUE: {e}")
        traceback.print_exc()