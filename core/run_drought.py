# run_drought.py - Version corrigée
from __future__ import annotations
import argparse
import json
import sys
import os

# Ajouter le répertoire parent au path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import absolu
from core.DroughtClaimsAnalyzer import DroughtClaimsAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Analyse complète sécheresse")
    parser.add_argument("file_path", help="Chemin vers le fichier de données (csv/xlsx/parquet)")
    parser.add_argument("--sep", default=";", help="Séparateur CSV (défaut ;)")
    parser.add_argument("--encoding", default="utf-8", help="Encodage (défaut utf-8)")
    args = parser.parse_args()

    analyzer = DroughtClaimsAnalyzer(config={
        "target_variable": "montant_charge_brut",
        "date_column": "date_sinistre",
    })
    

    resultats = analyzer.run_comprehensive_drought_analysis(
        file_path="data/run_04_dataset_imputed.xlsx",
        load_params={"sheet_name": 0},   # ou rien du tout
        preprocess_params={},
        visualizer_params={"monthly_agg": "sum", "monthly_min_count": 3},
    )

    print(json.dumps(resultats.get("insights", {}), ensure_ascii=False, indent=2))

    # 1) Lire les insights métier
    print("\n== INSIGHTS SÉCHERESSE ==")
    insights = resultats.get("insights", {})
    for k, v in insights.items():
        if isinstance(v, list):
            print(f"- {k}:")
            for item in v:
                print(f"  * {item}")
        else:
            print(f"- {k}: {v}")

    # 2) Où sont les figures / rapports
    print("\n== SORTIES ==")
    exports = resultats.get("exports", {})
    if exports:
        for k, v in exports.items():
            print(f"{k}: {v}")
    else:
        print("Aucun export disponible dans les résultats")

if __name__ == "__main__":
    main()