
import sys
from pathlib import Path


# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def _resolve_input_path(user_arg: str | None) -> Path:
    """Choisit le chemin du fichier d'entrée (utilisateur ou défauts)."""
    if user_arg:
        p = Path(user_arg)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    # Chemins par défaut (1) demandé, (2) fallback connu
    candidates = [        
        PROJECT_ROOT / "data" / "run_04_dataset_imputed.xlsx",  # fallback si le 1er n'existe pas
    ]
    for c in candidates:
        if c.exists():
            return c
    # Si aucun n'existe, on retourne le premier (demandé) quand même
    return candidates[0]

def main():
    try:
        # Import après avoir réglé le sys.path
        from core.run_drought import main as drought_main
    except ImportError as e:
        print(f"[ERREUR] Import échoué: {e}")
        print("Vérifie que le dossier 'core/' contient bien run_drought.py et les modules requis.")
        sys.exit(1)

    # Résoudre le fichier cible (arg utilisateur ou défaut)
    user_arg = sys.argv[1] if len(sys.argv) > 1 else None
    input_path = _resolve_input_path(user_arg)

    # Injecter l'argument dans sys.argv pour mimer l'appel CLI attendu par run_drought.py
    sys.argv = [sys.argv[0], str(input_path)]

    # Messages utiles
    if user_arg is None:
        print(f"[INFO] Aucun fichier spécifié. Utilisation du fichier par défaut : {input_path}")
        if not input_path.exists():
            print("[AVERTISSEMENT] Le fichier par défaut n'existe pas. "
                  "Merci d'en déposer un à cet emplacement ou de passer un chemin en argument.")

    try:
        drought_main()  # run_drought.py se charge de lire sys.argv[1]
    except Exception as e:
        print(f"[ERREUR] Pendant l'exécution : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()