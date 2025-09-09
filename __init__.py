__version__ = "1.0.0"

try:
    from core.pipeline import SinistreAnalysisPipeline
    from core.DataLoader import DataLoader
    from core.DataFilter import DataFilter
    from core.DataPreprocessor import DataPreprocessor
    from core.DataVisualizer import DataVisualizer
    from core.VariableManager import VariableManager
    from core.StatisticsManager import StatisticsManager
    from core.SynthesisManager import SynthesisManager
    
    __all__ = [
        "SinistreAnalysisPipeline",
        "DataLoader",
        "DataFilter", 
        "DataPreprocessor",
        "DataVisualizer",
        "VariableManager",
        "StatisticsManager",
        "SynthesisManager"
    ]
except ImportError as e:
    print(f"Erreur d'import: {e}")
    __all__ = []