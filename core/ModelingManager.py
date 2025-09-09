from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

logger = logging.getLogger(__name__)

@dataclass
class ModelResult:
    """Structure pour stocker les résultats d'un modèle"""
    model_name: str
    model: Any
    train_score: float
    test_score: float
    cv_mean: float
    cv_std: float
    rmse: float
    mae: float
    predictions: np.ndarray
    feature_importance: Dict[str, float] = None
    statsmodel_summary: str = None

class ModelingManager:
    """Gestionnaire de modélisation avancée pour sinistres sécheresse"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.results: Dict[str, ModelResult] = {}
        self.preprocessor: Optional[ColumnTransformer] = None
        self.target_transformer: Optional[Any] = None
        self.feature_names: List[str] = []
        self.preprocessing_info: Dict[str, Any] = {}
        
        # Configuration par défaut
        self.default_config = {
            'target_transformation': 'log',  # 'log', 'sqrt', 'none'
            'imputation_numeric': 'median',  # 'mean', 'median', 'knn'
            'imputation_categorical': 'most_frequent',
            'encoding_method': 'onehot',  # 'onehot', 'label', 'target'
            'scaling': True,
            'cv_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        }
        self.config = {**self.default_config, **self.config}

    def prepare_data(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prépare les données avec toutes les transformations nécessaires"""
        
        logger.info("Début de la préparation des données")
        
        # 1. Extraire et nettoyer les données
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Supprimer les lignes avec target manquant
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Données après nettoyage: {X.shape[0]} observations, {X.shape[1]} variables")
        
        # 2. Transformation de la variable cible
        self.target_transformer = self._prepare_target_transformation(y)
        y_transformed = self._transform_target(y)
        
        # 3. Préparer le préprocesseur pour les features
        self.preprocessor = self._create_preprocessor(X)
        
        # 4. Ajuster et transformer les features
        X_transformed = self.preprocessor.fit_transform(X)
        
        # 5. Stocker les noms des features après transformation
        self.feature_names = self._get_feature_names_after_preprocessing(X)
        
        # 6. Stocker les informations de préprocessing
        self._store_preprocessing_info(X, y)
        
        logger.info(f"Données transformées: {X_transformed.shape[1]} features après préprocessing")
        
        return X_transformed, y_transformed

    def _prepare_target_transformation(self, y: pd.Series) -> Dict[str, Any]:
        """Prépare la transformation de la variable cible"""
        
        transformation_type = self.config['target_transformation']
        transformer_info = {'type': transformation_type}
        
        if transformation_type == 'log':
            # Vérifier si transformation log possible
            min_val = y.min()
            if min_val <= 0:
                # Ajouter une constante pour avoir des valeurs positives
                shift = abs(min_val) + 1
                transformer_info['shift'] = shift
                logger.warning(f"Valeurs négatives détectées. Décalage appliqué: +{shift}")
            else:
                transformer_info['shift'] = 0
                
        elif transformation_type == 'sqrt':
            min_val = y.min()
            if min_val < 0:
                shift = abs(min_val)
                transformer_info['shift'] = shift
                logger.warning(f"Valeurs négatives détectées. Décalage appliqué: +{shift}")
            else:
                transformer_info['shift'] = 0
        
        return transformer_info

    def _transform_target(self, y: pd.Series) -> np.ndarray:
        """Applique la transformation à la variable cible"""
        
        transformer = self.target_transformer
        y_work = y.copy()
        
        if transformer['type'] == 'log':
            if transformer['shift'] > 0:
                y_work = y_work + transformer['shift']
            y_transformed = np.log(y_work)
            
        elif transformer['type'] == 'sqrt':
            if transformer['shift'] > 0:
                y_work = y_work + transformer['shift']
            y_transformed = np.sqrt(y_work)
            
        else:  # 'none'
            y_transformed = y_work.values
        
        return y_transformed

    def _inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
        """Inverse la transformation de la variable cible"""
        
        transformer = self.target_transformer
        
        if transformer['type'] == 'log':
            y_original = np.exp(y_transformed)
            if transformer['shift'] > 0:
                y_original = y_original - transformer['shift']
                
        elif transformer['type'] == 'sqrt':
            y_original = y_transformed ** 2
            if transformer['shift'] > 0:
                y_original = y_original - transformer['shift']
        else:
            y_original = y_transformed
        
        return y_original

    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Crée le préprocesseur pour les features"""
        
        # Identifier les types de colonnes
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
        
        transformers = []
        
        # Pipeline pour variables numériques
        if numeric_features:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.config['imputation_numeric'])),
                ('scaler', StandardScaler() if self.config['scaling'] else 'passthrough')
            ])
            transformers.append(('num', numeric_pipeline, numeric_features))
        
        # Pipeline pour variables catégorielles
        if categorical_features:
            # Filtrer les variables avec trop de catégories
            filtered_categorical = []
            for col in categorical_features:
                unique_count = X[col].nunique()
                if unique_count <= 20:  # Limite raisonnable
                    filtered_categorical.append(col)
                else:
                    logger.warning(f"Variable {col} ignorée: trop de catégories ({unique_count})")
            
            if filtered_categorical:
                if self.config['encoding_method'] == 'onehot':
                    categorical_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy=self.config['imputation_categorical'])),
                        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ])
                else:  # label encoding
                    categorical_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy=self.config['imputation_categorical'])),
                        ('encoder', LabelEncoder())
                    ])
                
                transformers.append(('cat', categorical_pipeline, filtered_categorical))
        
        # Pipeline pour variables booléennes
        if boolean_features:
            boolean_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('converter', 'passthrough')  # Garder comme numériques 0/1
            ])
            transformers.append(('bool', boolean_pipeline, boolean_features))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Ignorer les autres colonnes
        )
        
        return preprocessor

    def _get_feature_names_after_preprocessing(self, X_original: pd.DataFrame) -> List[str]:
        """Récupère les noms des features après préprocessing"""
        
        feature_names = []
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder':
                continue
                
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'bool':
                feature_names.extend(columns)
            elif name == 'cat':
                if self.config['encoding_method'] == 'onehot':
                    # Pour OneHot, récupérer tous les noms générés
                    encoder = transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_names = encoder.get_feature_names_out(columns)
                        feature_names.extend(cat_names)
                    else:
                        # Fallback pour anciennes versions
                        for col in columns:
                            unique_vals = X_original[col].dropna().unique()
                            for val in unique_vals:
                                feature_names.append(f"{col}_{val}")
                else:
                    feature_names.extend(columns)
        
        return feature_names

    def _store_preprocessing_info(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Stocke les informations de préprocessing pour diagnostic"""
        
        self.preprocessing_info = {
            'original_shape': X.shape,
            'target_transformation': self.target_transformer,
            'numeric_features': X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_features': X.select_dtypes(include=['object', 'category']).columns.tolist(),
            'boolean_features': X.select_dtypes(include=['bool']).columns.tolist(),
            'missing_values_before': X.isnull().sum().sum(),
            'target_stats_before': {
                'mean': y.mean(),
                'median': y.median(),
                'std': y.std(),
                'skew': y.skew(),
                'kurt': y.kurtosis()
            }
        }

    def fit_multiple_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, ModelResult]:
        """Entraîne plusieurs modèles et compare les performances"""
        
        logger.info("Début de l'entraînement multi-modèles")
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Configuration des modèles
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.config['random_state']),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.config['random_state']),
            'decision_tree': DecisionTreeRegressor(random_state=self.config['random_state'])
        }
        
        # Entraîner chaque modèle
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Entraînement du modèle: {model_name}")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Validation croisée
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=self.config['random_state']),
                    scoring='r2'
                )
                
                # Métriques
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                mae = mean_absolute_error(y_test, y_pred_test)
                
                # Importance des features (si disponible)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(zip(self.feature_names, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    feature_importance = dict(zip(self.feature_names, np.abs(model.coef_)))
                
                # Créer le résultat
                result = ModelResult(
                    model_name=model_name,
                    model=model,
                    train_score=train_r2,
                    test_score=test_r2,
                    cv_mean=cv_scores.mean(),
                    cv_std=cv_scores.std(),
                    rmse=rmse,
                    mae=mae,
                    predictions=y_pred_test,
                    feature_importance=feature_importance
                )
                
                results[model_name] = result
                
                logger.info(f"{model_name} - R²: {test_r2:.3f}, RMSE: {rmse:.3f}")
                
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {model_name}: {e}")
                continue
        
        self.results = results
        return results

    def fit_statsmodel_regression(self, X: np.ndarray, y: np.ndarray, model_name: str = 'statsmodel_ols') -> ModelResult:
        """Entraîne un modèle avec statsmodels pour analyse statistique approfondie"""
        
        logger.info("Entraînement avec statsmodels pour analyse statistique")
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Ajouter une constante pour l'intercept
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)
        
        # Entraîner le modèle
        model = sm.OLS(y_train, X_train_sm).fit()
        
        # Prédictions
        y_pred_test = model.predict(X_test_sm)
        
        # Métriques
        test_r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae = mean_absolute_error(y_test, y_pred_test)
        
        # Importance des features (coefficients)
        feature_names_with_const = ['const'] + self.feature_names
        coefficients = model.params
        feature_importance = dict(zip(feature_names_with_const[1:], np.abs(coefficients[1:])))  # Exclure constante
        
        # Créer le résultat avec résumé statsmodel
        result = ModelResult(
            model_name=model_name,
            model=model,
            train_score=model.rsquared,
            test_score=test_r2,
            cv_mean=0,  # Pas de CV pour statsmodel dans cette version
            cv_std=0,
            rmse=rmse,
            mae=mae,
            predictions=y_pred_test,
            feature_importance=feature_importance,
            statsmodel_summary=str(model.summary())
        )
        
        self.results[model_name] = result
        logger.info(f"Statsmodel OLS - R²: {test_r2:.3f}, R² ajusté: {model.rsquared_adj:.3f}")
        
        return result

    def analyze_residuals(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Analyse des résidus pour diagnostiquer le modèle"""
        
        if model_name not in self.results:
            logger.error(f"Modèle {model_name} non trouvé")
            return {}
        
        result = self.results[model_name]
        residuals = y_test - result.predictions
        
        # Tests statistiques sur les résidus
        analysis = {
            'residuals_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            },
            'normality_test': {
                'shapiro_stat': stats.shapiro(residuals)[0],
                'shapiro_pvalue': stats.shapiro(residuals)[1],
                'normal_distributed': stats.shapiro(residuals)[1] > 0.05
            }
        }
        
        # Test d'homoscédasticité (si statsmodel)
        if hasattr(result.model, 'resid') and hasattr(result.model, 'fittedvalues'):
            try:
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(result.model.resid, result.model.model.exog)
                analysis['heteroscedasticity'] = {
                    'breusch_pagan_stat': bp_stat,
                    'breusch_pagan_pvalue': bp_pvalue,
                    'homoscedastic': bp_pvalue > 0.05
                }
            except:
                logger.warning("Test d'hétéroscédasticité échoué")
        
        return analysis

    def generate_model_comparison(self) -> pd.DataFrame:
        """Génère un tableau comparatif des modèles"""
        
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train_R2': result.train_score,
                'Test_R2': result.test_score,
                'CV_Mean': result.cv_mean,
                'CV_Std': result.cv_std,
                'RMSE': result.rmse,
                'MAE': result.mae,
                'Overfit_Risk': result.train_score - result.test_score
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
        
        return comparison_df

    def get_feature_importance_summary(self, top_n: int = 15) -> Dict[str, Any]:
        """Résumé de l'importance des features across modèles"""
        
        if not self.results:
            return {}
        
        # Collecter les importances de tous les modèles
        all_importances = {}
        
        for model_name, result in self.results.items():
            if result.feature_importance:
                all_importances[model_name] = result.feature_importance
        
        if not all_importances:
            return {}
        
        # Calculer importance moyenne et rankings
        feature_avg_importance = {}
        
        for feature in self.feature_names:
            importances = []
            for model_name, importance_dict in all_importances.items():
                if feature in importance_dict:
                    importances.append(importance_dict[feature])
            
            if importances:
                feature_avg_importance[feature] = {
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances),
                    'models_count': len(importances)
                }
        
        # Top features
        sorted_features = sorted(
            feature_avg_importance.items(),
            key=lambda x: x[1]['mean_importance'],
            reverse=True
        )
        
        return {
            'top_features': sorted_features[:top_n],
            'total_features_analyzed': len(feature_avg_importance),
            'by_model': all_importances
        }

    def predict_with_confidence(self, X_new: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        """Prédictions avec intervalles de confiance"""
        
        if not model_name:
            # Utiliser le meilleur modèle (plus haut test R²)
            if not self.results:
                logger.error("Aucun modèle entraîné")
                return {}
            
            best_model_name = max(self.results.keys(), key=lambda k: self.results[k].test_score)
            model_name = best_model_name
        
        if model_name not in self.results:
            logger.error(f"Modèle {model_name} non trouvé")
            return {}
        
        result = self.results[model_name]
        
        # Prédictions (transformées)
        predictions_transformed = result.model.predict(X_new)
        
        # Inverse transformation
        predictions_original = self._inverse_transform_target(predictions_transformed)
        
        # Estimation des intervalles (approximation basée sur RMSE)
        rmse_original_scale = result.rmse  # RMSE déjà en échelle originale après inverse transform
        
        # Intervalles approximatifs (95% de confiance)
        lower_bound = predictions_original - 1.96 * rmse_original_scale
        upper_bound = predictions_original + 1.96 * rmse_original_scale
        
        return {
            'predictions': predictions_original,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'model_used': model_name,
            'rmse': rmse_original_scale
        }

    def get_modeling_summary(self) -> str:
        """Résumé complet de la modélisation"""
        
        if not self.results:
            return "Aucun modèle entraîné."
        
        lines = ["=== RÉSUMÉ DE MODÉLISATION ===\n"]
        
        # Configuration
        lines.append("🔧 CONFIGURATION:")
        lines.append(f"   Transformation cible: {self.target_transformer['type']}")
        lines.append(f"   Méthode d'encodage: {self.config['encoding_method']}")
        lines.append(f"   Imputation numérique: {self.config['imputation_numeric']}")
        lines.append(f"   Features finales: {len(self.feature_names)}")
        lines.append("")
        
        # Comparaison des modèles
        comparison = self.generate_model_comparison()
        if not comparison.empty:
            lines.append("🏆 TOP 3 MODÈLES:")
            for i, (_, row) in enumerate(comparison.head(3).iterrows(), 1):
                lines.append(f"   {i}. {row['Model']}: R² = {row['Test_R2']:.3f}, RMSE = {row['RMSE']:.0f}")
            lines.append("")
        
        # Importance des features
        feature_summary = self.get_feature_importance_summary(top_n=10)
        if feature_summary:
            lines.append("🎯 TOP 5 VARIABLES IMPORTANTES:")
            for i, (feature, importance_info) in enumerate(feature_summary['top_features'][:5], 1):
                lines.append(f"   {i}. {feature}: {importance_info['mean_importance']:.3f}")
            lines.append("")
        
        # Diagnostic
        best_model = max(self.results.keys(), key=lambda k: self.results[k].test_score)
        best_result = self.results[best_model]
        
        lines.append("📊 DIAGNOSTIC (MEILLEUR MODÈLE):")
        lines.append(f"   Modèle: {best_model}")
        lines.append(f"   R² test: {best_result.test_score:.3f}")
        lines.append(f"   RMSE: {best_result.rmse:.0f}")
        lines.append(f"   Surapprentissage: {best_result.train_score - best_result.test_score:.3f}")
        
        return "\n".join(lines)