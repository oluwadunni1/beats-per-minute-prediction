import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    PowerTransformer, 
    PolynomialFeatures
)

# ==========================================
# 1. DEFINE FEATURE GROUPS & LISTS
# ==========================================

# Define feature groups based on EDA skewness analysis
feature_groups = {
    'highly_skewed': [
        'InstrumentalScore',      # Skewness: 1.04
        'VocalContent',           # Skewness: 0.79
        'AcousticQuality'         # Skewness: 0.79
    ],
    
    'moderately_skewed': [
        'LivePerformanceLikelihood'  # Skewness: 0.39
    ],
    
    'approximately_normal': [
        'RhythmScore',            # Skewness: 0.17
        'AudioLoudness',          # Skewness: -0.31
        'MoodScore',              # Skewness: -0.27
        'TrackDurationMs',        # Skewness: -0.19
        'Energy'                  # Skewness: -0.03
    ]
}

# Final decision on which features to transform
features_to_transform = feature_groups['highly_skewed'].copy()

# Decision: Include LivePerformanceLikelihood if transformation reduces skewness significantly
# Adjust based on your transformation results
# features_to_transform.append('LivePerformanceLikelihood')  # Uncomment if beneficial

features_normal = feature_groups['approximately_normal'].copy()
if 'LivePerformanceLikelihood' not in features_to_transform:
    features_normal.append('LivePerformanceLikelihood')

# ==========================================
# 2. DEFINE PREPROCESSOR FUNCTIONS
# ==========================================

def create_linear_model_preprocessor():
    """Linear / Regularized models: Yeo-Johnson + StandardScaler"""
    return Pipeline([
        ('transform', ColumnTransformer(
            transformers=[
                ('yeo_johnson',
                 PowerTransformer(method='yeo-johnson', standardize=False),
                 features_to_transform),
                ('passthrough', 'passthrough', features_normal)
            ]
        )),
        ('scaler', StandardScaler())
    ])


def create_tree_model_preprocessor():
    """
    Tree-based models (RF, XGB, LGBM, CatBoost)
    - No scaling or transformations needed
    """
    return 'passthrough'


def create_neural_network_preprocessor():
    """Neural networks: Yeo-Johnson + MinMaxScaler"""
    return Pipeline([
        ('transform', ColumnTransformer(
            transformers=[
                ('yeo_johnson',
                 PowerTransformer(method='yeo-johnson', standardize=False),
                 features_to_transform),
                ('passthrough', 'passthrough', features_normal)
            ]
        )),
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ])


def create_svr_preprocessor():
    """
    SVR is extremely scale-sensitive.
    Uses same preprocessing as linear models.
    """
    return create_linear_model_preprocessor()

def create_polynomial_preprocessor(degree=2, interaction_only=False):
    """
    Polynomial regression base:
    - Yeo-Johnson transform
    - Polynomial feature expansion
    - StandardScaler
    """
    return Pipeline([
        ('transform', ColumnTransformer(
            transformers=[
                ('yeo_johnson',
                 PowerTransformer(method='yeo-johnson', standardize=False),
                 features_to_transform),
                ('passthrough', 'passthrough', features_normal)
            ]
        )),
        ('polynomial', PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )),
        ('scaler', StandardScaler())
    ])


# ---------- Polynomial Helper Functions ----------
def create_polynomial_interaction_preprocessor():
    """Polynomial degree 2: interactions only"""
    return create_polynomial_preprocessor(degree=2, interaction_only=True)

def create_polynomial_full_preprocessor():
    """Polynomial degree 2: all terms"""
    return create_polynomial_preprocessor(degree=2, interaction_only=False)


# ==========================================
# 3. DEFINE PIPELINE FACTORY
# ==========================================

pipeline_factory = {
    # Linear family
    'linear': create_linear_model_preprocessor,
    'ridge': create_linear_model_preprocessor,
    'lasso': create_linear_model_preprocessor,
    'elasticnet': create_linear_model_preprocessor,
    'gam': create_linear_model_preprocessor,

    # Kernel methods
    'svr': create_svr_preprocessor,

    # Tree-based
    'tree': create_tree_model_preprocessor,
    'random_forest': create_tree_model_preprocessor,
    'xgboost': create_tree_model_preprocessor,
    'lightgbm': create_tree_model_preprocessor,
    'catboost': create_tree_model_preprocessor,

    # Neural networks
    'neural_network': create_neural_network_preprocessor,

   # Polynomial regression
    'polynomial_interact': create_polynomial_interaction_preprocessor,
    'polynomial_full': create_polynomial_full_preprocessor
}

if __name__ == "__main__":
    print("✓ Preprocessing module loaded.")
    print(f"Features to transform: {features_to_transform}")
    print(f"Features to scale only: {features_normal}")
    print(f"Available pipelines: {list(pipeline_factory.keys())}")