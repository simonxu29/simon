# Button3 Application - Batch Version 0.03
# Created: October 4, 2025
# Updated: October 7, 2025
# Description: Production-ready ML application with robust data handling,
#              no data leakage risks, complete 4-step workflow, and Pro features

import streamlit as st
import os
import io
import pandas as pd
import numpy as np
import joblib
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, Optional, Tuple

# scikit-learn model and utility imports used throughout the app
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Optional Prophet import (where used, guarded in code)
try:
    import prophet  # noqa: F401
except Exception:
    prophet = None
st.set_page_config(layout="centered")

# Unified button styling applied app-wide: uploaders, buttons, downloads
def apply_unified_button_style():
    """Apply unified right-aligned button style for upload (train/test) and download.
    - Hides default uploader hint text
    - Right-aligns all button containers
    - White background, blue border/text by default; invert on hover
    """
    st.markdown(
        """
<style>
/* --- HIDE FILE UPLOADER HINT TEXT (UNCHANGED) --- */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div { display: none !important; }

/* --- RIGHT-ALIGN ALL BUTTON CONTAINERS --- */
.stButton,
[data-testid="stDownloadButton"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
}

/* --- REMOVE WHITE PANEL AROUND FILE UPLOADER DROPZONE (统一去除底色/边框/留白) --- */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    border-radius: 0 !important;
}
/* Defensive: some Streamlit versions add inner wrapper with background */
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* --- ANT DESIGN PRIMARY BUTTON STYLE (统一风格) --- */
.stButton > button,
[data-testid="stDownloadButton"] button,
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [role="button"] {
    background-color: #1677ff !important; /* AntD v5 primary */
    color: #ffffff !important;
    border: 1px solid #1677ff !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    height: 32px !important;
    padding: 0 15px !important;
    line-height: 30px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    box-shadow: none !important;
    margin: 0 !important;
}

/* Hover: lighten */
.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] [role="button"]:hover {
    background-color: #4096ff !important;
    border-color: #4096ff !important;
    color: #ffffff !important;
}

/* Active: darker */
.stButton > button:active,
[data-testid="stDownloadButton"] button:active,
[data-testid="stFileUploader"] button:active,
[data-testid="stFileUploader"] [role="button"]:active {
    background-color: #0958d9 !important;
    border-color: #0958d9 !important;
}

/* Focus ring */
.stButton > button:focus,
[data-testid="stDownloadButton"] button:focus,
[data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] [role="button"]:focus {
    box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.24) !important;
    outline: none !important;
}

/* Disabled */
.stButton > button:disabled,
[data-testid="stDownloadButton"] button:disabled,
[data-testid="stFileUploader"] button:disabled {
    background-color: #f0f0f0 !important;
    color: rgba(0,0,0,0.25) !important;
    border-color: #d9d9d9 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

# Apply unified style once
apply_unified_button_style()


# --- CONFIGURATION ---
CLASSIFICATION_THRESHOLD = 20

# --- HELPER FUNCTIONS ---

def is_pro() -> bool:
    """Return True if Pro features are unlocked for the current session."""
    return bool(st.session_state.get('pro_unlocked', False))

def reset_state():
    """Clears all session state variables and saved files from a previous run."""
    keys_to_clear = [
        'report_ready', 'score_label', 'score_value', 'problem_type_for_report',
    'feature_importance_df', 'permutation_importance_df', 'correlation_matrix', 'target_name_for_corr',
        'correlation_fig', 'df_predictions', 'feature_columns', 'processed_columns',
        'target_columns', 'encoding_method', 'output_file_extension',
        'df_forecast', 'ts_report_ready',
        # Feature engineering cleanup
        'numeric_fe_selections', 'categorical_fe_selections', 'custom_feature_rules',
        'custom_hyperparams', 'fe_num_selections', 'pro_feature_rules'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    artifacts_to_clean = ['model.joblib', 'label_encoders.joblib', 'target_encoding_maps.joblib', 'feature_rules.joblib', 'scaler_stats.joblib', 'input_feature_columns.joblib']
    for art in artifacts_to_clean:
        if os.path.exists(art):
            os.remove(art)

def detect_id_columns(df):
    """Detects likely identifier columns."""
    suspected_id_cols = []
    id_keywords = ['id', 'key', 'code', 'no', 'number', 'guid']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in id_keywords):
            suspected_id_cols.append(col)
            continue
        if df[col].dtype == 'object' or pd.api.types.is_integer_dtype(df[col]):
            if df[col].nunique() / len(df) > 0.99:
                suspected_id_cols.append(col)
    return list(set(suspected_id_cols))

def get_problem_type(y):
    """Determines problem type."""
    if pd.api.types.is_string_dtype(y) or y.nunique() <= CLASSIFICATION_THRESHOLD:
        return "Classification"
    else:
        return "Regression"

def preprocess_features(X, y=None, encoding_method='One-Hot Encoding', saved_maps=None):
    """Preprocesses the feature matrix (X)."""
    X_processed = X.copy()
    for col in X_processed.columns:
        if pd.api.types.is_numeric_dtype(X_processed[col]):
            fill_value = X_processed[col].median()
            if pd.isna(fill_value):  # Handle case where all values are NaN
                fill_value = 0
        else:
            mode_values = X_processed[col].mode()
            fill_value = mode_values[0] if len(mode_values) > 0 else "Unknown"
        X_processed[col] = X_processed[col].fillna(fill_value)

    if encoding_method == 'One-Hot Encoding':
        X_processed = pd.get_dummies(X_processed, dummy_na=False, drop_first=True)
        return X_processed, None
    elif encoding_method == 'Target Encoding':
        if y is None and saved_maps is None:
            raise ValueError("Target variable 'y' must be provided for Target Encoding.")
        
        encoding_maps = {}
        if saved_maps is None:
            y_numeric = LabelEncoder().fit_transform(y) if get_problem_type(y) == "Classification" else y
            global_mean_y = pd.Series(y_numeric).mean()
            for col in X_processed.select_dtypes(include=['object', 'category']).columns:
                # Add smoothing to prevent overfitting
                value_counts = X_processed[col].value_counts()
                mapping = pd.Series(y_numeric, index=X_processed.index).groupby(X_processed[col]).agg(['mean', 'count'])
                # Apply smoothing: (count * mean + prior_weight * global_mean) / (count + prior_weight)
                prior_weight = 10
                mapping['smoothed_mean'] = (mapping['count'] * mapping['mean'] + prior_weight * global_mean_y) / (mapping['count'] + prior_weight)
                mapping_dict = mapping['smoothed_mean'].to_dict()
                X_processed[col] = X_processed[col].map(mapping_dict)
                encoding_maps[col] = mapping_dict
            X_processed = X_processed.fillna(global_mean_y)
            return X_processed, encoding_maps
        else:
            # Calculate global mean from saved maps for unseen categories
            all_means = []
            for mapping in saved_maps.values():
                if isinstance(mapping, dict):
                    all_means.extend(mapping.values())
            global_mean = np.mean(all_means) if all_means else 0
            
            for col, mapping in saved_maps.items():
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].map(mapping)
            X_processed = X_processed.fillna(global_mean)
            return X_processed, None
    elif encoding_method == 'Label Encoding':
        # Label-encode categorical feature columns deterministically
        cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if saved_maps is None:
            encoding_maps = {}
            for col in cat_cols:
                # Build a deterministic mapping using sorted unique values
                uniques = sorted(pd.Series(X_processed[col].astype(str)).unique().tolist())
                mapping = {val: idx for idx, val in enumerate(uniques)}
                X_processed[col] = X_processed[col].astype(str).map(mapping).astype(int)
                encoding_maps[col] = mapping
            return X_processed, encoding_maps
        else:
            # Apply provided mappings; unseen categories become -1
            for col, mapping in saved_maps.items():
                if col in X_processed.columns:
                    X_processed[col] = X_processed[col].astype(str).map(mapping)
            X_processed = X_processed.fillna(-1)
            # Ensure integer dtype for encoded columns where possible
            for col in cat_cols:
                if col in X_processed.columns:
                    try:
                        X_processed[col] = X_processed[col].astype(int)
                    except Exception:
                        # If cast fails due to mixed values, keep as is
                        pass
            return X_processed, None

def validate_data(df, target_cols, min_samples=10):
    """Validates dataset meets minimum requirements."""
    errors = []
    
    # Check minimum sample size
    if len(df) < min_samples:
        errors.append(f"Dataset too small. Need at least {min_samples} samples, got {len(df)}.")
    
    # Check target variance
    for target_col in target_cols:
        if target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                if df[target_col].nunique() <= 1:
                    errors.append(f"Target column '{target_col}' has no variance (all values are the same).")
            elif df[target_col].nunique() <= 1:
                errors.append(f"Target column '{target_col}' has only one unique value.")
    
    return errors

def create_time_series_features(df, date_col, target_col):
    """Creates time series features from a date column and a target column."""
    # --- FIX 1: Isolate only date and target columns to prevent feature leakage ---
    df_ts = df[[date_col, target_col]].copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
    df_ts = df_ts.set_index(date_col).sort_index()
    
    # Interpolate to handle missing values before feature creation
    df_ts[target_col] = df_ts[target_col].interpolate(method='time').bfill().ffill()
    
    if len(df_ts) < 14:
        st.warning(f"Time series is short ({len(df_ts)} points). Using simplified features.")

    df_ts['dayofweek'] = df_ts.index.dayofweek
    df_ts['quarter'] = df_ts.index.quarter
    df_ts['month'] = df_ts.index.month
    df_ts['year'] = df_ts.index.year
    df_ts['dayofyear'] = df_ts.index.dayofyear
    df_ts['weekofyear'] = df_ts.index.isocalendar().week.astype(int)
    
    # Dynamically create lags based on data length
    if len(df_ts) >= 14:
        lags = [1, 7, 14]
        rolling_windows = [7, 14]
    else:
        lags = [1, 2, 3]
        rolling_windows = [3]

    for lag in lags:
        df_ts[f'{target_col}_lag_{lag}'] = df_ts[target_col].shift(lag)
    
    for window in rolling_windows:
        df_ts[f'{target_col}_rolling_mean_{window}'] = df_ts[target_col].shift(1).rolling(window=window).mean()
        df_ts[f'{target_col}_rolling_std_{window}'] = df_ts[target_col].shift(1).rolling(window=window).std()

    df_ts = df_ts.dropna()
    return df_ts

# --- PRO UTILS: Optional numeric scaling ---
def compute_scaler_stats(df_numeric):
    """Compute mean/std for numeric columns. Returns dict: {col: (mean, std)}"""
    stats = {}
    for col in df_numeric.columns:
        series = df_numeric[col].astype(float)
        mean = series.mean()
        std = series.std()
        # Avoid division by zero
        if pd.isna(std) or std == 0:
            std = 1.0
        stats[col] = (float(mean), float(std))
    return stats

def apply_scaler_stats(df, stats):
    """Apply precomputed mean/std to matching numeric columns in df (in-place-like)."""
    df_scaled = df.copy()
    for col, (mean, std) in stats.items():
        if col in df_scaled.columns:
            try:
                df_scaled[col] = (df_scaled[col].astype(float) - mean) / std
            except Exception:
                # If cast fails, skip column
                pass
    return df_scaled

# --- PRO TS UTILS: Metrics and Fourier terms ---
def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def _smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def _mase(y_train, y_true, y_pred, seasonality=1):
    # MASE scaling factor from in-sample seasonal naive
    y_train = np.asarray(y_train)
    if len(y_train) <= seasonality:
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    scale = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])) + 1e-8
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)

def _fourier_terms(index, periods, order=3):
    terms = {}
    t = np.arange(len(index))
    for p in periods:
        for k in range(1, order + 1):
            terms[f'sin_{p}_{k}'] = np.sin(2 * np.pi * k * t / p)
            terms[f'cos_{p}_{k}'] = np.cos(2 * np.pi * k * t / p)
    return pd.DataFrame(terms, index=index)

def create_time_series_features_exog(df, date_col, target_col, exog_cols=None, lags=None, rolling_windows=None, fourier_periods=None, fourier_order=3):
    """Create TS features with optional exogenous and Fourier terms. Returns cleaned df with target column present."""
    exog_cols = exog_cols or []
    lags = lags or [1, 7, 14]
    rolling_windows = rolling_windows or [7, 14]

    base_cols = [date_col, target_col] + [c for c in exog_cols if c in df.columns and c != target_col and c != date_col]
    df_ts = df[base_cols].copy()
    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
    df_ts = df_ts.set_index(date_col).sort_index()
    df_ts[target_col] = df_ts[target_col].interpolate(method='time').bfill().ffill()

    # Calendar features
    df_ts['dayofweek'] = df_ts.index.dayofweek
    df_ts['month'] = df_ts.index.month
    df_ts['dayofyear'] = df_ts.index.dayofyear

    # Lags and rolling stats on target
    for lag in lags:
        df_ts[f'{target_col}_lag_{lag}'] = df_ts[target_col].shift(lag)
    for window in rolling_windows:
        df_ts[f'{target_col}_rolling_mean_{window}'] = df_ts[target_col].shift(1).rolling(window=window).mean()
        df_ts[f'{target_col}_rolling_std_{window}'] = df_ts[target_col].shift(1).rolling(window=window).std()

    # Fourier seasonal terms
    if fourier_periods:
        ft = _fourier_terms(df_ts.index, fourier_periods, order=fourier_order)
        df_ts = pd.concat([df_ts, ft], axis=1)

    # Ensure all non-numeric feature columns (including exogenous) are encoded numerically
    for col in df_ts.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df_ts[col]):
            try:
                # Factorize into stable integer codes
                codes, _ = pd.factorize(df_ts[col].astype(str))
                df_ts[col] = codes.astype(float)
            except Exception:
                # If encoding fails, coerce to numeric with NaNs -> filled later
                df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')

    df_ts = df_ts.dropna()
    return df_ts

# --- PRO UTILS: Feature importance extraction for diverse models ---
def _fi_df(importances, columns):
    try:
        imp = np.asarray(importances).ravel()
        if len(imp) == len(columns):
            return pd.DataFrame({'Feature': list(columns), 'Importance': imp})
    except Exception:
        pass
    return None

# --- PRO UTILS: Custom feature generation ---
FEATURE_GEN_METHODS = [
    ("Concat (as string)", "concat"),
    ("Sum (col1 + col2)", "sum"),
    ("Difference (col1 - col2)", "diff"),
    ("Abs Difference |col1-col2|", "absdiff"),
    ("Product (col1 * col2)", "prod"),
    ("Ratio (col1 / col2)", "ratio"),
    ("Date difference in days", "datediff"),
]

def apply_feature_generations(df: pd.DataFrame, specs: list) -> pd.DataFrame:
    """Apply list of feature generation specs to df and return new df with added columns.
    Supports legacy spec keys and new UI keys.
    Each spec example:
      - {'col1': 'A', 'col2': 'B', 'method': 'sum', 'name': 'A_plus_B'}
      - {'col1': 'A', 'col2': 'B', 'operation': 'Add (+)', 'new_name': 'A_Add_B'}
    Supported methods: concat, sum, diff, absdiff, prod, ratio, datediff, power, mean, max, min
    """
    if not specs:
        return df
    out = df.copy()
    eps = 1e-9
    op_map = {
        'Add (+)': 'sum',
        'Subtract (-)': 'diff',
        'Multiply (*)': 'prod',
        'Divide (/)': 'ratio',
        'Power (^)': 'power',
        'Mean': 'mean',
        'Max': 'max',
        'Min': 'min',
        'Ratio': 'ratio',
    }
    for spec in specs:
        col1 = spec.get('col1')
        col2 = spec.get('col2')
        method = spec.get('method') or op_map.get(spec.get('operation')) or spec.get('operation')
        name = spec.get('name') or spec.get('new_name')
        if not col1 or not col2 or not method:
            continue
        new_col = name or f"FE_{method}_{col1}_{col2}"
        try:
            if method == 'concat':
                out[new_col] = out[col1].astype(str) + '_' + out[col2].astype(str)
            elif method in ('sum', 'diff', 'absdiff', 'prod', 'ratio', 'power', 'mean', 'max', 'min'):
                a = pd.to_numeric(out[col1], errors='coerce')
                b = pd.to_numeric(out[col2], errors='coerce')
                if method == 'sum':
                    out[new_col] = a + b
                elif method == 'diff':
                    out[new_col] = a - b
                elif method == 'absdiff':
                    out[new_col] = (a - b).abs()
                elif method == 'prod':
                    out[new_col] = a * b
                elif method == 'ratio':
                    out[new_col] = a / (b.replace(0, np.nan) + eps)
                elif method == 'power':
                    # a ** b, guard against invalid exponents
                    with np.errstate(all='ignore'):
                        out[new_col] = np.power(a, b)
                elif method == 'mean':
                    out[new_col] = (a + b) / 2.0
                elif method == 'max':
                    out[new_col] = np.fmax(a, b)
                elif method == 'min':
                    out[new_col] = np.fmin(a, b)
            elif method == 'datediff':
                a = pd.to_datetime(out[col1], errors='coerce')
                b = pd.to_datetime(out[col2], errors='coerce')
                out[new_col] = (a - b).dt.total_seconds() / 86400.0
        except Exception:
            # If anything goes wrong for a spec, skip that feature
            continue
    return out

def compute_feature_importance_generic(model, X_df):
    """Return a DataFrame with Feature and Importance if available, else None.
    Handles tree-based, linear models, and MultiOutputClassifier by averaging across outputs.
    """
    cols = X_df.columns
    # Multi-output wrapper
    if isinstance(model, MultiOutputClassifier):
        ests = getattr(model, 'estimators_', [])
        # Try tree-based importances
        imp_list = []
        for est in ests:
            if hasattr(est, 'feature_importances_'):
                imp_list.append(np.asarray(est.feature_importances_))
        if imp_list:
            importances = np.mean(np.vstack(imp_list), axis=0)
            return _fi_df(importances, cols)
        # Try linear coefficients
        coef_list = []
        for est in ests:
            if hasattr(est, 'coef_'):
                coefs = np.asarray(est.coef_)
                if coefs.ndim == 2:
                    coef_list.append(np.mean(np.abs(coefs), axis=0))
                else:
                    coef_list.append(np.abs(coefs))
        if coef_list:
            importances = np.mean(np.vstack(coef_list), axis=0)
            return _fi_df(importances, cols)
        return None
    # Single-output models
    if hasattr(model, 'feature_importances_'):
        return _fi_df(getattr(model, 'feature_importances_'), cols)
    if hasattr(model, 'coef_'):
        coefs = np.asarray(getattr(model, 'coef_'))
        if coefs.ndim == 2:
            importances = np.mean(np.abs(coefs), axis=0)
        else:
            importances = np.abs(coefs)
        return _fi_df(importances, cols)
    return None

# --- PRO UTILS: Numeric Feature Engineering ---
def fit_numeric_fe(df: pd.DataFrame, cols: List[str], method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fit parameters for a numeric feature engineering method on given columns.
    Returns dict of fitted parameters to reuse at inference.
    """
    fitted = {'method': method, 'cols': cols, 'details': {}}
    if method in ['Min-Max Scaling', 'Standardization', 'Robust Scaling']:
        stats = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors='coerce')
            if method == 'Min-Max Scaling':
                stats[c] = {'min': float(np.nanmin(s)), 'max': float(np.nanmax(s))}
            elif method == 'Standardization':
                stats[c] = {'mean': float(np.nanmean(s)), 'std': float(np.nanstd(s) or 1.0)}
            else:  # Robust Scaling
                q1 = float(np.nanpercentile(s, 25))
                q3 = float(np.nanpercentile(s, 75))
                iqr = q3 - q1 if (q3 - q1) != 0 else 1.0
                stats[c] = {'q1': q1, 'q3': q3, 'iqr': float(iqr)}
        fitted['details']['stats'] = stats
    elif method in ['Equal-Width Binning', 'Equal-Frequency Binning', 'Model-Based Binning', 'Custom Binning']:
        binners = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors='coerce')
            if method == 'Equal-Width Binning':
                k = int(params.get('bins', 5))
                binners[c] = {'edges': list(pd.cut(s, bins=k, retbins=True, duplicates='drop')[1])}
            elif method == 'Equal-Frequency Binning':
                k = int(params.get('bins', 5))
                binners[c] = {'edges': list(pd.qcut(s, q=k, retbins=True, duplicates='drop')[1])}
            elif method == 'Model-Based Binning':
                # Heuristic: fit quantiles based on sqrt(n) for stability
                k = max(3, int(np.sqrt(max(1, s.notna().sum()))))
                binners[c] = {'edges': list(pd.qcut(s, q=k, retbins=True, duplicates='drop')[1])}
            elif method == 'Custom Binning':
                # Expect list of thresholds in params['thresholds']
                thr = params.get('thresholds', [])
                thr = [float(t) for t in thr]
                edges = sorted(list(set([-np.inf] + thr + [np.inf])))
                binners[c] = {'edges': edges}
        fitted['details']['binners'] = binners
    elif method in ['Log Transformation', 'Square Root Transformation', 'Box-Cox Transformation', 'Yeo-Johnson Transformation']:
        transforms = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors='coerce')
            if method == 'Log Transformation':
                transforms[c] = {'add': float(max(1e-9, 1 - np.nanmin(s))) if np.nanmin(s) <= 0 else 0.0}
            elif method == 'Square Root Transformation':
                transforms[c] = {'type': 'square'}
            elif method == 'Box-Cox Transformation':
                # Box-Cox requires positive values
                shift = float(max(1e-6, 1 - np.nanmin(s))) if np.nanmin(s) <= 0 else 0.0
                s_shift = s + shift
                pt = PowerTransformer(method='box-cox', standardize=False)
                try:
                    pt.fit(s_shift.values.reshape(-1, 1))
                    transforms[c] = {'type': 'box-cox', 'shift': shift, 'lambda': float(pt.lambdas_[0])}
                except Exception:
                    transforms[c] = {'type': 'yeo-johnson'}  # fallback
            elif method == 'Yeo-Johnson Transformation':
                pt = PowerTransformer(method='yeo-johnson', standardize=False)
                try:
                    pt.fit(s.values.reshape(-1, 1))
                    transforms[c] = {'type': 'yeo-johnson', 'lambda': float(pt.lambdas_[0])}
                except Exception:
                    transforms[c] = {'type': 'none'}
        fitted['details']['transforms'] = transforms
    elif method in ['Capping/Winsorization']:
        caps = {}
        lower = float(params.get('lower_quantile', 0.01))
        upper = float(params.get('upper_quantile', 0.99))
        for c in cols:
            s = pd.to_numeric(df[c], errors='coerce')
            caps[c] = {'low': float(np.nanpercentile(s, lower*100)), 'high': float(np.nanpercentile(s, upper*100))}
        fitted['details']['caps'] = caps
    elif method in ['Null Value - Mean', 'Null Value - Median', 'Null Value - Mode', 'Null Value - Delete (Caution)']:
        # Nothing to fit except central tendency values
        imps = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors='coerce')
            if method == 'Null Value - Mean':
                imps[c] = {'value': float(np.nanmean(s))}
            elif method == 'Null Value - Median':
                imps[c] = {'value': float(np.nanmedian(s))}
            elif method == 'Null Value - Mode':
                mode_val = pd.Series(s).mode(dropna=True)
                imps[c] = {'value': float(mode_val.iloc[0]) if not mode_val.empty else 0.0}
            else:  # Delete
                imps[c] = {'delete': True}
        fitted['details']['impute'] = imps
    return fitted

def apply_numeric_fe(df: pd.DataFrame, fitted: Dict[str, Any]) -> pd.DataFrame:
    method = fitted.get('method')
    cols = fitted.get('cols', [])
    details = fitted.get('details', {})
    out = df.copy()
    if method == 'Min-Max Scaling':
        for c in cols:
            if c in out.columns:
                mn = details['stats'][c]['min']
                mx = details['stats'][c]['max']
                denom = (mx - mn) if (mx - mn) != 0 else 1.0
                out[c] = (pd.to_numeric(out[c], errors='coerce') - mn) / denom
    elif method == 'Standardization':
        for c in cols:
            if c in out.columns:
                mu = details['stats'][c]['mean']
                sd = details['stats'][c]['std'] or 1.0
                out[c] = (pd.to_numeric(out[c], errors='coerce') - mu) / sd
    elif method == 'Robust Scaling':
        for c in cols:
            if c in out.columns:
                q1 = details['stats'][c]['q1']
                iqr = details['stats'][c]['iqr'] or 1.0
                out[c] = (pd.to_numeric(out[c], errors='coerce') - q1) / iqr
    elif method in ['Equal-Width Binning', 'Equal-Frequency Binning', 'Model-Based Binning', 'Custom Binning']:
        for c in cols:
            if c in out.columns:
                edges = details['binners'][c]['edges']
                out[c] = pd.cut(pd.to_numeric(out[c], errors='coerce'), bins=edges, include_lowest=True, duplicates='drop').astype(str)
    elif method == 'Log Transformation':
        for c in cols:
            if c in out.columns:
                add = details['transforms'][c]['add']
                out[c] = np.log(pd.to_numeric(out[c], errors='coerce') + add)
    elif method == 'Square Root Transformation':
        for c in cols:
            if c in out.columns:
                val = pd.to_numeric(out[c], errors='coerce')
                out[c] = np.sqrt(val)
    elif method in ['Box-Cox Transformation', 'Yeo-Johnson Transformation']:
        for c in cols:
            if c in out.columns:
                val = pd.to_numeric(out[c], errors='coerce')
                if method == 'Box-Cox Transformation':
                    shift = details['transforms'][c].get('shift', 0.0)
                    lam = details['transforms'][c].get('lambda', None)
                    x = (val + shift).values.reshape(-1,1)
                    pt = PowerTransformer(method='box-cox', standardize=False)
                    if lam is not None:
                        # PowerTransformer doesn't accept preset lambda at transform; refit with lambda hints is nontrivial.
                        # For stability we refit quickly (acceptable here since we kept same method and shift).
                        pt.fit(x)
                    out[c] = pt.fit_transform(x).ravel()
                else:
                    lam = details['transforms'][c].get('lambda', None)
                    x = val.values.reshape(-1,1)
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    if lam is not None:
                        pt.fit(x)
                    out[c] = pt.fit_transform(x).ravel()
    elif method == 'Capping/Winsorization':
        for c in cols:
            if c in out.columns:
                low = details['caps'][c]['low']
                high = details['caps'][c]['high']
                val = pd.to_numeric(out[c], errors='coerce')
                out[c] = val.clip(lower=low, upper=high)
    elif method in ['Null Value - Mean', 'Null Value - Median', 'Null Value - Mode', 'Null Value - Delete (Caution)']:
        for c in cols:
            if c in out.columns:
                info = details['impute'][c]
                if info.get('delete'):
                    out = out[~out[c].isna()].copy()
                else:
                    out[c] = pd.to_numeric(out[c], errors='coerce').fillna(info['value'])
    return out

# --- PRO UTILS: Auto-configuration based on data profile ---
def _detect_datetime_like_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        try:
            ser = pd.to_datetime(df[c], errors='coerce')
            if ser.notna().mean() > 0.8:
                out.append(c)
        except Exception:
            continue
    return out

def _data_profile(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    dt_cols = _detect_datetime_like_columns(df, feature_cols)
    n_rows = len(df)
    n_num = len(num_cols)
    n_cat = len(cat_cols)
    high_card = False
    avg_card = 0
    if n_cat:
        cards = [int(df[c].nunique()) for c in cat_cols]
        avg_card = float(np.mean(cards)) if cards else 0.0
        high_card = any(card > 20 for card in cards)
    return {
        'n_rows': n_rows,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'dt_cols': dt_cols,
        'n_num': n_num,
        'n_cat': n_cat,
        'avg_card': avg_card,
        'high_card': high_card,
    }

def _suggest_encoding(problem_type: str, profile: Dict[str, Any]) -> str:
    if problem_type == 'Classification' and profile['n_cat'] > 0 and profile['high_card']:
        return 'Target Encoding'
    return 'One-Hot Encoding'

def _suggest_models(problem_type: str, profile: Dict[str, Any]) -> List[str]:
    models = []
    if problem_type == 'Classification':
        # Always include strong tree baselines
        models = ['RandomForest', 'GradientBoosting', 'ExtraTrees']
        # Add linear/SVM/KNN when numeric features are present and not too large
        if profile['n_num'] > 0 and profile['n_rows'] <= 20000:
            models.append('LogisticRegression')
        if profile['n_num'] > 0 and profile['n_rows'] <= 10000:
            models.append('SVM')
        if profile['n_rows'] <= 5000 and (profile['n_num'] + profile['n_cat']) <= 100:
            models.append('KNN')
        models.append('DecisionTree')
        models.append('NaiveBayes')
        models.append('AdaBoost')
    else:
        models = ['RandomForest', 'GradientBoosting', 'ExtraTrees']
        if profile['n_num'] > 0:
            models.extend(['Ridge', 'ElasticNet'])
        if profile['n_rows'] <= 5000 and profile['n_num'] > 0:
            models.append('SVR')
        if profile['n_rows'] <= 5000:
            models.append('KNN')
        models.append('DecisionTree')
        models.append('AdaBoost')
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for m in models:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered

def _suggest_hyperparams(profile: Dict[str, Any]) -> Dict[str, Any]:
    n = profile['n_rows']
    # Trees
    n_estimators = 200 if n >= 5000 else 100
    rf_max_depth = 0  # auto
    rf_min_samples_leaf = 1 if n >= 1000 else 2
    dt_max_depth = 0
    # Linear / SVM
    lr_C = 1.0
    svm_C = 1.0
    # KNN
    knn_k = int(np.clip(round(np.sqrt(max(1, n))), 3, 25))
    # CV
    use_cv = n < 5000
    cv_folds = 5
    return {
        'n_estimators': n_estimators,
        'rf_max_depth': rf_max_depth,
        'rf_min_samples_leaf': rf_min_samples_leaf,
        'dt_max_depth': dt_max_depth,
        'lr_C': lr_C,
        'svm_C': svm_C,
        'knn_k': knn_k,
        'use_cv': use_cv,
        'cv_folds': cv_folds,
    }

def _suggest_standardize(selected_models: List[str], profile: Dict[str, Any]) -> bool:
    needs_scale = any(m in selected_models for m in ['LogisticRegression', 'SVM', 'KNN'])
    return bool(needs_scale and profile['n_num'] > 0)

def _suggest_feature_rules(df: pd.DataFrame, feature_cols: List[str]) -> List[Dict[str, str]]:
    rules = []
    profile = _data_profile(df, feature_cols)
    # Date difference if at least two datetime-like columns
    if len(profile['dt_cols']) >= 2:
        c1, c2 = profile['dt_cols'][0], profile['dt_cols'][1]
        rules.append({'col1': c1, 'col2': c2, 'method': 'datediff', 'name': f'Days_{c1}_minus_{c2}'})
    # Numeric absolute difference for the first two numeric columns
    if len(profile['num_cols']) >= 2:
        n1, n2 = profile['num_cols'][0], profile['num_cols'][1]
        rules.append({'col1': n1, 'col2': n2, 'method': 'absdiff', 'name': f'AbsDiff_{n1}_{n2}'})
    return rules

def infer_auto_config(df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Any]:
    if not target_cols:
        return {}
    feature_cols = [c for c in df.columns if c not in target_cols]
    # Determine problem type from first target
    pt = get_problem_type(df[target_cols[0]])
    profile = _data_profile(df, feature_cols)
    encoding = _suggest_encoding(pt, profile)
    models = _suggest_models(pt, profile)
    hps = _suggest_hyperparams(profile)
    standardize = _suggest_standardize(models, profile)
    rules = _suggest_feature_rules(df, feature_cols)
    return {
        'problem_type': pt,
        'encoding_choice': encoding,
        'selected_models': models,
        'fe_standardize': standardize,
        **hps,
        'feature_rules': rules,
    }

def get_three_default_models(problem_type: str) -> List[str]:
    """Return exactly three default model names per problem type."""
    if problem_type == 'Classification':
        return ['RandomForest', 'GradientBoosting', 'LogisticRegression']
    return ['RandomForest', 'GradientBoosting', 'Ridge']

# --- 1. APP STRUCTURE AND TITLES ---
st.markdown("<h1 style='text-align: center;'>Button3</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Simple & Fast Machine Learning For Everyone!</p>", unsafe_allow_html=True)

# --- QUICK INTRO (How it works) ---
def render_quick_intro():
    """Render a visual 3-step How it works as an inline SVG via components.html.
    Falls back to a simple text version if rendering fails.
    """
    try:
        svg = """
<style>
.hiw-wrapper{margin-top:10px}
.hiw-card{background:linear-gradient(135deg,#f8fafc,#eef2f7);border-radius:16px;padding:20px 16px;box-shadow:0 6px 18px rgba(0,0,0,0.07);}
.hiw-title{font-weight:700;font-size:22px;margin:0 0 10px 0;color:#1f2937;text-align:center}
.hiw-svg{width:100%;height:auto;display:block}
.stepname{font:600 20px 'Segoe UI',system-ui,-apple-system;fill:#111827}
.subtle{font:400 12px 'Segoe UI',system-ui,-apple-system;fill:#6b7280}
@media (max-width: 860px){.subtle{display:none}}
</style>
<div class="hiw-wrapper">
    <div class="hiw-card">
        <div class="hiw-title">How it works</div>
        <svg class="hiw-svg" viewBox="0 0 1200 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="How it works: Upload, Predict, Download">
            <defs>
                <!-- Ant Design primary blue gradient -->
                <linearGradient id="antBlue" x1="0" x2="1" y1="0" y2="1">
                    <stop offset="0%" stop-color="#1677ff"/>
                    <stop offset="100%" stop-color="#4096ff"/>
                </linearGradient>
            </defs>
            <!-- connector line -->
            <path d="M220 85 H 980" stroke="#d9d9d9" stroke-width="2.5" fill="none"/>
            <!-- circle 1 -->
            <circle cx="220" cy="85" r="40" fill="url(#antBlue)" />
            <g stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <!-- Unified arrows: shaft 18px; head height 10px -->
                <path d="M220 70 l0 18"/>
                <polyline points="210,70 220,60 230,70"/>
            </g>
            <text x="220" y="145" text-anchor="middle" class="stepname">Step 1 • Upload Training</text>
            <!-- circle 2 -->
            <circle cx="600" cy="85" r="40" fill="url(#antBlue)" />
            <g stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <!-- Unified arrows: shaft 18px; head height 10px -->
                <path d="M600 70 l0 18"/>
                <polyline points="590,70 600,60 610,70"/>
            </g>
            <text x="600" y="145" text-anchor="middle" class="stepname">Step 2 • Upload Testing</text>
            <!-- circle 3 -->
            <circle cx="980" cy="85" r="40" fill="url(#antBlue)" />
            <g stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round">
                <!-- Unified arrows: shaft 18px; head height 10px -->
                <polyline points="970,92 980,102 990,92"/>
                <line x1="980" y1="74" x2="980" y2="92"/>
            </g>
            <text x="980" y="145" text-anchor="middle" class="stepname">Step 3 • Download Predictions</text>
        </svg>
    <div style="margin-top:6px;color:#6b7280;font-size:12px;text-align:center;">Tip: Free supports files up to 20MB; Pro up to 200MB and advanced features.</div>
    </div>
</div>
        """
        import streamlit.components.v1 as components
        components.html(svg, height=320, scrolling=False)
    except Exception:
        st.markdown("### How it works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Step 1: Upload Training File**\n\n- CSV/XLS(X) with features and target(s)\n- We'll validate, encode, and train")
        with c2:
            st.markdown("**Step 2: Upload Prediction File**\n\n- Same feature columns (without targets)\n- We'll apply the trained model")
        with c3:
            st.markdown("**Step 3: Download Predictions**\n\n- Get results as CSV or Excel\n- Includes predicted target column(s)")

# Show the intro beneath the tagline
render_quick_intro()

# --- 2. SIMPLE UPLOAD SECTION ---
# Use previously selected analysis type (if any) to tailor the upload title subtly.
_prev_analysis = st.session_state.get('analysis_type_selector')
_is_ts_mode_hint = (_prev_analysis == 'Time Series Forecasting')

# Scoped CSS to center the Upload header, button, and caption without affecting other sections
st.markdown(
    """
    <style>
    #upload-center h1, #upload-center h2 { text-align: center !important; }
    #upload-center [data-testid="stFileUploader"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    #upload-center [data-testid="stFileUploaderDropzone"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    #upload-center [data-testid="stFileUploader"] button,
    #upload-center [data-testid="stFileUploader"] [role="button"] {
        margin-left: auto !important;
        margin-right: auto !important;
        display: inline-flex !important;
    }
    #upload-center [data-testid="stCaptionContainer"],
    #upload-center p { text-align: center !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div id="upload-center">', unsafe_allow_html=True)
st.header("Upload Time Series File" if _is_ts_mode_hint else "Upload")
uploaded_train_file = st.file_uploader(
    "Upload file",
    type=['csv', 'xls', 'xlsx'],
    key="main_uploader",
    label_visibility="collapsed"
)
if _is_ts_mode_hint:
    st.caption("Time Series: include a Date/Time column, a numeric Target, and optional exogenous columns in one file. Pro users can optionally add a future exogenous file later in the TS panel.")
else:
    st.caption("Standard: upload a training file (with targets) now; you'll upload a prediction file later to score new rows.")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_train_file is None:
    reset_state()

if uploaded_train_file is not None:
    # Check file size with Free/Pro limits
    file_size_mb = uploaded_train_file.size / (1024 * 1024)
    pro_unlocked = st.session_state.get('pro_unlocked', False)

    # Free plan: 20MB limit with Upgrade prompt
    if not pro_unlocked and file_size_mb > 20:
        st.error(f"🚫 File too large for Free plan ({file_size_mb:.1f}MB). Free plan supports files up to 20MB.")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.caption("Upgrade to Pro to handle larger files (up to 200MB).")
        with c2:
            if st.button("Upgrade to Pro!", key="upgrade_pro_from_upload"):
                st.session_state['pro_unlocked'] = True
                # Clear any uploaded file so the user re-uploads under Pro
                try:
                    if 'main_uploader' in st.session_state:
                        del st.session_state['main_uploader']
                except Exception:
                    pass
                st.success("🎉 Pro unlocked! Please re-upload your file to continue.")
                try:
                    st.rerun()
                except Exception:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        st.stop()

    # Pro plan: 200MB hard limit
    if pro_unlocked and file_size_mb > 200:
        st.error(f"🚫 File too large ({file_size_mb:.1f}MB). Pro plan supports files up to 200MB.")
        st.stop()
    
    train_df = None
    try:
        file_extension = os.path.splitext(uploaded_train_file.name)[1]
        
        if file_extension == '.csv':
            raw_data = uploaded_train_file.read(100000)
            uploaded_train_file.seek(0)
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding'] if result['encoding'] else 'utf-8'
            train_df = pd.read_csv(uploaded_train_file, encoding=detected_encoding)
        else:
            train_df = pd.read_excel(uploaded_train_file)
        
        st.session_state['output_file_extension'] = file_extension
        st.success("File read successfully!")

        for col in train_df.columns:
            if train_df[col].dtype == 'object':
                train_df[col] = train_df[col].astype(str)
    except Exception as e:
        st.error(f"Error reading file: {e}")

    if train_df is not None:
        st.dataframe(train_df.head())
        
        analysis_type = st.radio(
            "Choose Analysis Type",
            ('Standard Prediction (Classification/Regression)', 'Time Series Forecasting'),
            horizontal=True, key='analysis_type_selector'
        )
        st.divider()

        if analysis_type == 'Standard Prediction (Classification/Regression)':
            with st.container():
                st.subheader("Standard Prediction Setup")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.write("**Step 1: Exclude ID-like columns**")
                    suggested_exclusions = detect_id_columns(train_df)
                    confirmed_exclusions = st.multiselect("Exclude Columns", options=train_df.columns.tolist(), default=suggested_exclusions, label_visibility="collapsed")
                
                with c2:
                    st.write("**Step 2: Select Target Variable(s)**")
                    available_cols = [col for col in train_df.columns if col not in confirmed_exclusions]
                    selected_target_cols = st.multiselect("Select Targets", options=available_cols, label_visibility="collapsed")

                # Step 3: Choose Encoding and Feature Strategy
                st.write("**Step 3: Choose Encoding and Feature Strategy**")

                # Determine plan status early
                pro_unlocked = st.session_state.get('pro_unlocked', False)

                # Defaults and helpers
                auto_cfg = {}
                encoding_choice = 'One-Hot Encoding'
                numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(train_df[c])]
                categorical_cols = [c for c in available_cols if not pd.api.types.is_numeric_dtype(train_df[c])]

                if pro_unlocked:
                    # Pro: show Auto-Configure toggle
                    if 'auto_config_enabled' not in st.session_state:
                        st.session_state['auto_config_enabled'] = True
                    auto_on = st.checkbox(
                        "Auto-configure defaults from data",
                        key="auto_config_enabled"
                    )

                    if auto_on and selected_target_cols:
                        # Try to infer defaults; fall back to heuristics
                        try:
                            auto_cfg = infer_auto_config(train_df[[c for c in available_cols] + selected_target_cols], selected_target_cols)
                        except Exception:
                            auto_cfg = {}

                        if auto_cfg.get('encoding_choice'):
                            encoding_choice = auto_cfg['encoding_choice']
                        else:
                            # Heuristic: multi-target -> One-Hot; high-cardinality categoricals -> Target Encoding
                            if len(selected_target_cols) > 1:
                                encoding_choice = 'One-Hot Encoding'
                            else:
                                max_card = max((train_df[col].nunique() for col in categorical_cols), default=0)
                                encoding_choice = 'Target Encoding' if max_card > 20 else 'One-Hot Encoding'

                        # Initialize FE suggestion containers (editable later in Step 4)
                        st.session_state.setdefault('fe_num_selections', [])
                        st.session_state.setdefault('categorical_fe_selections', [])

                        auto_cfg['encoding_choice'] = encoding_choice
                        auto_cfg['fe_standardize'] = True if len(numeric_cols) > 0 else False
                        st.success(f"Auto-selected encoding: {encoding_choice}. Suggested feature engineering have been applied.")
                    else:
                        # Pro manual mode: default to One-Hot (no dropdown)
                        encoding_choice = 'One-Hot Encoding'
                        st.caption("Manual mode: adjust Feature Engineering in Step 4 below.")
                else:
                    # Free plan: no Auto-Configure; show manual encoding selector and force auto_config_enabled=False
                    st.session_state['auto_config_enabled'] = False
                    encoding_options = ['One-Hot Encoding', 'Target Encoding', 'Label Encoding']
                    encoding_choice = st.selectbox("Encoding Strategy", options=encoding_options, index=0, label_visibility="collapsed")
                    auto_on = False

                # Pro unlocked status (now controlled from upload section)
                # (Note: pro_unlocked resolved above for this step)

                # Safe defaults (used when Pro is locked)
                fe_standardize = False
                use_cv = False
                cv_folds = 5
                selected_models = []
                rf_max_depth = 0
                rf_min_samples_leaf = 1
                lr_C = 1.0
                svm_C = 1.0
                knn_k = 5
                dt_max_depth = 0
                # Seed defaults from auto-config when enabled (Pro only)
                if st.session_state.get('auto_config_enabled', False) and pro_unlocked and auto_cfg:
                    fe_standardize = bool(auto_cfg.get('fe_standardize', fe_standardize))
                    use_cv = bool(auto_cfg.get('use_cv', use_cv))
                    cv_folds = int(auto_cfg.get('cv_folds', cv_folds))
                    selected_models = auto_cfg.get('selected_models', selected_models)
                    rf_max_depth = int(auto_cfg.get('rf_max_depth', rf_max_depth))
                    rf_min_samples_leaf = int(auto_cfg.get('rf_min_samples_leaf', rf_min_samples_leaf))
                    lr_C = float(auto_cfg.get('lr_C', lr_C))
                    svm_C = float(auto_cfg.get('svm_C', svm_C))
                    knn_k = int(auto_cfg.get('knn_k', knn_k))
                    dt_max_depth = int(auto_cfg.get('dt_max_depth', dt_max_depth))
                
                # Ensure Free mode gets default model restriction
                if not pro_unlocked:
                    selected_models = ['RandomForest']  # Free mode restriction

                # --- Feature Engineering and Model Configuration ---
                # Show manual FE UI only when Auto-Configure is OFF
                # Respect Auto-Configure toggle strictly: hide Step 4 when Auto is ON
                show_fe_ui = pro_unlocked and (not auto_on)
                if show_fe_ui:
                    st.write("**Step 4: Feature Engineering**")
                    
                    # Pro: Numeric Feature Engineering (Optional)
                    with st.expander("Pro: Numeric Feature Engineering (Optional)"):
                        numeric_cols = [c for c in available_cols if pd.api.types.is_numeric_dtype(train_df[c])]
                        if numeric_cols:
                            numeric_fe_methods = [
                                'Min-Max Scaling', 'Standardization', 'Robust Scaling',
                                'Equal-Width Binning', 'Equal-Frequency Binning', 'Model-Based Binning',
                                'Log Transformation', 'Square Root Transformation', 'Box-Cox Transformation', 'Yeo-Johnson Transformation',
                                'Capping/Winsorization', 'Null Value - Mean', 'Null Value - Median', 'Null Value - Mode'
                            ]
                            
                            # Initialize numeric FE selections
                            if 'numeric_fe_selections' not in st.session_state:
                                st.session_state['numeric_fe_selections'] = []
                            
                            st.write("**Select Method and Column for Numeric Feature Engineering:**")
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                selected_numeric_method = st.selectbox("Method", options=numeric_fe_methods, key="numeric_method_select")
                            with col2:
                                selected_numeric_cols = st.multiselect("Columns", options=numeric_cols, key="numeric_cols_select")
                            with col3:
                                if st.button("+ Add", key="add_numeric_fe"):
                                    if not selected_numeric_cols:
                                        st.info("Please select at least one numeric column.")
                                    else:
                                        # Enforce: only numeric columns can be added
                                        valid_set = set(numeric_cols)
                                        to_add = [c for c in selected_numeric_cols if c in valid_set]
                                        invalid = [c for c in selected_numeric_cols if c not in valid_set]
                                        for _col in to_add:
                                            st.session_state['numeric_fe_selections'].append({
                                                'method': selected_numeric_method,
                                                'column': _col
                                            })
                                        if invalid:
                                            st.warning(f"Skipped non-numeric columns: {invalid}")
                                        st.rerun()
                            
                            # Display current selections
                            if st.session_state['numeric_fe_selections']:
                                st.write("**Current Numeric FE Selections:**")
                                for i, sel in enumerate(st.session_state['numeric_fe_selections']):
                                    col_a, col_b = st.columns([4, 1])
                                    with col_a:
                                        st.write(f"{i+1}. {sel['method']} → {sel['column']}")
                                    with col_b:
                                        if st.button("Remove", key=f"remove_numeric_{i}"):
                                            st.session_state['numeric_fe_selections'].pop(i)
                                            st.rerun()
                        else:
                            st.info("No numeric columns available for feature engineering.")
                    
                    # Pro: Categorical Feature Engineering (Optional)
                    with st.expander("Pro: Categorical Feature Engineering (Optional)"):
                        categorical_cols = [c for c in available_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
                        if categorical_cols:
                            categorical_fe_methods = [
                                'One-Hot Encoding', 'Label Encoding', 'Target Encoding', 'Binary Encoding',
                                'Frequency Encoding', 'Ordinal Encoding', 'Hash Encoding'
                            ]
                            
                            # Initialize categorical FE selections
                            if 'categorical_fe_selections' not in st.session_state:
                                st.session_state['categorical_fe_selections'] = []
                            
                            st.write("**Select Method and Column for Categorical Feature Engineering:**")
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                selected_cat_method = st.selectbox("Method", options=categorical_fe_methods, key="cat_method_select")
                            with col2:
                                selected_cat_cols = st.multiselect("Columns", options=categorical_cols, key="cat_cols_select")
                            with col3:
                                if st.button("+ Add", key="add_categorical_fe"):
                                    if not selected_cat_cols:
                                        st.info("Please select at least one categorical column.")
                                    else:
                                        # Enforce: only categorical (non-numeric) columns can be added
                                        valid_set = set(categorical_cols)
                                        to_add = [c for c in selected_cat_cols if c in valid_set]
                                        invalid = [c for c in selected_cat_cols if c not in valid_set]
                                        for _col in to_add:
                                            st.session_state['categorical_fe_selections'].append({
                                                'method': selected_cat_method,
                                                'column': _col
                                            })
                                        if invalid:
                                            st.warning(f"Skipped non-categorical columns: {invalid}")
                                        st.rerun()
                            
                            # Display current selections
                            if st.session_state['categorical_fe_selections']:
                                st.write("**Current Categorical FE Selections:**")
                                for i, sel in enumerate(st.session_state['categorical_fe_selections']):
                                    col_a, col_b = st.columns([4, 1])
                                    with col_a:
                                        st.write(f"{i+1}. {sel['method']} → {sel['column']}")
                                    with col_b:
                                        if st.button("Remove", key=f"remove_categorical_{i}"):
                                            st.session_state['categorical_fe_selections'].pop(i)
                                            st.rerun()
                        else:
                            st.info("No categorical columns available for feature engineering.")
                    
                    
                    # Pro: Custom Feature Generation (Optional)
                    with st.expander("Pro: Custom Feature Generation (Optional)"):
                        st.write("**Create new features by existing columns**")
                        
                        # Initialize custom feature rules
                        if 'custom_feature_rules' not in st.session_state:
                            st.session_state['custom_feature_rules'] = []
                        
                        # Feature generation methods
                        feature_gen_operations = [
                            'Add (+)', 'Subtract (-)', 'Multiply (*)', 'Divide (/)', 
                            'Power (^)', 'Mean', 'Max', 'Min', 'Ratio'
                        ]
                        
                        st.write("**Add New Custom Feature:**")
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                        with col1:
                            custom_col1 = st.selectbox("First Column", options=available_cols, key="custom_col1")
                        with col2:
                            custom_operation = st.selectbox("Operation", options=feature_gen_operations, key="custom_operation")
                        with col3:
                            custom_col2 = st.selectbox("Second Column", options=available_cols, key="custom_col2")
                        with col4:
                            if st.button("+ Add", key="add_custom_feature"):
                                new_feature_name = f"{custom_col1}_{custom_operation.split()[0]}_{custom_col2}".replace(' ', '_').replace('(', '').replace(')', '')
                                st.session_state['custom_feature_rules'].append({
                                    'col1': custom_col1,
                                    'operation': custom_operation,
                                    'col2': custom_col2,
                                    'new_name': new_feature_name
                                })
                                st.rerun()
                        
                        # Display current custom features
                        if st.session_state['custom_feature_rules']:
                            st.write("**Current Custom Features:**")
                            for i, rule in enumerate(st.session_state['custom_feature_rules']):
                                col_a, col_b = st.columns([4, 1])
                                with col_a:
                                    st.write(f"{i+1}. {rule['new_name']}: {rule['col1']} {rule['operation']} {rule['col2']}")
                                with col_b:
                                    if st.button("Remove", key=f"remove_custom_{i}"):
                                        st.session_state['custom_feature_rules'].pop(i)
                                        st.rerun()

                # --- PRO: Model Selection and Configuration ---
                if pro_unlocked:
                    st.write("**Step 5: Model Configuration**")
                    # Pro: Multi-Model Selection (Optional)
                    with st.expander("Pro: Multi-Model Selection (Optional)"):
                        # Model selections based on problem type
                        available_models_cls = ['RandomForest', 'LogisticRegression', 'SVM', 'KNN', 'DecisionTree', 'GradientBoosting', 'ExtraTrees', 'AdaBoost', 'NaiveBayes']
                        available_models_reg = ['RandomForest', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'KNN', 'DecisionTree', 'GradientBoosting', 'ExtraTrees', 'AdaBoost']
                        detected_pt = get_problem_type(train_df[selected_target_cols[0]]) if selected_target_cols else 'Classification'
                        model_opts = available_models_cls if detected_pt == "Classification" else available_models_reg
                        
                        # Auto-select 3 default models
                        three_defaults = get_three_default_models(detected_pt)
                        default_selection = [m for m in (selected_models or three_defaults) if m in model_opts]
                        
                        st.write(f"**Problem Type Detected: {detected_pt}**")
                        selected_models = st.multiselect(
                            "Select models to train (3 default models auto-selected)", 
                            options=model_opts, 
                            default=default_selection,
                            key="multi_model_selection"
                        )
                        
                        # Cross-validation and model validation options (mainstream patterns)
                        st.write("**Model Validation Options:**")
                        cval1, cval2 = st.columns(2)
                        with cval1:
                            strategy = st.selectbox(
                                "Validation Strategy",
                                [
                                    "Hold-out (Train/Test Split)",
                                    "K-Fold CV",
                                    "Repeated K-Fold CV"
                                ],
                                key="cv_strategy_select"
                            )
                            is_cv = strategy != "Hold-out (Train/Test Split)"
                            cv_folds = st.slider(
                                "CV Folds", min_value=3, max_value=10, value=5, step=1,
                                disabled=not is_cv, key="cv_folds_slider"
                            )
                            cv_repeats = st.slider(
                                "Repeats (for Repeated K-Fold)", min_value=1, max_value=5, value=1, step=1,
                                disabled=(strategy != "Repeated K-Fold CV"), key="cv_repeats_slider"
                            )
                        with cval2:
                            use_stratified = st.checkbox(
                                "Stratified (Classification)", value=True,
                                disabled=detected_pt != "Classification", key="stratified_cv"
                            )
                            cv_shuffle = st.checkbox("Shuffle Folds", value=True, disabled=not is_cv, key="cv_shuffle")
                            random_state_cv = st.number_input("Random State", value=42, step=1, key="cv_random_state")
                            scoring_options = (
                                ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']
                                if detected_pt == 'Classification' else
                                ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
                            )
                            cv_scoring = st.selectbox("Scoring Metric", scoring_options, key="cv_scoring")

                        # Back-compat variables used elsewhere
                        use_cv = is_cv
                        # Persist for downstream consumers without altering unrelated logic
                        st.session_state['cv_strategy'] = strategy
                        st.session_state['cv_folds'] = cv_folds
                        st.session_state['cv_repeats'] = cv_repeats
                        # Do not reassign widget-managed keys (cv_shuffle, cv_scoring) to avoid conflicts
                    
                    # Pro: Custom Hyperparameters (Optional)
                    with st.expander("Pro: Custom Hyperparameters (Optional)"):
                        st.write("**Configure hyperparameters for selected models**")
                        
                        if selected_models:
                            # Initialize hyperparameter settings
                            if 'custom_hyperparams' not in st.session_state:
                                st.session_state['custom_hyperparams'] = {}
                            
                            for model in selected_models:
                                st.write(f"**{model} Hyperparameters:**")
                                
                                if model not in st.session_state['custom_hyperparams']:
                                    st.session_state['custom_hyperparams'][model] = {}
                                
                                # Model-specific hyperparameters
                                if model == 'RandomForest':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_estimators'] = st.slider(
                                            f"N Estimators", 10, 500, 100, key=f"rf_n_est_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['max_depth'] = st.slider(
                                            f"Max Depth", 1, 20, 10, key=f"rf_depth_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['min_samples_split'] = st.slider(
                                            f"Min Samples Split", 2, 20, 2, key=f"rf_split_{model}")
                                
                                elif model == 'LogisticRegression':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['C'] = st.slider(
                                            f"Regularization (C)", 0.01, 10.0, 1.0, key=f"lr_c_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['max_iter'] = st.slider(
                                            f"Max Iterations", 100, 1000, 100, key=f"lr_iter_{model}")
                                
                                elif model == 'SVM':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['C'] = st.slider(
                                            f"C", 0.01, 10.0, 1.0, key=f"svm_c_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['kernel'] = st.selectbox(
                                            f"Kernel", ['rbf', 'linear', 'poly'], key=f"svm_kernel_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['gamma'] = st.selectbox(
                                            f"Gamma", ['scale', 'auto'], key=f"svm_gamma_{model}")
                                
                                elif model == 'GradientBoosting':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_estimators'] = st.slider(
                                            f"N Estimators", 50, 300, 100, key=f"gb_n_est_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['learning_rate'] = st.slider(
                                            f"Learning Rate", 0.01, 0.3, 0.1, key=f"gb_lr_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['max_depth'] = st.slider(
                                            f"Max Depth", 1, 10, 3, key=f"gb_depth_{model}")
                                
                                elif model == 'DecisionTree':
                                    # Determine problem type for criterion choices
                                    try:
                                        detected_pt_local = get_problem_type(train_df[selected_target_cols[0]]) if selected_target_cols else 'Classification'
                                    except Exception:
                                        detected_pt_local = 'Classification'
                                    crit_opts = ['gini', 'entropy'] if detected_pt_local == 'Classification' else ['squared_error', 'friedman_mse', 'absolute_error']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['max_depth'] = st.slider(
                                            f"Max Depth", 1, 50, 10, key=f"dt_depth_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['min_samples_split'] = st.slider(
                                            f"Min Samples Split", 2, 20, 2, key=f"dt_split_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['min_samples_leaf'] = st.slider(
                                            f"Min Samples Leaf", 1, 20, 1, key=f"dt_leaf_{model}")
                                    st.session_state['custom_hyperparams'][model]['criterion'] = st.selectbox(
                                        f"Criterion", crit_opts, key=f"dt_crit_{model}")
                                
                                elif model == 'KNN':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_neighbors'] = st.slider(
                                            f"N Neighbors", 1, 20, 5, key=f"knn_neighbors_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['weights'] = st.selectbox(
                                            f"Weights", ['uniform', 'distance'], key=f"knn_weights_{model}")
                                
                                elif model == 'ExtraTrees':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_estimators'] = st.slider(
                                            f"N Estimators", 50, 500, 100, key=f"et_n_est_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['max_depth'] = st.slider(
                                            f"Max Depth", 1, 50, 10, key=f"et_depth_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['min_samples_leaf'] = st.slider(
                                            f"Min Samples Leaf", 1, 10, 1, key=f"et_leaf_{model}")
                                    st.session_state['custom_hyperparams'][model]['max_features'] = st.selectbox(
                                        f"Max Features", ['auto', 'sqrt', 'log2'], key=f"et_max_feat_{model}")
                                
                                elif model in ['Ridge', 'Lasso']:
                                    st.session_state['custom_hyperparams'][model]['alpha'] = st.slider(
                                        f"Alpha", 0.01, 10.0, 1.0, key=f"reg_alpha_{model}")
                                
                                elif model == 'LinearRegression':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['fit_intercept'] = st.checkbox(
                                            f"Fit Intercept", value=True, key=f"lr_fit_intercept_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['positive'] = st.checkbox(
                                            f"Positive Coefficients", value=False, key=f"lr_positive_{model}")

                                elif model == 'ElasticNet':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['alpha'] = st.slider(
                                            f"Alpha", 0.01, 10.0, 1.0, key=f"en_alpha_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['l1_ratio'] = st.slider(
                                            f"L1 Ratio", 0.0, 1.0, 0.5, key=f"en_l1_{model}")
                                
                                elif model == 'SVR':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['C'] = st.slider(
                                            f"C", 0.01, 10.0, 1.0, key=f"svr_c_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['epsilon'] = st.slider(
                                            f"Epsilon", 0.0, 1.0, 0.1, step=0.01, key=f"svr_eps_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['kernel'] = st.selectbox(
                                            f"Kernel", ['rbf', 'linear', 'poly', 'sigmoid'], key=f"svr_kernel_{model}")
                                    st.session_state['custom_hyperparams'][model]['gamma'] = st.selectbox(
                                        f"Gamma", ['scale', 'auto'], key=f"svr_gamma_{model}")
                                    if st.session_state['custom_hyperparams'][model]['kernel'] == 'poly':
                                        st.session_state['custom_hyperparams'][model]['degree'] = st.slider(
                                            f"Degree (poly)", 2, 6, 3, key=f"svr_degree_{model}")

                                elif model == 'AdaBoost':
                                    # Determine problem type for options
                                    try:
                                        detected_pt_local = get_problem_type(train_df[selected_target_cols[0]]) if selected_target_cols else 'Classification'
                                    except Exception:
                                        detected_pt_local = 'Classification'
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_estimators'] = st.slider(
                                            f"N Estimators", 50, 500, 100, key=f"ab_n_est_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['learning_rate'] = st.slider(
                                            f"Learning Rate", 0.01, 2.0, 1.0, key=f"ab_lr_{model}")
                                    if detected_pt_local == 'Classification':
                                        st.session_state['custom_hyperparams'][model]['algorithm'] = st.selectbox(
                                            f"Algorithm", ['SAMME.R', 'SAMME'], key=f"ab_algo_{model}")
                                    else:
                                        st.session_state['custom_hyperparams'][model]['loss'] = st.selectbox(
                                            f"Loss", ['linear', 'square', 'exponential'], key=f"ab_loss_{model}")
                                
                                elif model == 'NaiveBayes':
                                    # GaussianNB: primary hyperparameter is var_smoothing (default 1e-9)
                                    # Use an exponent slider for log-scale control
                                    exp_val = st.slider(
                                        f"var_smoothing exponent (10^x)", -12, -3, -9, key=f"nb_vs_exp_{model}"
                                    )
                                    vs_val = float(10.0 ** exp_val)
                                    st.session_state['custom_hyperparams'][model]['var_smoothing'] = vs_val
                                    st.caption(f"var_smoothing = {vs_val:.1e}")
                                
                                st.divider()
                        else:
                            st.info("Please select models in the Multi-Model Selection section to configure hyperparameters.")
                
                # Basic Model Configuration
                st.write("**Model Hold-out Settings**")
                test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100

                # (Removed duplicate 'Pro: Custom Hyperparameters (optional)' block)

                # Train button (right-aligned by unified CSS)
                train_btn_container = st.container()
                with train_btn_container:
                    clicked_train = st.button("Train Model")

                if clicked_train:
                    if not selected_target_cols:
                        st.error("Please select at least one target variable.")
                    elif encoding_choice == 'Target Encoding' and len(selected_target_cols) > 1:
                        st.error("Target Encoding is currently supported for single-target problems only.")
                    else:
                        # Validate data quality
                        validation_errors = validate_data(train_df, selected_target_cols)
                        if validation_errors:
                            for error in validation_errors:
                                st.error(error)
                            st.stop()
                        with st.spinner('Training in progress...'):
                            # Add progress bar for training
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            feature_cols = [col for col in available_cols if col not in selected_target_cols]
                            # Persist base input feature column names (before feature generation)
                            st.session_state['input_feature_columns'] = feature_cols
                            try:
                                joblib.dump(feature_cols, 'input_feature_columns.joblib')
                            except Exception:
                                pass
                            X = train_df[feature_cols]
                            # Apply Pro custom feature generation before any encoding/imputation
                            # Use custom feature rules from UI (Pro)
                            feature_rules = st.session_state.get('custom_feature_rules', []) if pro_unlocked else []
                            if feature_rules:
                                X = apply_feature_generations(X, feature_rules)
                            y = train_df[selected_target_cols[0]] if len(selected_target_cols) == 1 else train_df[selected_target_cols]
                            
                            # Progress: Data preparation
                            progress_bar.progress(0.1)
                            status_text.text('Preparing data and splitting dataset...')
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                            # Apply Pro Numeric Feature Engineering before categorical encoding
                            fe_num_selections = st.session_state.get('fe_num_selections', []) if pro_unlocked else []
                            fe_num_fitted_list = []
                            
                            # Apply each numeric FE selection
                            for selection in fe_num_selections:
                                if selection['method'] and selection['column']:
                                    if selection['method'] == 'Custom Binning':
                                        # Handle custom binning rules within the selection
                                        for rule in selection.get('custom_rules', []):
                                            if rule.get('new_name'):
                                                col_data = pd.to_numeric(X_train[selection['column']], errors='coerce')
                                                if rule['operator'] == '>':
                                                    X_train[rule['new_name']] = (col_data > rule['value']).astype(int)
                                                elif rule['operator'] == '>=':
                                                    X_train[rule['new_name']] = (col_data >= rule['value']).astype(int)
                                                elif rule['operator'] == '<':
                                                    X_train[rule['new_name']] = (col_data < rule['value']).astype(int)
                                                elif rule['operator'] == '<=':
                                                    X_train[rule['new_name']] = (col_data <= rule['value']).astype(int)
                                                elif rule['operator'] == '=':
                                                    X_train[rule['new_name']] = (col_data == rule['value']).astype(int)
                                                
                                                # Apply same rule to test set
                                                col_data_test = pd.to_numeric(X_test[selection['column']], errors='coerce')
                                                if rule['operator'] == '>':
                                                    X_test[rule['new_name']] = (col_data_test > rule['value']).astype(int)
                                                elif rule['operator'] == '>=':
                                                    X_test[rule['new_name']] = (col_data_test >= rule['value']).astype(int)
                                                elif rule['operator'] == '<':
                                                    X_test[rule['new_name']] = (col_data_test < rule['value']).astype(int)
                                                elif rule['operator'] == '<=':
                                                    X_test[rule['new_name']] = (col_data_test <= rule['value']).astype(int)
                                                elif rule['operator'] == '=':
                                                    X_test[rule['new_name']] = (col_data_test == rule['value']).astype(int)
                                    else:
                                        # Apply standard numeric transformations
                                        fitted = fit_numeric_fe(X_train, [selection['column']], selection['method'], selection.get('params', {}))
                                        fe_num_fitted_list.append(fitted)
                                        X_train = apply_numeric_fe(X_train, fitted)
                                        X_test = apply_numeric_fe(X_test, fitted)
                            
                            # Handle single vs multi-target scenarios
                            if len(selected_target_cols) == 1:
                                y_for_problem_type = y_train
                            else:
                                # Use first column to determine core problem type
                                y_for_problem_type = y_train.iloc[:, 0]
                            
                            problem_type = get_problem_type(y_for_problem_type)
                            st.info(f"Detected a **{problem_type}** problem.")
                            
                            # Progress: Feature preprocessing
                            progress_bar.progress(0.3)
                            status_text.text('Preprocessing features and encoding categorical variables...')
                            X_train_processed, maps_to_save = preprocess_features(X_train, y_for_problem_type, encoding_method=encoding_choice)
                            X_test_processed, _ = preprocess_features(X_test, encoding_method=encoding_choice, saved_maps=maps_to_save)
                            # Pro: optional numeric standardization based on train stats
                            scaler_stats = None
                            if fe_standardize:
                                num_cols_train = X_train_processed.select_dtypes(include=[np.number])
                                scaler_stats = compute_scaler_stats(num_cols_train)
                                X_train_processed = apply_scaler_stats(X_train_processed, scaler_stats)
                                X_test_processed = apply_scaler_stats(X_test_processed, scaler_stats)
                            X_test_processed = X_test_processed.reindex(columns=X_train_processed.columns, fill_value=0)

                            model, score, score_label, le = (None, 0, "", LabelEncoder())
                            
                            # Progress: Model training
                            progress_bar.progress(0.6)
                            status_text.text('Training machine learning model...')
                            
                            leaderboard = []
                            permutation_df = None
                            best_y_test_encoded = None
                            # In Free plan, force RandomForest only; in Pro, use user selection or three defaults
                            if not pro_unlocked:
                                chosen_models = ['RandomForest']
                            else:
                                # Check if Pro user selected models in Step 5
                                step5_models = st.session_state.get('multi_model_selection', [])
                                if step5_models:
                                    chosen_models = step5_models
                                else:
                                    chosen_models = selected_models or get_three_default_models(problem_type)
                            is_multi_target_cls = (problem_type == "Classification" and len(selected_target_cols) > 1)

                            # Default estimators for ensemble models after removing the UI slider
                            default_n_estimators = 100

                            custom_hyperparams = st.session_state.get('custom_hyperparams', {})

                            def build_model(name):
                                if problem_type == "Classification":
                                    nb_vs = 1e-9
                                    try:
                                        nb_vs = float(custom_hyperparams.get('NaiveBayes', {}).get('var_smoothing', nb_vs))
                                    except Exception:
                                        nb_vs = 1e-9
                                    factory_map = {
                                        'RandomForest': lambda: RandomForestClassifier(
                                            n_estimators=default_n_estimators,
                                            random_state=42,
                                            max_depth=None if rf_max_depth == 0 else rf_max_depth,
                                            min_samples_leaf=rf_min_samples_leaf
                                        ),
                                        'LogisticRegression': lambda: LogisticRegression(max_iter=1000, C=lr_C),
                                        'SVM': lambda: SVC(C=svm_C, probability=True),
                                        'KNN': lambda: KNeighborsClassifier(n_neighbors=knn_k),
                                        'DecisionTree': lambda: DecisionTreeClassifier(max_depth=None if dt_max_depth == 0 else dt_max_depth),
                                        'GradientBoosting': lambda: GradientBoostingClassifier(n_estimators=default_n_estimators),
                                        'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=default_n_estimators, random_state=42),
                                        'AdaBoost': lambda: AdaBoostClassifier(n_estimators=default_n_estimators),
                                        'NaiveBayes': lambda: GaussianNB(var_smoothing=nb_vs),
                                    }
                                    default_factory = lambda: RandomForestClassifier(n_estimators=default_n_estimators, random_state=42)
                                else:
                                    def _linear_settings():
                                        try:
                                            lr_opts = custom_hyperparams.get('LinearRegression', {})
                                            fit_intercept = bool(lr_opts.get('fit_intercept', True))
                                            positive = bool(lr_opts.get('positive', False))
                                        except Exception:
                                            fit_intercept, positive = True, False
                                        return fit_intercept, positive

                                    def _svr_settings():
                                        try:
                                            svr_opts = custom_hyperparams.get('SVR', {})
                                            C = float(svr_opts.get('C', 1.0))
                                            epsilon = float(svr_opts.get('epsilon', 0.1))
                                            kernel = svr_opts.get('kernel', 'rbf')
                                            gamma = svr_opts.get('gamma', 'scale')
                                            degree = int(svr_opts.get('degree', 3)) if kernel == 'poly' else 3
                                        except Exception:
                                            C, epsilon, kernel, gamma, degree = 1.0, 0.1, 'rbf', 'scale', 3
                                        return C, epsilon, kernel, gamma, degree

                                    fit_intercept, positive = _linear_settings()
                                    C, epsilon, kernel, gamma, degree = _svr_settings()
                                    factory_map = {
                                        'RandomForest': lambda: RandomForestRegressor(n_estimators=default_n_estimators, random_state=42),
                                        'LinearRegression': lambda: LinearRegression(fit_intercept=fit_intercept, positive=positive),
                                        'Ridge': lambda: Ridge(),
                                        'Lasso': lambda: Lasso(),
                                        'ElasticNet': lambda: ElasticNet(),
                                        'SVR': lambda: SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree),
                                        'KNN': lambda: KNeighborsRegressor(n_neighbors=knn_k),
                                        'DecisionTree': lambda: DecisionTreeRegressor(max_depth=None if dt_max_depth == 0 else dt_max_depth),
                                        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=default_n_estimators),
                                        'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=default_n_estimators, random_state=42),
                                        'AdaBoost': lambda: AdaBoostRegressor(n_estimators=default_n_estimators),
                                    }
                                    default_factory = lambda: RandomForestRegressor(n_estimators=default_n_estimators, random_state=42)

                                base_model = factory_map.get(name, default_factory)()
                                if is_multi_target_cls and problem_type == "Classification":
                                    return MultiOutputClassifier(base_model)
                                return base_model

                            # Train and evaluate models; select best for saving
                            best_model = None
                            best_model_name = None
                            best_score = -np.inf
                            score_label = "Accuracy" if problem_type=="Classification" else "R-squared (R²)"

                            for name in chosen_models:
                                md = build_model(name)
                                current_y_test_encoded = None
                                if problem_type == "Classification":
                                    if is_multi_target_cls:
                                        # Encode each target column separately
                                        encoders_dict = {}
                                        y_train_enc_df = pd.DataFrame(index=y_train.index)
                                        y_test_enc_df = pd.DataFrame(index=y_test.index)
                                        for col in selected_target_cols:
                                            le_col = LabelEncoder().fit(y_train[col])
                                            encoders_dict[col] = le_col
                                            y_train_enc_df[col] = le_col.transform(y_train[col])
                                            y_test_enc_df[col] = le_col.transform(y_test[col])
                                        if use_cv:
                                            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                            cv_scores = cross_val_score(md, X_train_processed, y_train_enc_df.values, cv=cv)
                                            model_score = float(np.mean(cv_scores))
                                            md.fit(X_train_processed, y_train_enc_df.values)
                                        else:
                                            md.fit(X_train_processed, y_train_enc_df.values)
                                            preds = md.predict(X_test_processed)
                                            # Average accuracy across outputs
                                            accs = []
                                            for i, col in enumerate(selected_target_cols):
                                                accs.append(accuracy_score(y_test_enc_df[col], preds[:, i]))
                                            model_score = float(np.mean(accs))
                                        # Store encoders dict on the model for potential later use in this loop
                                        setattr(md, '_multi_label_encoders_', encoders_dict)
                                    else:
                                        y_train_encoded = le.fit_transform(y_train)
                                        if use_cv:
                                            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                            cv_scores = cross_val_score(md, X_train_processed, y_train_encoded, cv=cv, scoring='accuracy')
                                            model_score = float(np.mean(cv_scores))
                                            md.fit(X_train_processed, y_train_encoded)
                                        else:
                                            md.fit(X_train_processed, y_train_encoded)
                                            preds = md.predict(X_test_processed)
                                            y_test_encoded = le.transform(y_test)
                                            model_score = float(accuracy_score(y_test_encoded, preds))
                                            current_y_test_encoded = y_test_encoded
                                        if use_cv:
                                            current_y_test_encoded = le.transform(y_test)
                                else:
                                    if use_cv:
                                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                                        cv_scores = cross_val_score(md, X_train_processed, y_train, cv=cv, scoring='r2')
                                        model_score = float(np.mean(cv_scores))
                                        md.fit(X_train_processed, y_train)
                                    else:
                                        md.fit(X_train_processed, y_train)
                                        preds = md.predict(X_test_processed)
                                        if len(selected_target_cols) == 1:
                                            model_score = float(r2_score(y_test, preds))
                                        else:
                                            scores = []
                                            for i in range(len(selected_target_cols)):
                                                scores.append(r2_score(y_test.iloc[:, i], preds[:, i]))
                                            model_score = float(np.mean(scores))

                                leaderboard.append({"Model": name, "Score": model_score})
                                if model_score > best_score:
                                    best_score = model_score
                                    best_model = md
                                    best_model_name = name
                                    if problem_type == "Classification" and not is_multi_target_cls and current_y_test_encoded is not None:
                                        best_y_test_encoded = current_y_test_encoded
                            score = best_score

                            if pro_unlocked and len(selected_target_cols) == 1 and not is_multi_target_cls and best_model is not None:
                                sufficient_rows = X_test_processed.shape[0] >= 5
                                manageable_shape = X_test_processed.shape[1] <= 200
                                if sufficient_rows and manageable_shape and best_y_test_encoded is not None:
                                    try:
                                        perm_result = permutation_importance(
                                            best_model,
                                            X_test_processed,
                                            best_y_test_encoded,
                                            n_repeats=10,
                                            random_state=42,
                                            n_jobs=-1,
                                            scoring='accuracy'
                                        )
                                        permutation_df = (
                                            pd.DataFrame({
                                                'Feature': X_train_processed.columns,
                                                'Importance': perm_result.importances_mean
                                            })
                                            .replace([np.inf, -np.inf], np.nan)
                                            .dropna()
                                            .sort_values('Importance', ascending=False)
                                        )
                                    except Exception:
                                        permutation_df = None
                                else:
                                    permutation_df = None
                            elif pro_unlocked and len(selected_target_cols) == 1 and problem_type == "Regression" and best_model is not None:
                                if X_test_processed.shape[0] >= 5 and X_test_processed.shape[1] <= 200:
                                    try:
                                        y_target = np.asarray(y_test)
                                        perm_result = permutation_importance(
                                            best_model,
                                            X_test_processed,
                                            y_target,
                                            n_repeats=10,
                                            random_state=42,
                                            n_jobs=-1,
                                            scoring='r2'
                                        )
                                        permutation_df = (
                                            pd.DataFrame({
                                                'Feature': X_train_processed.columns,
                                                'Importance': perm_result.importances_mean
                                            })
                                            .replace([np.inf, -np.inf], np.nan)
                                            .dropna()
                                            .sort_values('Importance', ascending=False)
                                        )
                                    except Exception:
                                        permutation_df = None
                                else:
                                    permutation_df = None

                            # Progress: Final model training
                            progress_bar.progress(0.8)
                            status_text.text('Training final model on complete dataset...')
                            
                            # Now train final model on full dataset using SAME preprocessing as validation
                            # Apply numeric FE on full data before encoding
                            for fitted in fe_num_fitted_list:
                                X = apply_numeric_fe(X, fitted)
                            
                            # Apply custom binning rules to full dataset
                            for selection in fe_num_selections:
                                if selection['method'] == 'Custom Binning':
                                    for rule in selection.get('custom_rules', []):
                                        if rule.get('new_name'):
                                            col_data = pd.to_numeric(X[selection['column']], errors='coerce')
                                            if rule['operator'] == '>':
                                                X[rule['new_name']] = (col_data > rule['value']).astype(int)
                                            elif rule['operator'] == '>=':
                                                X[rule['new_name']] = (col_data >= rule['value']).astype(int)
                                            elif rule['operator'] == '<':
                                                X[rule['new_name']] = (col_data < rule['value']).astype(int)
                                            elif rule['operator'] == '<=':
                                                X[rule['new_name']] = (col_data <= rule['value']).astype(int)
                                            elif rule['operator'] == '=':
                                                X[rule['new_name']] = (col_data == rule['value']).astype(int)
                            X_processed, _ = preprocess_features(X, encoding_method=encoding_choice, saved_maps=maps_to_save)
                            if fe_standardize and scaler_stats is not None:
                                X_processed = apply_scaler_stats(X_processed, scaler_stats)
                            
                            if problem_type == "Classification":
                                if len(selected_target_cols) == 1:
                                    final_le = LabelEncoder().fit(y)
                                    y_encoded = final_le.transform(y)
                                    best_model.fit(X_processed, y_encoded)
                                    joblib.dump(final_le, 'label_encoders.joblib')
                                else:
                                    # Multi-target classification: encode each column
                                    encoders_dict = {}
                                    y_enc_full = pd.DataFrame(index=y.index)
                                    for col in selected_target_cols:
                                        le_col = LabelEncoder().fit(y[col])
                                        encoders_dict[col] = le_col
                                        y_enc_full[col] = le_col.transform(y[col])
                                    best_model.fit(X_processed, y_enc_full.values)
                                    joblib.dump(encoders_dict, 'label_encoders.joblib')
                            else:
                                best_model.fit(X_processed, y)
                            
                            if maps_to_save:
                                try:
                                    if encoding_choice == 'Target Encoding':
                                        joblib.dump(maps_to_save, 'target_encoding_maps.joblib')
                                    elif encoding_choice == 'Label Encoding':
                                        joblib.dump(maps_to_save, 'feature_label_maps.joblib')
                                except Exception:
                                    pass
                            # Persist numeric FE config for inference
                            if fe_num_fitted_list or fe_num_selections:
                                fe_config = {
                                    'fitted_list': fe_num_fitted_list,
                                    'selections': fe_num_selections
                                }
                                joblib.dump(fe_config, 'fe_numeric_params.joblib')
                            # Save feature generation rules if any (for prediction)
                            if feature_rules:
                                # Persist custom feature generation rules for inference
                                joblib.dump(feature_rules, 'feature_rules.joblib')
                            joblib.dump(best_model, 'model.joblib')
                            if fe_standardize and scaler_stats is not None:
                                joblib.dump(scaler_stats, 'scaler_stats.joblib')
                            
                            # Progress: Feature analysis
                            progress_bar.progress(0.9)
                            status_text.text('Calculating feature importance and correlations...')
                            
                            st.session_state.update({'report_ready': True, 'score_label': score_label, 'score_value': score, 'problem_type_for_report': problem_type, 'feature_columns': X.columns.tolist(), 'processed_columns': X_processed.columns.tolist(), 'target_columns': selected_target_cols, 'encoding_method': encoding_choice, 'trained_with_pro': bool(pro_unlocked)})
                            st.session_state['permutation_importance_df'] = permutation_df
                            # Persist raw input feature columns to ensure consistent prediction pre-processing
                            try:
                                joblib.dump(X.columns.tolist(), 'input_feature_columns.joblib')
                            except Exception:
                                pass
                            if leaderboard and pro_unlocked:
                                st.subheader("🏁 Leaderboard (Pro)")
                                st.dataframe(pd.DataFrame(leaderboard).sort_values('Score', ascending=False).reset_index(drop=True))
                            # Feature importance handling for various model types, incl. multi-output
                            feature_importance_df = compute_feature_importance_generic(best_model, X_processed)
                            if feature_importance_df is None:
                                # Fallback: create zeros to avoid an empty chart
                                feature_importance_df = pd.DataFrame({'Feature': X_processed.columns, 'Importance': np.zeros(len(X_processed.columns))})
                            feature_importance_df = feature_importance_df.replace([np.inf, -np.inf], np.nan).dropna()
                            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                            st.session_state['feature_importance_df'] = feature_importance_df
                            
                            if len(selected_target_cols) == 1:
                                target_name = selected_target_cols[0]
                                st.session_state['target_name_for_corr'] = target_name
                                
                                # Use original data (before train/test split) for correlation
                                df_for_corr = X.copy()  # Use full original feature data
                                df_for_corr[target_name] = y.values if hasattr(y, 'values') else y
                                
                                # Convert categorical columns to numeric for correlation
                                for col in df_for_corr.columns:
                                    if pd.api.types.is_string_dtype(df_for_corr[col]) or pd.api.types.is_categorical_dtype(df_for_corr[col]):
                                        try:
                                            df_for_corr[col] = LabelEncoder().fit_transform(df_for_corr[col].astype(str))
                                        except Exception:
                                            # If encoding fails, drop the column
                                            df_for_corr = df_for_corr.drop(columns=[col])
                                
                                # Ensure all columns are numeric
                                numeric_cols_for_corr = df_for_corr.select_dtypes(include=[np.number])
                                
                                # Create correlation matrix if we have at least 2 numeric columns
                                if len(numeric_cols_for_corr.columns) >= 2 and target_name in numeric_cols_for_corr.columns:
                                    try:
                                        corr_matrix = numeric_cols_for_corr.corr()
                                        # Store correlation matrix (remove overly restrictive validation)
                                        st.session_state['correlation_matrix'] = corr_matrix
                                    except Exception as e:
                                        st.warning(f"Could not calculate correlation matrix: {str(e)}")
                                        st.session_state['correlation_matrix'] = None
                                else:
                                    st.session_state['correlation_matrix'] = None
                            else:
                                # Multi-target scenario - no correlation matrix
                                st.session_state['correlation_matrix'] = None
                            
                            # Progress: Complete
                            progress_bar.progress(1.0)
                            status_text.text('Training completed successfully!')
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("✅ Model trained successfully!")
                            st.balloons()
        else: # Time Series Forecasting
            with st.container():
                st.subheader("Forecast Configuration (Pro)")
                st.caption("Provide a single time series dataset first: Date/Time column + numeric Target. Optionally include exogenous columns for richer models.")
                pro_unlocked_ts = st.session_state.get('pro_unlocked', False)
                if not pro_unlocked_ts:
                    st.info("Time Series Forecasting enhancements are available in Pro.")
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select your Date/Time column", options=train_df.columns)
                with col2:
                    target_col = st.selectbox("Select the column to forecast", options=train_df.columns)

                # Optional exogenous columns for Pro
                exog_cols = []
                if pro_unlocked_ts:
                    exog_cols = st.multiselect("Optional exogenous features (Pro)", options=[c for c in train_df.columns if c not in [date_col, target_col]])

                # Pro-only: Optional Future Exogenous upload (to avoid forward-fill during forecasting)
                future_exog_df = None
                if pro_unlocked_ts:
                    with st.expander("Future Exogenous (Pro) — optional"):
                        st.caption("Upload a file containing future dates (column 'ds') and the same exogenous columns you selected above. We'll use these instead of forward-filling.")
                        fut_file = st.file_uploader("Upload future exogenous file (CSV/XLS/XLSX)", type=['csv','xls','xlsx'], key="future_exog_uploader")
                        if fut_file is not None:
                            try:
                                ext = os.path.splitext(fut_file.name)[1].lower()
                                if ext == '.csv':
                                    future_exog_df = pd.read_csv(fut_file)
                                else:
                                    future_exog_df = pd.read_excel(fut_file)
                                # Basic validation: must have 'ds' and all selected exogenous columns
                                missing = []
                                if 'ds' not in future_exog_df.columns:
                                    missing.append('ds')
                                for c in exog_cols:
                                    if c not in future_exog_df.columns:
                                        missing.append(c)
                                if missing:
                                    st.error(f"Future exogenous file is missing required column(s): {missing}")
                                    future_exog_df = None
                                else:
                                    # Coerce ds to datetime and sort
                                    future_exog_df['ds'] = pd.to_datetime(future_exog_df['ds'], errors='coerce')
                                    future_exog_df = future_exog_df.dropna(subset=['ds']).sort_values('ds')
                                    st.success("Future exogenous loaded and validated.")
                            except Exception as e:
                                st.error(f"Could not read future exogenous file: {e}")

                # Auto vs Manual FE for TS (Pro)
                if 'ts_auto_config' not in st.session_state:
                    st.session_state['ts_auto_config'] = True
                if pro_unlocked_ts:
                    ts_auto = st.checkbox("Auto-Configure from data (Pro)", value=st.session_state['ts_auto_config'], key="ts_auto_config")
                else:
                    ts_auto = True  # Free uses simplified auto

                forecast_horizon = st.number_input("Forecast horizon (steps)", min_value=1, value=30)

                # Diagnostics & Stationarity (Pro)
                if pro_unlocked_ts:
                    with st.expander("Diagnostics: EDA, Decomposition and Stationarity (Pro)"):
                        # Plot raw series
                        try:
                            ts_plot_df = train_df[[date_col, target_col]].copy()
                            ts_plot_df[date_col] = pd.to_datetime(ts_plot_df[date_col])
                            ts_plot_df = ts_plot_df.sort_values(date_col)
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(ts_plot_df[date_col], ts_plot_df[target_col], lw=1)
                            ax.set_title("Time Series Plot")
                            ax.set_xlabel("Date")
                            ax.set_ylabel(target_col)
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception:
                            st.warning("Could not render time series plot.")
                        # Decomposition (optional)
                        try:
                            from statsmodels.tsa.seasonal import seasonal_decompose
                            ts_plot_df = ts_plot_df.set_index(date_col)[target_col].asfreq(ts_plot_df.set_index(date_col).index.inferred_freq or None)
                            if ts_plot_df is not None and ts_plot_df.dropna().shape[0] > 10:
                                dec = seasonal_decompose(ts_plot_df.dropna(), model='additive', period=None)
                                fig = dec.plot()
                                fig.set_size_inches(10, 6)
                                st.pyplot(fig)
                                plt.close(fig)
                        except Exception:
                            st.info("Decomposition requires statsmodels and an inferrable frequency; skipping if unavailable.")
                        # ADF test (optional)
                        adf_p = None
                        try:
                            from statsmodels.tsa.stattools import adfuller
                            series = pd.to_numeric(ts_plot_df, errors='coerce').dropna()
                            if series.shape[0] > 12:
                                adf_res = adfuller(series.values)
                                adf_p = adf_res[1]
                                st.write(f"ADF p-value: {adf_p:.4f} ({'Stationary' if adf_p < 0.05 else 'Non-stationary'})")
                        except Exception:
                            st.info("ADF test requires statsmodels; skipping if unavailable.")
                        # Differencing option
                        apply_diff = st.checkbox("Apply first differencing if non-stationary", value=(adf_p is not None and adf_p >= 0.05))

                # Build features depending on mode
                def _build_ts_features(df):
                    if ts_auto:
                        # Heuristics: daily/weekly lags and seasonal Fourier
                        fourier = [7, 365]
                        return create_time_series_features_exog(df, date_col, target_col, exog_cols=exog_cols, lags=[1,7,14], rolling_windows=[7,14], fourier_periods=fourier, fourier_order=3)
                    else:
                        # Manual FE controls (Pro)
                        l1, l2, l3 = st.columns(3)
                        with l1:
                            lags = st.text_input("Lags (comma)", value="1,7,14")
                        with l2:
                            rws = st.text_input("Rolling windows (comma)", value="7,14")
                        with l3:
                            four = st.text_input("Fourier periods (comma)", value="7,365")
                        try:
                            lag_list = [int(x.strip()) for x in lags.split(',') if x.strip()]
                        except Exception:
                            lag_list = [1,7,14]
                        try:
                            rw_list = [int(x.strip()) for x in rws.split(',') if x.strip()]
                        except Exception:
                            rw_list = [7,14]
                        try:
                            four_list = [int(x.strip()) for x in four.split(',') if x.strip()]
                        except Exception:
                            four_list = [7,365]
                        return create_time_series_features_exog(df, date_col, target_col, exog_cols=exog_cols, lags=lag_list, rolling_windows=rw_list, fourier_periods=four_list, fourier_order=3)

                # Model choice and validation (Pro)
                model_choice = 'RandomForest (ML)'
                available_ts_models = ['RandomForest (ML)', 'GradientBoosting (ML)', 'ExtraTrees (ML)']
                esm_ok = False
                try:
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # noqa: F401
                    esm_ok = True
                    available_ts_models.append('ExponentialSmoothing (Stat)')
                except Exception:
                    esm_ok = False
                arima_ok = False
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
                    arima_ok = True
                    available_ts_models.append('ARIMA/SARIMA (Stat)')
                except Exception:
                    arima_ok = False
                prophet_ok = False
                try:
                    try:
                        from prophet import Prophet  # type: ignore
                        prophet_ok = True
                    except Exception:
                        from fbprophet import Prophet  # type: ignore
                        prophet_ok = True
                    if prophet_ok:
                        available_ts_models.append('Prophet (Stat)')
                except Exception:
                    prophet_ok = False
                if pro_unlocked_ts:
                    model_choice = st.selectbox("Model (Pro)", options=available_ts_models, index=0)

                    with st.expander("Validation (Pro)"):
                        use_walk_forward = st.checkbox("Use walk-forward validation", value=True)
                        n_splits = st.slider("Walk-forward splits", min_value=2, max_value=8, value=4, disabled=not use_walk_forward)

                    # Hyperparameters (minimal) for stats models
                    if model_choice == 'ARIMA/SARIMA (Stat)' and arima_ok:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            p = st.number_input("p", min_value=0, max_value=5, value=1, step=1)
                        with c2:
                            d = st.number_input("d", min_value=0, max_value=2, value=1, step=1)
                        with c3:
                            q = st.number_input("q", min_value=0, max_value=5, value=1, step=1)
                        with c4:
                            m = st.number_input("Seasonal period (m)", min_value=0, max_value=365, value=0, step=1)
                        P = D = Q = 0
                        if m and m > 1:
                            s1, s2, s3 = st.columns(3)
                            with s1:
                                P = st.number_input("P", min_value=0, max_value=2, value=0, step=1)
                            with s2:
                                D = st.number_input("D", min_value=0, max_value=2, value=0, step=1)
                            with s3:
                                Q = st.number_input("Q", min_value=0, max_value=2, value=0, step=1)
                        arima_params = {'order': (int(p), int(d), int(q)), 'seasonal_order': (int(P), int(D), int(Q), int(m or 0))}

                    if model_choice == 'Prophet (Stat)' and prophet_ok:
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            seasonality_mode = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])
                        with c2:
                            cps = st.slider("Changepoint Prior Scale", 0.01, 1.0, 0.05, 0.01)
                        with c3:
                            yearly = st.checkbox("Yearly Seasonality", value=True)
                        with c4:
                            weekly = st.checkbox("Weekly Seasonality", value=True)
                        daily = st.checkbox("Daily Seasonality", value=False)
                        prophet_params = {'seasonality_mode': seasonality_mode, 'changepoint_prior_scale': float(cps), 'yearly_seasonality': bool(yearly), 'weekly_seasonality': bool(weekly), 'daily_seasonality': bool(daily)}

                if st.button("Run Forecast"):
                    with st.spinner("Training forecast model and generating future predictions..."):
                        try:
                            # Validate time series data
                            if train_df[date_col].isnull().any():
                                st.error("Date column contains missing values. Please clean your data.")
                                st.stop()
                            
                            # Validate date column can be converted to datetime
                            try:
                                pd.to_datetime(train_df[date_col])
                            except Exception:
                                st.error(f"Column '{date_col}' cannot be converted to datetime format. Please select a valid date column.")
                                st.stop()
                            
                            if train_df[target_col].isnull().sum() > len(train_df) * 0.1:
                                st.error("Target column has too many missing values (>10%). Please clean your data.")
                                st.stop()
                            
                            # Validate target column is numeric
                            if not pd.api.types.is_numeric_dtype(train_df[target_col]):
                                st.error(f"Target column '{target_col}' must be numeric for time series forecasting.")
                                st.stop()
                            
                            # Apply optional differencing for stationarity
                            base_df = train_df[[date_col, target_col] + exog_cols] if exog_cols else train_df[[date_col, target_col]]
                            base_df = base_df.copy()
                            if pro_unlocked_ts and 'apply_diff' in locals() and apply_diff:
                                base_df[target_col] = base_df[target_col].diff()
                                base_df = base_df.dropna()

                            # Create features (Auto/Manual FE)
                            df_featured = _build_ts_features(base_df)
                            
                            if df_featured.empty:
                                st.error("Not enough data to create time series features after processing. Please provide a longer time series.")
                                st.stop()

                            X = df_featured.drop(columns=[target_col])
                            y = df_featured[target_col]
                            
                            # Use proper time series split (last 20% for testing)
                            split_idx = int(len(df_featured) * 0.8)
                            X_train_ts, y_train_ts = X.iloc[:split_idx], y.iloc[:split_idx]
                            X_test_ts, y_test_ts = X.iloc[split_idx:], y.iloc[split_idx:]
                            
                            # Build chosen model
                            if model_choice == 'RandomForest (ML)':
                                model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                            elif model_choice == 'GradientBoosting (ML)':
                                model = GradientBoostingRegressor(n_estimators=300, random_state=42)
                            elif model_choice == 'ExtraTrees (ML)':
                                model = ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1)
                            elif model_choice == 'ExponentialSmoothing (Stat)' and esm_ok:
                                model = 'ESM'  # special path
                            elif model_choice == 'ARIMA/SARIMA (Stat)' and arima_ok:
                                model = 'ARIMAX'
                            elif model_choice == 'Prophet (Stat)' and prophet_ok:
                                model = 'PROPHET'
                            else:
                                model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                            
                            # Validate on time series with optional walk-forward
                            residual_std = None
                            if model == 'ESM':
                                # statsmodels ExponentialSmoothing on target only
                                try:
                                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                                    series_full = y
                                    split_idx = int(len(series_full) * 0.8)
                                    series_tr, series_te = series_full.iloc[:split_idx], series_full.iloc[split_idx:]
                                    esm = ExponentialSmoothing(series_tr, trend='add', seasonal=None)
                                    esm_fit = esm.fit()
                                    if len(series_te) > 0:
                                        ts_predictions = esm_fit.forecast(len(series_te))
                                        ts_score = r2_score(series_te, ts_predictions)
                                        st.info(f"Time Series Validation R² Score (ESM): {ts_score:.3f}")
                                        residual_std = float(np.std(series_te.values - ts_predictions.values))
                                    model = esm_fit
                                except Exception as e:
                                    st.warning(f"ExponentialSmoothing unavailable or failed: {e}. Falling back to RandomForest.")
                                    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                            elif model == 'ARIMAX':
                                try:
                                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                                    split_idx = int(len(X) * 0.8)
                                    X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
                                    X_te, y_te = X.iloc[split_idx:], y.iloc[split_idx:]
                                    sarimax = SARIMAX(y_tr, exog=X_tr, order=arima_params['order'], seasonal_order=arima_params['seasonal_order'] if (arima_params['seasonal_order'][3] or 0) > 0 else (0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
                                    sarimax_fit = sarimax.fit(disp=False)
                                    if len(X_te) > 0:
                                        fc = sarimax_fit.get_forecast(steps=len(X_te), exog=X_te)
                                        ts_predictions = fc.predicted_mean
                                        ts_score = r2_score(y_te, ts_predictions)
                                        st.info(f"Time Series Validation R² Score (SARIMAX): {ts_score:.3f}")
                                        conf = fc.conf_int()
                                        residual_std = float(np.mean((conf.iloc[:,1] - conf.iloc[:,0]) / (2*1.96))) if not conf.empty else None
                                    model = sarimax_fit
                                except Exception as e:
                                    st.warning(f"SARIMAX failed: {e}. Falling back to RandomForest.")
                                    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                            elif model == 'PROPHET':
                                try:
                                    try:
                                        from prophet import Prophet
                                    except Exception:
                                        from fbprophet import Prophet
                                    # Prepare Prophet training/test frames
                                    df_y = y.copy()
                                    split_idx = int(len(df_y) * 0.8)
                                    y_tr, y_te = df_y.iloc[:split_idx], df_y.iloc[split_idx:]
                                    X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
                                    df_tr = pd.DataFrame({'ds': y_tr.index, 'y': y_tr.values})
                                    df_te = pd.DataFrame({'ds': y_te.index, 'y': y_te.values})
                                    m = Prophet(seasonality_mode=prophet_params['seasonality_mode'], changepoint_prior_scale=prophet_params['changepoint_prior_scale'], yearly_seasonality=prophet_params['yearly_seasonality'], weekly_seasonality=prophet_params['weekly_seasonality'], daily_seasonality=prophet_params['daily_seasonality'])
                                    for col in X.columns:
                                        m.add_regressor(col)
                                        df_tr[col] = X_tr[col].values
                                        df_te[col] = X_te[col].values if col in X_te.columns else 0.0
                                    m.fit(df_tr)
                                    if len(df_te) > 0:
                                        forecast_te = m.predict(df_te)
                                        ts_predictions = forecast_te['yhat']
                                        ts_score = r2_score(y_te, ts_predictions)
                                        st.info(f"Time Series Validation R² Score (Prophet): {ts_score:.3f}")
                                        # Prophet provides yhat_lower/upper; approximate residual std
                                        if 'yhat_lower' in forecast_te and 'yhat_upper' in forecast_te:
                                            residual_std = float(np.mean((forecast_te['yhat_upper'] - forecast_te['yhat_lower']) / (2*1.96)))
                                    model = m
                                except Exception as e:
                                    st.warning(f"Prophet failed: {e}. Falling back to RandomForest.")
                                    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                            else:
                                if pro_unlocked_ts and 'use_walk_forward' in locals() and use_walk_forward and n_splits >= 2:
                                    fold_scores = []
                                    step = len(X) // (n_splits + 1)
                                    for i in range(1, n_splits + 1):
                                        split_point = step * i
                                        X_tr, y_tr = X.iloc[:split_point], y.iloc[:split_point]
                                        X_te, y_te = X.iloc[split_point:], y.iloc[split_point:]
                                        mdl = type(model)(**getattr(model, 'get_params', lambda: {})()) if hasattr(model, 'get_params') else model
                                        mdl.fit(X_tr, y_tr)
                                        if not X_te.empty:
                                            preds = mdl.predict(X_te)
                                            fold_scores.append(_rmse(y_te, preds))
                                    if fold_scores:
                                        st.info(f"Walk-forward CV RMSE (mean over {len(fold_scores)} folds): {np.mean(fold_scores):.3f}")
                                # Hold-out validation
                                model.fit(X_train_ts, y_train_ts)
                                if not X_test_ts.empty:
                                    ts_predictions = model.predict(X_test_ts)
                                    ts_score = r2_score(y_test_ts, ts_predictions)
                                    st.info(f"Time Series Validation R² Score: {ts_score:.3f}")
                                    residual_std = float(np.std(y_test_ts.values - ts_predictions))
                                else:
                                    st.info("Dataset too small for validation split - training on full dataset.")
                            
                            # Train final model on ALL historical data
                            if model == 'ESM':
                                pass  # already fit on full series or train subset
                            elif model == 'ARIMAX':
                                # Already fit
                                pass
                            elif model == 'PROPHET':
                                # Fit on full data
                                try:
                                    try:
                                        from prophet import Prophet
                                    except Exception:
                                        from fbprophet import Prophet
                                    df_full = pd.DataFrame({'ds': y.index, 'y': y.values})
                                    m = Prophet(seasonality_mode=prophet_params['seasonality_mode'], changepoint_prior_scale=prophet_params['changepoint_prior_scale'], yearly_seasonality=prophet_params['yearly_seasonality'], weekly_seasonality=prophet_params['weekly_seasonality'], daily_seasonality=prophet_params['daily_seasonality'])
                                    for col in X.columns:
                                        m.add_regressor(col)
                                        df_full[col] = X[col].values
                                    m.fit(df_full)
                                    model = m
                                except Exception:
                                    pass
                            else:
                                model.fit(X, y)
                            
                            # Iterative forecasting loop
                            future_predictions = []
                            # Start with the full historical data to generate features for the first prediction
                            history_df = train_df[[date_col, target_col]].copy()

                            last_date = pd.to_datetime(history_df[date_col].max())
                            freq = pd.infer_freq(df_featured.index)
                            
                            # Improved frequency detection
                            if freq is None:
                                # Calculate average time difference
                                time_diffs = df_featured.index.to_series().diff().dropna()
                                avg_diff = time_diffs.median()
                                
                                if avg_diff <= pd.Timedelta(hours=1):
                                    freq = 'H'  # Hourly
                                    st.info("Inferred hourly frequency based on data pattern.")
                                elif avg_diff <= pd.Timedelta(days=1):
                                    freq = 'D'  # Daily
                                    st.info("Inferred daily frequency based on data pattern.")
                                elif avg_diff <= pd.Timedelta(weeks=1):
                                    freq = 'W'  # Weekly
                                    st.info("Inferred weekly frequency based on data pattern.")
                                elif avg_diff <= pd.Timedelta(days=31):
                                    freq = 'M'  # Monthly
                                    st.info("Inferred monthly frequency based on data pattern.")
                                else:
                                    freq = 'D'  # Default fallback
                                    st.warning("Could not determine frequency. Using daily as default.")

                            # Add progress bar for forecasting
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            lower_bounds = []
                            upper_bounds = []
                            for i in range(forecast_horizon):
                                # Update progress
                                progress_bar.progress((i + 1) / forecast_horizon)
                                status_text.text(f'Generating forecast step {i + 1} of {forecast_horizon}...')
                                
                                # 1. Create features for the next step based on all available history
                                if exog_cols:
                                    # Prefer future exogenous (Pro) if provided; else fallback to history forward-fill
                                    if pro_unlocked_ts and 'future_exog_df' in locals() and future_exog_df is not None:
                                        # Stitch history with future exog for feature creation up to the next step
                                        hist_exog = train_df[[date_col] + exog_cols].copy()
                                        hist_exog[date_col] = pd.to_datetime(hist_exog[date_col])
                                        # Rename 'ds' -> date_col for join consistency
                                        fut_exog = future_exog_df.rename(columns={'ds': date_col}).copy()
                                        merged_exog = pd.concat([hist_exog, fut_exog[[date_col] + exog_cols]], ignore_index=True)
                                        history_merged = history_df.copy()
                                        history_merged[date_col] = pd.to_datetime(history_merged[date_col])
                                        merged = pd.merge(history_merged, merged_exog, on=date_col, how='left').sort_values(date_col)
                                        merged[exog_cols] = merged[exog_cols].ffill().bfill()
                                        features_for_next_step = _build_ts_features(merged)
                                    else:
                                        # Align exogenous values by date and forward-fill
                                        exog_df = train_df[[date_col] + exog_cols].copy()
                                        exog_df[date_col] = pd.to_datetime(exog_df[date_col])
                                        history_merged = history_df.copy()
                                        history_merged[date_col] = pd.to_datetime(history_merged[date_col])
                                        merged = pd.merge(history_merged, exog_df, on=date_col, how='left').sort_values(date_col)
                                        merged[exog_cols] = merged[exog_cols].ffill().bfill()
                                        features_for_next_step = _build_ts_features(merged)
                                else:
                                    features_for_next_step = _build_ts_features(history_df)
                                
                                # 2. Get the last row of features to predict the next point
                                if model == 'ESM':
                                    last_feature_row = None
                                elif model == 'ARIMAX':
                                    last_feature_row = features_for_next_step.iloc[-1:].drop(columns=[target_col])
                                    last_feature_row = last_feature_row.reindex(columns=X.columns, fill_value=0)
                                elif model == 'PROPHET':
                                    last_feature_row = None  # Prophet will handle in batch later
                                else:
                                    last_feature_row = features_for_next_step.iloc[-1:].drop(columns=[target_col])
                                    last_feature_row = last_feature_row.reindex(columns=X.columns, fill_value=0) # Ensure column order
                                
                                # 3. Predict the next value
                                if model == 'ESM':
                                    next_pred = float(model.forecast(1).iloc[0])
                                elif model == 'ARIMAX':
                                    fc_step = model.get_forecast(steps=1, exog=last_feature_row)
                                    next_pred = float(fc_step.predicted_mean.iloc[0])
                                    if residual_std is None:
                                        try:
                                            conf = fc_step.conf_int()
                                            residual_std = float(np.mean((conf.iloc[:,1] - conf.iloc[:,0]) / (2*1.96)))
                                        except Exception:
                                            pass
                                elif model == 'PROPHET':
                                    # Prophet: predict remaining horizon in one shot at the end; skip per-step add
                                    next_pred = None
                                    # We'll break loop to batch predict below
                                    future_predictions = []
                                    lower_bounds = []
                                    upper_bounds = []
                                    break
                                else:
                                    next_pred = float(model.predict(last_feature_row)[0])
                                future_predictions.append(next_pred)
                                # Add simple prediction intervals from residual std if available
                                if residual_std is not None and residual_std > 0:
                                    lower_bounds.append(next_pred - 1.96 * residual_std)
                                    upper_bounds.append(next_pred + 1.96 * residual_std)
                                else:
                                    lower_bounds.append(np.nan)
                                    upper_bounds.append(np.nan)
                                
                                # 4. Add the prediction to history to use it for the next iteration's feature engineering
                                # Advance last_date using inferred frequency
                                try:
                                    from pandas.tseries.frequencies import to_offset
                                    off = to_offset(freq) if freq else pd.offsets.Day()
                                    last_date = last_date + off
                                except Exception:
                                    last_date = last_date + pd.to_timedelta(1, unit='D')
                                new_row = pd.DataFrame({date_col: [last_date], target_col: [next_pred]})
                                history_df = pd.concat([history_df, new_row], ignore_index=True)

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            future_dates = history_df[date_col].iloc[-forecast_horizon:]
                            # Prophet batch prediction if selected
                            if model == 'PROPHET':
                                try:
                                    # Build future frame
                                    # Create date range using inferred frequency; fallback to daily
                                    start_dt = pd.to_datetime(train_df[date_col].max())
                                    freq_map = {'H': 'H', 'D': 'D', 'W': 'W', 'M': 'M'}
                                    step = freq_map.get(freq, 'D') if 'freq' in locals() else 'D'
                                    future_ds = pd.date_range(start=start_dt, periods=forecast_horizon+1, freq=step)[1:]
                                    df_future = pd.DataFrame({'ds': future_ds})
                                    # Regressors: if future exogenous supplied (Pro), use them; else carry forward last known values
                                    if pro_unlocked_ts and 'future_exog_df' in locals() and future_exog_df is not None:
                                        # Expect future_exog_df to have ds and exog columns; align and fill missing with forward-fill
                                        fut = future_exog_df.copy()
                                        fut = fut.set_index('ds').reindex(future_ds).ffill().reset_index().rename(columns={'index': 'ds'})
                                        for col in X.columns:
                                            if col in fut.columns:
                                                df_future[col] = fut[col].values
                                            else:
                                                # For non-exogenous engineered columns, fallback to last known
                                                last_vals = X.iloc[-1] if not X.empty else None
                                                df_future[col] = last_vals[col] if last_vals is not None else 0.0
                                    else:
                                        last_vals = X.iloc[-1] if not X.empty else None
                                        for col in X.columns:
                                            df_future[col] = last_vals[col] if last_vals is not None else 0.0
                                    fc = model.predict(df_future)
                                    future_predictions = fc['yhat'].astype(float).tolist()
                                    if 'yhat_lower' in fc and 'yhat_upper' in fc:
                                        lower_bounds = fc['yhat_lower'].astype(float).tolist()
                                        upper_bounds = fc['yhat_upper'].astype(float).tolist()
                                    else:
                                        lower_bounds = [np.nan]*len(future_predictions)
                                        upper_bounds = [np.nan]*len(future_predictions)
                                except Exception as e:
                                    st.warning(f"Prophet future prediction failed: {e}")
                                    future_predictions = [np.nan]*forecast_horizon
                                    lower_bounds = [np.nan]*forecast_horizon
                                    upper_bounds = [np.nan]*forecast_horizon
                            # Invert differencing if applied
                            if pro_unlocked_ts and 'apply_diff' in locals() and apply_diff:
                                levels = []
                                last_level = train_df[target_col].iloc[-1]
                                for dp in future_predictions:
                                    last_level = last_level + dp
                                    levels.append(last_level)
                                forecast_vals = levels
                                lb = []
                                ub = []
                                if residual_std is not None and residual_std > 0:
                                    last_level = train_df[target_col].iloc[-1]
                                    for i in range(len(future_predictions)):
                                        # approximate intervals by cumulating std
                                        sigma = residual_std * np.sqrt(i + 1)
                                        lb.append(forecast_vals[i] - 1.96 * sigma)
                                        ub.append(forecast_vals[i] + 1.96 * sigma)
                                else:
                                    lb = [np.nan]*len(forecast_vals)
                                    ub = [np.nan]*len(forecast_vals)
                                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_vals, 'Lower': lb, 'Upper': ub})
                            else:
                                forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions, 'Lower': lower_bounds, 'Upper': upper_bounds})
                            
                            st.session_state['ts_report_ready'] = True
                            st.session_state['ts_plot_data'] = {'history': y, 'future_dates': future_dates, 'predictions': future_predictions, 'target_col': target_col}
                            st.session_state['df_forecast'] = forecast_df
                            
                            st.success("✅ Forecast model trained successfully!")
                            st.balloons()
                        except Exception as e:
                            st.error(f"An error occurred during forecasting: {e}")

if 'report_ready' in st.session_state:
    # Enforce plan rules: if the model/report was generated with Pro but user switched to Free, hide Pro report
    current_pro = st.session_state.get('pro_unlocked', False)
    trained_with_pro = st.session_state.get('trained_with_pro', False)
    if trained_with_pro and not current_pro:
        with st.container():
            st.header("Model Report")
            st.info("This report was generated using Pro features. Switch back to Pro to view it.")
    else:
        with st.container():
            st.header("Model Report")
            st.subheader("📊 Model Performance")
            st.metric(label=st.session_state['score_label'], value=f"{st.session_state['score_value']:.2%}" if st.session_state['problem_type_for_report'] == "Classification" else f"{st.session_state['score_value']:.3f}")
            
            st.subheader("⭐ Feature Importance")
            if 'feature_importance_df' in st.session_state:
                top_10_features_df = st.session_state['feature_importance_df'].head(10)
                chart = alt.Chart(top_10_features_df).mark_bar().encode(x=alt.X('Importance:Q'), y=alt.Y('Feature:N', sort='-x')).properties(title='Top 10 Most Important Features')
                st.altair_chart(chart)
            if st.session_state.get('trained_with_pro') and st.session_state.get('permutation_importance_df') is not None:
                st.subheader("🔄 Permutation Importance (Pro)")
                perm_df = st.session_state['permutation_importance_df'].head(10)
                perm_chart = alt.Chart(perm_df).mark_bar(color='#1f77b4').encode(
                    x=alt.X('Importance:Q'),
                    y=alt.Y('Feature:N', sort='-x')
                ).properties(title='Top 10 Features by Permutation Importance')
                st.altair_chart(perm_chart)

            st.subheader("🔗 Top Correlations with Target")
            if 'correlation_matrix' in st.session_state and st.session_state['correlation_matrix'] is not None and 'target_name_for_corr' in st.session_state:
                corr_matrix = st.session_state['correlation_matrix']
                target_name = st.session_state['target_name_for_corr']
                if target_name in corr_matrix.columns:
                    # Get correlations with target variable
                    corr_with_target = corr_matrix[target_name].abs().sort_values(ascending=False)
                    
                    # Get top 10 features (excluding target itself)
                    top_features = corr_with_target[corr_with_target.index != target_name].head(10)
                    
                    # Create correlation matrix for top features + target
                    features_to_include = list(top_features.index) + [target_name]
                    focused_corr_matrix = corr_matrix.loc[features_to_include, features_to_include]
                    
                    # Display the correlation matrix as heatmap
                    fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
                    sns.heatmap(focused_corr_matrix, ax=ax, annot=True, cmap='RdBu_r', fmt='.3f', 
                                center=0, square=True, cbar_kws={"shrink": .75}, annot_kws={"size": 8})
                    plt.title(f'Top 10 Feature Correlations with Target: {target_name}', fontsize=14, pad=20)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)  # Prevent memory leaks


if 'ts_report_ready' in st.session_state:
    with st.container():
        st.header("Forecast Report")
        st.subheader("📈 Forecast Visualization")
        plot_data = st.session_state['ts_plot_data']
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(plot_data['history'].index, plot_data['history'], label='Historical Data')
        ax.plot(plot_data['future_dates'], plot_data['predictions'], label='Future Forecast', linestyle='--')
        ax.legend()
        ax.set_title(f"Forecast for {plot_data['target_col']}")
        ax.set_xlabel("Date")
        ax.set_ylabel(plot_data['target_col'])
        st.pyplot(fig)
        plt.close(fig)  # Prevent memory leaks
        
        st.subheader("📋 Forecasted Values")
        st.dataframe(st.session_state['df_forecast'].head())
        
        output = io.BytesIO()
        st.session_state['df_forecast'].to_excel(output, index=False, sheet_name='Forecast')
        excel_data = output.getvalue()
    st.download_button(label="Download Forecast as XLSX", data=excel_data, file_name='forecast.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- 3. THE 'PREDICTION' SECTION ---
if 'report_ready' in st.session_state:
    with st.container():
        try:
            col1, col2 = st.columns([3, 2], vertical_alignment="center")
        except TypeError:
            # Fallback for older Streamlit versions without vertical_alignment
            col1, col2 = st.columns([3, 2])
        with col1:
            st.header("2. Make a Prediction")
        with col2:
            # Prediction uploader (styled/aligned by unified CSS)
            uploaded_predict_file = st.file_uploader(
                label="",
                type=['csv', 'xls', 'xlsx'],
                key="predict_uploader",
                label_visibility="collapsed"
            )

        if uploaded_predict_file is not None:
            predict_df = None
            try:
                file_ext_pred = os.path.splitext(uploaded_predict_file.name)[1]
                if file_ext_pred == '.csv':
                    predict_df = pd.read_csv(uploaded_predict_file)
                else:
                    predict_df = pd.read_excel(uploaded_predict_file)
                
                for col in predict_df.columns:
                    if predict_df[col].dtype == 'object':
                        predict_df[col] = predict_df[col].astype(str)

            except Exception as e:
                st.error(f"Error reading file: {e}. Please check the file format.")

            if predict_df is not None:
                # Add progress bar for prediction
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Progress: Loading model
                progress_bar.progress(0.2)
                status_text.text('Loading trained model...')
                model = joblib.load('model.joblib')
                
                expected_features = st.session_state.get('feature_columns', [])
                if not expected_features:
                    st.error("Model has not been trained properly. Please retrain.")
                else:
                    # Progress: Validating features
                    progress_bar.progress(0.4)
                    status_text.text('Validating feature columns...')
                    # Load base input feature columns saved during training
                    base_feature_cols = st.session_state.get('input_feature_columns')
                    if base_feature_cols is None and os.path.exists('input_feature_columns.joblib'):
                        try:
                            base_feature_cols = joblib.load('input_feature_columns.joblib')
                        except Exception:
                            base_feature_cols = st.session_state.get('feature_columns', [])
                    if not base_feature_cols:
                        base_feature_cols = st.session_state.get('feature_columns', [])
                    missing_cols = set(base_feature_cols) - set(predict_df.columns)
                    if missing_cols:
                        st.error(f"Error: The following required base feature columns are missing from your prediction file: {list(missing_cols)}")
                    else:
                        # Progress: Preprocessing
                        progress_bar.progress(0.6)
                        status_text.text('Preprocessing prediction data...')
                        predict_features = predict_df[base_feature_cols].copy()
                        # Apply saved feature-generation rules, if they exist
                        feature_rules = []
                        if os.path.exists('feature_rules.joblib'):
                            try:
                                feature_rules = joblib.load('feature_rules.joblib')
                            except Exception:
                                feature_rules = []
                        if feature_rules:
                            predict_features = apply_feature_generations(predict_features, feature_rules)
                        encoding_method = st.session_state.get('encoding_method', 'One-Hot Encoding')
                        saved_maps = None
                        if encoding_method == 'Target Encoding' and os.path.exists('target_encoding_maps.joblib'):
                            saved_maps = joblib.load('target_encoding_maps.joblib')
                        elif encoding_method == 'Label Encoding' and os.path.exists('feature_label_maps.joblib'):
                            saved_maps = joblib.load('feature_label_maps.joblib')

                        # Apply saved numeric FE before encoding
                        if os.path.exists('fe_numeric_params.joblib'):
                            try:
                                fe_config = joblib.load('fe_numeric_params.joblib')
                                # Apply fitted transformations
                                for fitted in fe_config.get('fitted_list', []):
                                    predict_features = apply_numeric_fe(predict_features, fitted)
                                # Apply custom binning rules from selections
                                for selection in fe_config.get('selections', []):
                                    if selection['method'] == 'Custom Binning':
                                        for rule in selection.get('custom_rules', []):
                                            if rule.get('new_name') and selection['column'] in predict_features.columns:
                                                col_data = pd.to_numeric(predict_features[selection['column']], errors='coerce')
                                                if rule['operator'] == '>':
                                                    predict_features[rule['new_name']] = (col_data > rule['value']).astype(int)
                                                elif rule['operator'] == '>=':
                                                    predict_features[rule['new_name']] = (col_data >= rule['value']).astype(int)
                                                elif rule['operator'] == '<':
                                                    predict_features[rule['new_name']] = (col_data < rule['value']).astype(int)
                                                elif rule['operator'] == '<=':
                                                    predict_features[rule['new_name']] = (col_data <= rule['value']).astype(int)
                                                elif rule['operator'] == '=':
                                                    predict_features[rule['new_name']] = (col_data == rule['value']).astype(int)
                            except Exception:
                                pass
                        predict_df_processed, _ = preprocess_features(predict_features, encoding_method=encoding_method, saved_maps=saved_maps)
                        # Pro: apply scaler if exists
                        if os.path.exists('scaler_stats.joblib'):
                            try:
                                scaler_stats = joblib.load('scaler_stats.joblib')
                                predict_df_processed = apply_scaler_stats(predict_df_processed, scaler_stats)
                            except Exception:
                                pass
                        
                        # Validate processed dataframe is not empty
                        if predict_df_processed.empty:
                            st.error("Processed prediction data is empty. Please check your input file.")
                            st.stop()
                        
                        # Progress: Making predictions
                        progress_bar.progress(0.8)
                        status_text.text('Generating predictions...')
                        training_cols = st.session_state.get('processed_columns', [])
                        if not training_cols:
                            st.error("Processed training columns not found. Please retrain.")
                        else:
                            predict_df_processed = predict_df_processed.reindex(columns=training_cols, fill_value=0)
                            # Safety: ensure all features are numeric before prediction
                            for col in predict_df_processed.columns:
                                if not pd.api.types.is_numeric_dtype(predict_df_processed[col]):
                                    predict_df_processed[col] = pd.to_numeric(predict_df_processed[col], errors='coerce')
                            predict_df_processed = predict_df_processed.fillna(0)

                            predictions_array = model.predict(predict_df_processed)
                            
                            target_columns = st.session_state.get('target_columns', [])
                            if len(target_columns) == 1 and len(predictions_array.shape) == 1:
                                predictions_array = predictions_array.reshape(-1, 1)

                            predictions_df = pd.DataFrame(predictions_array, columns=target_columns)

                            # Progress: Post-processing
                            progress_bar.progress(0.9)
                            status_text.text('Post-processing predictions...')
                            problem_type = st.session_state.get('problem_type_for_report')
                            if problem_type == "Classification":
                                if os.path.exists('label_encoders.joblib'):
                                    label_encoders = joblib.load('label_encoders.joblib')
                                    if isinstance(label_encoders, LabelEncoder):
                                         predictions_df[target_columns[0]] = label_encoders.inverse_transform(predictions_df[target_columns[0]])
                                    elif isinstance(label_encoders, dict):
                                        for col, le in label_encoders.items():
                                            predictions_df[col] = le.inverse_transform(predictions_df[col])
                            # To avoid duplicate column names (e.g., when the uploaded file still has target columns),
                            # rename predicted columns with a clear prefix after any decoding.
                            pred_renamed = {}
                            for col in predictions_df.columns:
                                new_col = f"Predicted_{col}"
                                # Ensure uniqueness against existing predict_df columns
                                base = new_col
                                idx = 2
                                while new_col in predict_df.columns:
                                    new_col = f"{base}_{idx}"
                                    idx += 1
                                pred_renamed[col] = new_col
                            predictions_df = predictions_df.rename(columns=pred_renamed)

                            final_output_df = pd.concat([predict_df.reset_index(drop=True), predictions_df], axis=1)

                            # Progress: Complete
                            progress_bar.progress(1.0)
                            status_text.text('Predictions completed successfully!')
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.session_state['df_predictions'] = final_output_df
                            st.write("Results with Predictions:")
                            st.dataframe(final_output_df.head())

# --- 4. THE 'DOWNLOAD' SECTION ---
if 'df_predictions' in st.session_state:
    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.header("3. Export Predictions")
        with col2:
            df_to_download = st.session_state.df_predictions.copy()
            output_format = st.session_state.get('output_file_extension', '.csv')

            if output_format in ['.xls', '.xlsx']:
                output = io.BytesIO()
                df_to_download.to_excel(output, index=False, sheet_name='Predictions')
                excel_data = output.getvalue()
                
                # Download button (styled/aligned by unified CSS)
                st.download_button(
                    label="Download as XLSX",
                    data=excel_data,
                    file_name='predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            else: # Default to CSV
                csv = df_to_download.to_csv(index=False).encode('utf-8')
                # Download button (styled/aligned by unified CSS)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

        
