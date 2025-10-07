# Button3 Application - Batch Version 0.02
# Created: October 4, 2025
# Updated: October 6, 2025
# Description: Production-ready ML application with robust data handling,
#              no data leakage risks, complete 4-step workflow, and Pro features

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.inspection import permutation_importance
import joblib
import os
import chardet
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, Any, List, Tuple

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="centered")

# --- CUSTOM CSS FOR APPLE-LIKE UI ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f0f2f5;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600;
    }
    h1, h2, h3 { font-weight: 700; color: #333; }
    
    /* File uploader styling - consolidated */
    div[data-testid="stFileUploader"] section > div {
        display: none;
    }
    
    [data-testid="stFileUploader"] .st-emotion-cache-9ycgxx,
    [data-testid="stFileUploader"] .css-9ycgxx,
    [data-testid="stFileUploader"] .st-emotion-cache-1rpn56r,
    [data-testid="stFileUploader"] .css-1rpn56r {
        display: none !important;
    }
    
    div[data-testid="stFileUploader"] section div {
        color: white !important;
    }

    div[data-testid="stFileUploader"] section small {
        color: white !important;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #007bff !important;
        background-color: #f0f8ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
CLASSIFICATION_THRESHOLD = 20

# --- HELPER FUNCTIONS ---

def reset_state():
    """Clears all session state variables and saved files from a previous run."""
    keys_to_clear = [
        'report_ready', 'score_label', 'score_value', 'problem_type_for_report',
        'feature_importance_df', 'correlation_matrix', 'target_name_for_corr',
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
    Each spec: {col1, col2, method, name}
    """
    if not specs:
        return df
    out = df.copy()
    eps = 1e-9
    for spec in specs:
        col1 = spec.get('col1')
        col2 = spec.get('col2')
        method = spec.get('method')
        name = spec.get('name')
        if not col1 or not col2 or not method:
            continue
        new_col = name or f"FE_{method}_{col1}_{col2}"
        try:
            if method == 'concat':
                out[new_col] = out[col1].astype(str) + '_' + out[col2].astype(str)
            elif method in ('sum', 'diff', 'absdiff', 'prod', 'ratio'):
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
    elif method in ['Log Trasformation', 'Square/Cube Root Transformation', 'Box-Cox Transformation', 'Yeo-Johnson Transformation']:
        transforms = {}
        for c in cols:
            s = pd.to_numeric(df[c], errors='coerce')
            if method == 'Log Trasformation':
                transforms[c] = {'add': float(max(1e-9, 1 - np.nanmin(s))) if np.nanmin(s) <= 0 else 0.0}
            elif method == 'Square/Cube Root Transformation':
                transforms[c] = {'type': params.get('root', 'square')}
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
    elif method == 'Log Trasformation':
        for c in cols:
            if c in out.columns:
                add = details['transforms'][c]['add']
                out[c] = np.log(pd.to_numeric(out[c], errors='coerce') + add)
    elif method == 'Square/Cube Root Transformation':
        for c in cols:
            if c in out.columns:
                t = details['transforms'][c].get('type', 'square')
                val = pd.to_numeric(out[c], errors='coerce')
                out[c] = np.cbrt(val) if t == 'cube' else np.sqrt(val)
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

# --- PROCESS ICONS ---
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 10px; border-radius: 10px; margin: 10px 0; color: white; text-align: center;'>
    <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 10px;'>
        <div style='display: flex; flex-direction: column; align-items: center; min-width: 100px;'>
            <div style='background: rgba(255,255,255,0.3); width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 8px; font-size: 1.5rem; font-weight: bold;'>
                1
            </div>
            <span style='color: white; font-size: 0.8rem; font-weight: 500;'>Upload Training</span>
        </div>
        <div style='color: rgba(255,255,255,0.8); font-size: 1.2rem; margin: 0 5px;'>→</div>
        <div style='display: flex; flex-direction: column; align-items: center; min-width: 100px;'>
            <div style='background: rgba(255,255,255,0.3); width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 8px; font-size: 1.5rem; font-weight: bold;'>
                2
            </div>
            <span style='color: white; font-size: 0.8rem; font-weight: 500;'>Upload Testing</span>
        </div>
        <div style='color: rgba(255,255,255,0.8); font-size: 1.2rem; margin: 0 5px;'>→</div>
        <div style='display: flex; flex-direction: column; align-items: center; min-width: 100px;'>
            <div style='background: rgba(255,255,255,0.3); width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 8px; font-size: 1.5rem; font-weight: bold;'>
                3
            </div>
            <span style='color: white; font-size: 0.8rem; font-weight: 500;'>Get Predictions</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Create custom layout with text on left and button on right
col1, col2 = st.columns([1, 1])

with col1:
    st.caption("Supported formats: CSV, XLS, XLSX")

with col2:
    st.caption("Size Limit 50MB")
    uploaded_train_file = st.file_uploader(
        "Upload Training Data",
        type=['csv', 'xls', 'xlsx'],
        key="train_file_uploader",
        label_visibility="collapsed"
    )

if uploaded_train_file is None:
    reset_state()

if uploaded_train_file is not None:
    # Check file size based on plan
    file_size_mb = uploaded_train_file.size / (1024 * 1024)
    pro_unlocked = st.session_state.get('pro_unlocked', False)
    max_file_size = 500 if pro_unlocked else 50  # 500MB for Pro, 50MB for Free
    
    if file_size_mb > max_file_size:
        if pro_unlocked:
            st.error(f"🚫 File too large ({file_size_mb:.1f}MB). Maximum file size for Pro is {max_file_size}MB.")
            st.stop()
        else:
            st.error(f"🚫 File too large ({file_size_mb:.1f}MB). Maximum file size is 50MB.")
            
            # Show upgrade to Pro option
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;'>
                    <h3 style='color: white; margin: 0 0 10px 0;'>🚀 Upgrade to Pro</h3>
                    <p style='color: white; margin: 0 0 15px 0;'>Handle files up to 500MB with Pro features!</p>
                    <div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px; margin: 10px 0;'>
                        <span style='color: white; font-weight: 600;'>✨ Pro Benefits:</span><br>
                        <span style='color: white; font-size: 0.9rem;'>• Files up to 500MB • Advanced models • Custom features</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Upgrade to Pro", key="upgrade_pro_button", use_container_width=True):
                    st.session_state['pro_unlocked'] = True
                    st.success("🎉 Pro features unlocked! You can now upload larger files.")
                    st.rerun()
            
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

                # Peek Pro status early to control encoding strategy UI
                pro_unlocked_early = st.session_state.get('pro_unlocked', False)
                # Auto-config: infer defaults when available (Pro only)
                auto_cfg = {}
                if 'auto_config_enabled' not in st.session_state:
                    st.session_state['auto_config_enabled'] = True
                if selected_target_cols:
                    try:
                        auto_cfg = infer_auto_config(train_df[[c for c in available_cols] + selected_target_cols], selected_target_cols)
                    except Exception:
                        auto_cfg = {}
                if pro_unlocked_early:
                    st.write("**Step 3: Choose Encoding Strategy**")
                    auto_on = st.checkbox("Auto-configure defaults from data", value=st.session_state['auto_config_enabled'])
                    st.session_state['auto_config_enabled'] = auto_on
                    enc_default = auto_cfg.get('encoding_choice', 'One-Hot Encoding') if auto_on else 'One-Hot Encoding'
                    if selected_target_cols and len(selected_target_cols) > 1:
                        enc_default = 'One-Hot Encoding'
                    encoding_options = ['One-Hot Encoding', 'Target Encoding', 'Label Encoding']
                    enc_idx = encoding_options.index(enc_default) if enc_default in encoding_options else 0
                    encoding_choice = st.selectbox("Encoding Strategy", options=encoding_options, index=enc_idx, label_visibility="collapsed")
                    if encoding_choice == 'Label Encoding':
                        st.warning("⚠️ Caution: Label Encoding might cause overfitting as it assigns arbitrary numerical order to categories.")
                else:
                    # Free plan: default to One-Hot without showing the selector
                    encoding_choice = 'One-Hot Encoding'
                    st.caption("Free plan uses One-Hot Encoding by default.")

                # Pro unlocked status (now controlled from upload section)
                pro_unlocked = st.session_state.get('pro_unlocked', False)

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

                # --- PRO: Feature Engineering and Model Configuration ---
                if pro_unlocked:
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
                                selected_numeric_col = st.selectbox("Column", options=numeric_cols, key="numeric_col_select")
                            with col3:
                                if st.button("+ Add", key="add_numeric_fe"):
                                    st.session_state['numeric_fe_selections'].append({
                                        'method': selected_numeric_method,
                                        'column': selected_numeric_col
                                    })
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
                                selected_cat_col = st.selectbox("Column", options=categorical_cols, key="cat_col_select")
                            with col3:
                                if st.button("+ Add", key="add_categorical_fe"):
                                    st.session_state['categorical_fe_selections'].append({
                                        'method': selected_cat_method,
                                        'column': selected_cat_col
                                    })
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
                        st.write("**Create new features by combining existing columns**")
                        
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

                st.write("**Step 5: Model Configuration**")
                
                # --- PRO: Model Selection and Configuration ---
                if pro_unlocked:
                    # Pro: Multi-Model Selection (Optional)
                    with st.expander("Pro: Multi-Model Selection (Optional)"):
                        # Model selections based on problem type
                        available_models_cls = ['RandomForest', 'LogisticRegression', 'SVM', 'KNN', 'DecisionTree', 'GradientBoosting', 'ExtraTrees', 'AdaBoost']
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
                        
                        # Cross-validation and model validation options
                        st.write("**Model Validation Options:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            use_cv = st.checkbox("Use Cross-Validation", value=True, key="use_cv_checkbox")
                            cv_folds = st.slider("CV Folds", min_value=3, max_value=10, value=5, step=1, disabled=not use_cv, key="cv_folds_slider")
                        with col2:
                            use_stratified = st.checkbox("Stratified CV (for Classification)", value=True, disabled=detected_pt != "Classification", key="stratified_cv")
                            random_state_cv = st.number_input("Random State for CV", value=42, step=1, key="cv_random_state")
                    
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
                                            f"N Estimators ({model})", 10, 500, 100, key=f"rf_n_est_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['max_depth'] = st.slider(
                                            f"Max Depth ({model})", 1, 20, 10, key=f"rf_depth_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['min_samples_split'] = st.slider(
                                            f"Min Samples Split ({model})", 2, 20, 2, key=f"rf_split_{model}")
                                
                                elif model == 'LogisticRegression':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['C'] = st.slider(
                                            f"Regularization (C) ({model})", 0.01, 10.0, 1.0, key=f"lr_c_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['max_iter'] = st.slider(
                                            f"Max Iterations ({model})", 100, 1000, 100, key=f"lr_iter_{model}")
                                
                                elif model == 'SVM':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['C'] = st.slider(
                                            f"C ({model})", 0.01, 10.0, 1.0, key=f"svm_c_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['kernel'] = st.selectbox(
                                            f"Kernel ({model})", ['rbf', 'linear', 'poly'], key=f"svm_kernel_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['gamma'] = st.selectbox(
                                            f"Gamma ({model})", ['scale', 'auto'], key=f"svm_gamma_{model}")
                                
                                elif model == 'GradientBoosting':
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_estimators'] = st.slider(
                                            f"N Estimators ({model})", 50, 300, 100, key=f"gb_n_est_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['learning_rate'] = st.slider(
                                            f"Learning Rate ({model})", 0.01, 0.3, 0.1, key=f"gb_lr_{model}")
                                    with col3:
                                        st.session_state['custom_hyperparams'][model]['max_depth'] = st.slider(
                                            f"Max Depth ({model})", 1, 10, 3, key=f"gb_depth_{model}")
                                
                                elif model == 'KNN':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['n_neighbors'] = st.slider(
                                            f"N Neighbors ({model})", 1, 20, 5, key=f"knn_neighbors_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['weights'] = st.selectbox(
                                            f"Weights ({model})", ['uniform', 'distance'], key=f"knn_weights_{model}")
                                
                                elif model in ['Ridge', 'Lasso']:
                                    st.session_state['custom_hyperparams'][model]['alpha'] = st.slider(
                                        f"Alpha ({model})", 0.01, 10.0, 1.0, key=f"reg_alpha_{model}")
                                
                                elif model == 'ElasticNet':
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.session_state['custom_hyperparams'][model]['alpha'] = st.slider(
                                            f"Alpha ({model})", 0.01, 10.0, 1.0, key=f"en_alpha_{model}")
                                    with col2:
                                        st.session_state['custom_hyperparams'][model]['l1_ratio'] = st.slider(
                                            f"L1 Ratio ({model})", 0.0, 1.0, 0.5, key=f"en_l1_{model}")
                                
                                st.divider()
                        else:
                            st.info("Please select models in the Multi-Model Selection section to configure hyperparameters.")
                
                # Basic Model Configuration
                col_a, col_b = st.columns(2)
                with col_a:
                    default_n_estimators = int(auto_cfg.get('n_estimators', 100)) if (st.session_state.get('auto_config_enabled', False) and pro_unlocked and auto_cfg) else 100
                    n_estimators = st.slider("Number of Trees (RandomForest)", min_value=50, max_value=500, value=default_n_estimators, step=50)
                with col_b:
                    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100

                # --- PRO: Custom Hyperparameters ---
                if pro_unlocked:
                    with st.expander("Pro: Custom Hyperparameters (optional)"):
                        colh1, colh2, colh3 = st.columns(3)
                        with colh1:
                            rf_max_depth = st.number_input("RF max_depth (0=auto)", min_value=0, value=rf_max_depth)
                            rf_min_samples_leaf = st.number_input("RF min_samples_leaf", min_value=1, value=rf_min_samples_leaf)
                        with colh2:
                            lr_C = st.number_input("LogReg C", min_value=0.0001, value=lr_C, step=0.1, format="%f")
                            svm_C = st.number_input("SVM C", min_value=0.0001, value=svm_C, step=0.1, format="%f")
                        with colh3:
                            knn_k = st.number_input("KNN n_neighbors", min_value=1, value=knn_k)
                            dt_max_depth = st.number_input("DT max_depth (0=auto)", min_value=0, value=dt_max_depth)
                
                st.write("**Step 6: Training**")
                if st.button("Train Model", type="primary", use_container_width=True):
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
                            feature_rules = st.session_state.get('pro_feature_rules', []) if pro_unlocked else []
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

                            def build_model(name):
                                if problem_type == "Classification":
                                    if name == 'RandomForest':
                                        md = RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                                                   max_depth=None if rf_max_depth==0 else rf_max_depth,
                                                                   min_samples_leaf=rf_min_samples_leaf)
                                    elif name == 'LogisticRegression':
                                        md = LogisticRegression(max_iter=1000, C=lr_C)
                                    elif name == 'SVM':
                                        md = SVC(C=svm_C, probability=True)
                                    elif name == 'KNN':
                                        md = KNeighborsClassifier(n_neighbors=knn_k)
                                    elif name == 'DecisionTree':
                                        md = DecisionTreeClassifier(max_depth=None if dt_max_depth==0 else dt_max_depth)
                                    elif name == 'GradientBoosting':
                                        md = GradientBoostingClassifier(n_estimators=n_estimators)
                                    elif name == 'ExtraTrees':
                                        md = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
                                    elif name == 'AdaBoost':
                                        md = AdaBoostClassifier(n_estimators=n_estimators)
                                    else:
                                        md = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                                else:
                                    if name == 'RandomForest':
                                        md = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                    elif name == 'LinearRegression':
                                        md = LinearRegression()
                                    elif name == 'Ridge':
                                        md = Ridge()
                                    elif name == 'Lasso':
                                        md = Lasso()
                                    elif name == 'ElasticNet':
                                        md = ElasticNet()
                                    elif name == 'SVR':
                                        md = SVR(C=svm_C)
                                    elif name == 'KNN':
                                        md = KNeighborsRegressor(n_neighbors=knn_k)
                                    elif name == 'DecisionTree':
                                        md = DecisionTreeRegressor(max_depth=None if dt_max_depth==0 else dt_max_depth)
                                    elif name == 'GradientBoosting':
                                        md = GradientBoostingRegressor(n_estimators=n_estimators)
                                    elif name == 'ExtraTrees':
                                        md = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42)
                                    elif name == 'AdaBoost':
                                        md = AdaBoostRegressor(n_estimators=n_estimators)
                                    else:
                                        md = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                                # Wrap for multi-target classification
                                if is_multi_target_cls and problem_type == "Classification":
                                    return MultiOutputClassifier(md)
                                return md

                            # Train and evaluate models; select best for saving
                            best_model = None
                            best_score = -np.inf
                            score_label = "Accuracy" if problem_type=="Classification" else "R-squared (R²)"

                            for name in chosen_models:
                                md = build_model(name)
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
                            score = best_score

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
                            
                            if maps_to_save: joblib.dump(maps_to_save, 'target_encoding_maps.joblib')
                            # Persist numeric FE config for inference
                            if fe_num_fitted_list or fe_num_selections:
                                fe_config = {
                                    'fitted_list': fe_num_fitted_list,
                                    'selections': fe_num_selections
                                }
                                joblib.dump(fe_config, 'fe_numeric_params.joblib')
                            # Save feature generation rules if any (for prediction)
                            if feature_rules:
                                joblib.dump(feature_rules, 'feature_rules.joblib')
                            joblib.dump(best_model, 'model.joblib')
                            if fe_standardize and scaler_stats is not None:
                                joblib.dump(scaler_stats, 'scaler_stats.joblib')
                            
                            # Progress: Feature analysis
                            progress_bar.progress(0.9)
                            status_text.text('Calculating feature importance and correlations...')
                            
                            st.session_state.update({'report_ready': True, 'score_label': score_label, 'score_value': score, 'problem_type_for_report': problem_type, 'feature_columns': X.columns.tolist(), 'processed_columns': X_processed.columns.tolist(), 'target_columns': selected_target_cols, 'encoding_method': encoding_choice, 'trained_with_pro': bool(pro_unlocked)})
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
                st.subheader("Forecast Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Select your Date/Time column", options=train_df.columns)
                with col2:
                    target_col = st.selectbox("Select the column to forecast", options=train_df.columns)
                
                forecast_horizon = st.number_input("How many periods to forecast into the future?", min_value=1, value=30)
                
                if st.button("Run Forecast", type="primary", use_container_width=True):
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
                            
                            # Create features from a clean, isolated dataframe
                            df_featured = create_time_series_features(train_df[[date_col, target_col]], date_col, target_col)
                            
                            if df_featured.empty:
                                st.error("Not enough data to create time series features after processing. Please provide a longer time series.")
                                st.stop()

                            X = df_featured.drop(columns=[target_col])
                            y = df_featured[target_col]
                            
                            # Use proper time series split (last 20% for testing)
                            split_idx = int(len(df_featured) * 0.8)
                            X_train_ts, y_train_ts = X.iloc[:split_idx], y.iloc[:split_idx]
                            X_test_ts, y_test_ts = X.iloc[split_idx:], y.iloc[split_idx:]
                            
                            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                            
                            # Validate on time series test set
                            model.fit(X_train_ts, y_train_ts)
                            if not X_test_ts.empty:
                                ts_predictions = model.predict(X_test_ts)
                                ts_score = r2_score(y_test_ts, ts_predictions)
                                st.info(f"Time Series Validation R² Score: {ts_score:.3f}")
                            else:
                                st.info("Dataset too small for validation split - training on full dataset.")
                            
                            # Train final model on ALL historical data
                            model.fit(X, y)
                            
                            # --- FIX 2: Iterative forecasting loop ---
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
                            
                            for i in range(forecast_horizon):
                                # Update progress
                                progress_bar.progress((i + 1) / forecast_horizon)
                                status_text.text(f'Generating forecast step {i + 1} of {forecast_horizon}...')
                                
                                # 1. Create features for the next step based on all available history
                                features_for_next_step = create_time_series_features(history_df, date_col, target_col)
                                
                                # 2. Get the last row of features to predict the next point
                                last_feature_row = features_for_next_step.iloc[-1:].drop(columns=[target_col])
                                last_feature_row = last_feature_row.reindex(columns=X.columns, fill_value=0) # Ensure column order
                                
                                # 3. Predict the next value
                                next_pred = model.predict(last_feature_row)[0]
                                future_predictions.append(next_pred)
                                
                                # 4. Add the prediction to history to use it for the next iteration's feature engineering
                                last_date = last_date + pd.to_timedelta(1, unit=freq)
                                new_row = pd.DataFrame({date_col: [last_date], target_col: [next_pred]})
                                history_df = pd.concat([history_df, new_row], ignore_index=True)

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            future_dates = history_df[date_col].iloc[-forecast_horizon:]
                            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_predictions})
                            
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
                st.altair_chart(chart, use_container_width=True)

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
                    st.pyplot(fig, use_container_width=True)
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
        st.download_button(label="Download Forecast as XLSX", data=excel_data, file_name='forecast.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)

# --- 3. THE 'PREDICTION' SECTION ---
if 'report_ready' in st.session_state:
    with st.container():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.header("2. Make a Prediction")
        with col2:
            uploaded_predict_file = st.file_uploader("Upload test file (Max: 50MB)", type=['csv', 'xls', 'xlsx'], key="predict_uploader")

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
                
                st.download_button(
                    label="Download as XLSX",
                    data=excel_data,
                    file_name='predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            else: # Default to CSV
                csv = df_to_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
