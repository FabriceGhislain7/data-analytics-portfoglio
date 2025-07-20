import pandas as pd
import numpy as np
import streamlit as st
import os
from config import TITANIC_CSV_PATH, TITANIC_URL

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """
    Load the Titanic dataset from local file or URL.
    
    Returns:
        pd.DataFrame: Loaded dataset or None if error
    """
    try:
        # Try loading from local file first
        if os.path.exists(TITANIC_CSV_PATH):
            data = pd.read_csv(TITANIC_CSV_PATH)
            st.success(f"Dataset loaded from local file: {TITANIC_CSV_PATH}")
        else:
            # Fallback to URL
            data = pd.read_csv(TITANIC_URL)
            st.info(f"Dataset loaded from URL: {TITANIC_URL}")
        
        return data
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def load_titanic_data():
    """
    Legacy function name for compatibility.
    """
    return load_data()

# =============================================================================
# DATASET INFORMATION FUNCTIONS
# =============================================================================

def get_dataset_info(df):
    """
    Get comprehensive dataset information.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Dataset information
    """
    info = {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'survivors': df['Survived'].sum(),
        'survival_rate': (df['Survived'].sum() / len(df)) * 100,
        'avg_age': df['Age'].mean(),
        'avg_fare': df['Fare'].mean(),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }
    return info

def get_basic_stats(df):
    """
    Get basic statistical information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Basic statistics
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict()
    }

def get_survival_stats(df, group_by_column):
    """
    Get survival statistics grouped by a specific column.
    
    Args:
        df (pd.DataFrame): Input dataset
        group_by_column (str): Column to group by
        
    Returns:
        pd.DataFrame: Survival statistics
    """
    stats = df.groupby(group_by_column).agg({
        'Survived': ['count', 'sum', 'mean']
    }).round(3)
    
    # Flatten column names
    stats.columns = ['Total_Count', 'Survived_Count', 'Survival_Rate']
    stats['Survival_Rate'] = stats['Survival_Rate'] * 100  # Convert to percentage
    stats = stats.reset_index()
    
    return stats

# =============================================================================
# MISSING VALUES ANALYSIS
# =============================================================================

def check_missing_values(df):
    """
    Analyze missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Missing values analysis
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_percentage
    })
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    return {
        'count': missing_count,
        'percentage': missing_percentage,
        'summary': missing_df,
        'total_missing': missing_count.sum()
    }

def check_duplicates(df):
    """
    Check for duplicate rows in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Duplicate analysis
    """
    duplicated_mask = df.duplicated()
    duplicate_count = duplicated_mask.sum()
    duplicate_rows = df[duplicated_mask]
    duplicate_indices = df[duplicated_mask].index
    
    return {
        'count': duplicate_count,
        'mask': duplicated_mask,
        'duplicates': duplicate_rows,
        'indices': duplicate_indices
    }

# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

def check_data_quality(df):
    """
    Comprehensive data quality assessment.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Data quality report
    """
    quality_report = {
        'basic_info': get_basic_stats(df),
        'missing_values': check_missing_values(df),
        'duplicates': check_duplicates(df),
        'data_types': analyze_data_types(df),
        'unique_values': get_unique_value_counts(df)
    }
    
    return quality_report

def analyze_data_types(df):
    """
    Analyze data types and suggest improvements.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Data type analysis
    """
    type_analysis = {}
    
    for col in df.columns:
        col_info = {
            'current_type': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'unique_count': df[col].nunique(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        
        # Suggest appropriate type
        if df[col].dtype == 'object':
            # Check if it could be categorical
            if df[col].nunique() < 10:
                col_info['suggested_type'] = 'category'
            else:
                col_info['suggested_type'] = 'string'
        else:
            col_info['suggested_type'] = str(df[col].dtype)
        
        type_analysis[col] = col_info
    
    return type_analysis

def get_unique_value_counts(df):
    """
    Get unique value counts for categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        dict: Unique value counts
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    unique_counts = {}
    
    for col in categorical_cols:
        if col != 'Name':  # Skip Name column as it's mostly unique
            unique_counts[col] = df[col].value_counts().to_dict()
    
    return unique_counts

# =============================================================================
# DATASET CLEANING FUNCTIONS
# =============================================================================

def apply_cleaning_method(df, method):
    """
    Apply specific cleaning method to the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        method (str): Cleaning method to apply
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df_cleaned = df.copy()
    
    if method == "method_1":
        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()
    
    elif method == "method_2":
        # Remove columns with >50% missing values
        threshold = len(df_cleaned) * 0.5
        df_cleaned = df_cleaned.dropna(axis=1, thresh=threshold)
    
    elif method == "method_3":
        # Remove rows with missing Age or Embarked
        df_cleaned = df_cleaned.dropna(subset=['Age', 'Embarked'])
    
    elif method == "method_4":
        # Impute missing Age with mean
        df_cleaned['Age'].fillna(df_cleaned['Age'].mean(), inplace=True)
        # Drop columns with too many missing values
        threshold = len(df_cleaned) * 0.5
        df_cleaned = df_cleaned.dropna(axis=1, thresh=threshold)
        # Fill missing Embarked with mode
        df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)
    
    return df_cleaned

def get_cleaning_comparison(original_df, cleaned_df, method_name):
    """
    Compare original and cleaned datasets.
    
    Args:
        original_df (pd.DataFrame): Original dataset
        cleaned_df (pd.DataFrame): Cleaned dataset
        method_name (str): Name of cleaning method applied
        
    Returns:
        dict: Comparison statistics
    """
    comparison = {
        'method': method_name,
        'original_shape': original_df.shape,
        'cleaned_shape': cleaned_df.shape,
        'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
        'columns_removed': original_df.shape[1] - cleaned_df.shape[1],
        'missing_values_before': original_df.isnull().sum().sum(),
        'missing_values_after': cleaned_df.isnull().sum().sum(),
        'data_retention': (cleaned_df.shape[0] / original_df.shape[0]) * 100
    }
    
    return comparison

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def create_family_size(df):
    """
    Create family size feature.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with family size feature
    """
    df_new = df.copy()
    df_new['Family_Size'] = df_new['SibSp'] + df_new['Parch'] + 1
    return df_new

def create_age_groups(df, bins=None, labels=None):
    """
    Create age group categories.
    
    Args:
        df (pd.DataFrame): Input dataset
        bins (list): Age bins for categorization
        labels (list): Labels for age groups
        
    Returns:
        pd.DataFrame: Dataset with age groups
    """
    from config import AGE_BINS, AGE_LABELS
    
    df_new = df.copy()
    
    if bins is None:
        bins = AGE_BINS
    if labels is None:
        labels = AGE_LABELS
    
    df_new['Age_Group'] = pd.cut(df_new['Age'], bins=bins, labels=labels, include_lowest=True)
    
    return df_new

def create_fare_categories(df):
    """
    Create fare categories using quantiles.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Dataset with fare categories
    """
    from config import FARE_QUANTILES, FARE_LABELS
    
    df_new = df.copy()
    df_new['Fare_Category'] = pd.qcut(df_new['Fare'], q=FARE_QUANTILES, labels=FARE_LABELS)
    
    return df_new