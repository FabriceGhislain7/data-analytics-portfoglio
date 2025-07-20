import streamlit as st
import pandas as pd
import numpy as np
import sys
from config import *
from utils.data_loader import (
    load_data, get_dataset_info, get_basic_stats, 
    check_missing_values, check_duplicates, check_data_quality
)
from utils.visualizations import (
    plot_missing_values_heatmap, plot_missing_values_bar,
    plot_data_types_distribution, plot_survival_overview
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(**PAGE_CONFIG)

# =============================================================================
# 1. PROJECT OVERVIEW
# =============================================================================

def show_project_overview():
    """Display project overview section with dataset description and objectives."""
    
    st.title(PROJECT_INFO["title"])
    st.markdown(PROJECT_INFO["description"])
    st.markdown("---")
    
    # Dataset Description
    st.header("1. Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset Description")
        st.markdown("""
        **The data has been split into two groups:**
        
        - **Training set (train.csv)** - Used to build machine learning models with known outcomes
        - **Test set (test.csv)** - Used to evaluate model performance on unseen data
        
        The training set provides the **ground truth** for each passenger. Models are based on features 
        like passengers' gender and class, with feature engineering to create new features.
        """)
    
    with col2:
        st.info("""
        **Quick Facts:**
        - ML Competition Format
        - Binary Classification  
        - Passenger Demographics
        - Historical Disaster Data
        """)
    
    # Variable Definitions
    st.subheader("Variable Definitions")
    
    col_vars1, col_vars2 = st.columns(2)
    
    with col_vars1:
        with st.expander("Core Variables"):
            for var, definition in list(VARIABLE_DEFINITIONS.items())[:6]:
                st.write(f"**{var}**: {definition}")
    
    with col_vars2:
        with st.expander("Additional Variables"):
            for var, definition in list(VARIABLE_DEFINITIONS.items())[6:]:
                st.write(f"**{var}**: {definition}")
    
    # Variable Notes
    with st.expander("Important Variable Notes"):
        for var, note in VARIABLE_NOTES.items():
            st.markdown(f"**{var}**:")
            st.markdown(note)
    
    # Key Findings Preview
    with st.expander("Key Findings from Analysis"):
        st.subheader("Main Insights")
        for finding in PROJECT_INFO["key_findings"]:
            st.markdown(f"• {finding}")
        
        st.subheader("Analysis Techniques Used")
        for technique in PROJECT_INFO["techniques"]:
            st.markdown(f"• {technique}")

# =============================================================================
# 2. DATA COLLECTION AND UNDERSTANDING
# =============================================================================

def show_data_understanding():
    """Display data collection and understanding section."""
    
    st.header("2. Data Collection and Understanding")
    
    # Load data
    with st.spinner("Loading dataset..."):
        data = load_data()
    
    if data is not None:
        st.success(MESSAGES["data_loaded"])
        
        # 2.1 Dataset Structure
        show_dataset_structure(data)
        
        # 2.2 Missing Values Analysis
        show_missing_values_analysis(data)
        
        # 2.3 Duplicate Analysis
        show_duplicate_analysis(data)
        
    else:
        st.error(MESSAGES["no_data"])

def show_dataset_structure(data):
    """Display dataset structure and basic information."""
    
    st.subheader("2.1. Structure of Dataset")
    
    # Library versions
    with st.expander("Environment Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.code(f"Pandas: {pd.__version__}")
        with col2:
            st.code(f"NumPy: {np.__version__}")
        with col3:
            st.code(f"Python: {sys.version.split()[0]}")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**First 5 rows:**")
        st.dataframe(data.head(), use_container_width=True)
    
    with col2:
        st.markdown("**Last 5 rows:**")
        st.dataframe(data.tail(), use_container_width=True)
    
    # Dataset dimensions and basic info
    info = get_dataset_info(data)
    basic_stats = get_basic_stats(data)
    
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Passengers", f"{info['rows']:,}")
    with col2:
        st.metric("Features", info['columns'])
    with col3:
        st.metric("Survival Rate", f"{info['survival_rate']:.1f}%")
    with col4:
        st.metric("Missing Values", info['missing_values'])
    
    st.info(f"The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
    
    # Column information
    st.subheader("Column Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Column Names:**")
        column_df = pd.DataFrame({
            'Column': data.columns,
            'Index': range(len(data.columns))
        })
        st.dataframe(column_df, use_container_width=True)
    
    with col2:
        st.markdown("**Data Types:**")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Detailed dataset summary
    st.subheader("Detailed Dataset Summary")
    
    summary_data = []
    for col in data.columns:
        summary_data.append({
            'Column': col,
            'Non-Null Count': data[col].count(),
            'Null Count': data[col].isnull().sum(),
            'Data Type': str(data[col].dtype),
            'Unique Values': data[col].nunique(),
            'Null Percentage': f"{(data[col].isnull().sum() / len(data)) * 100:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Data types visualization
    with st.expander("Data Types Distribution"):
        fig_types = plot_data_types_distribution(data)
        st.pyplot(fig_types)
    
    # Memory usage
    with st.expander("Memory Usage Analysis"):
        memory_df = pd.DataFrame({
            'Column': ['Index'] + list(data.columns),
            'Memory Usage (bytes)': data.memory_usage(deep=True).values,
        })
        memory_df['Memory Usage (KB)'] = memory_df['Memory Usage (bytes)'] / 1024
        memory_df['Memory Usage (MB)'] = memory_df['Memory Usage (KB)'] / 1024
        st.dataframe(memory_df, use_container_width=True)
        
        total_memory = memory_df['Memory Usage (MB)'].sum()
        st.info(f"Total memory usage: {total_memory:.2f} MB")

def show_missing_values_analysis(data):
    """Display missing values analysis."""
    
    st.subheader("2.2. Missing Values Analysis")
    
    # Calculate missing values
    missing_info = check_missing_values(data)
    
    if missing_info['total_missing'] > 0:
        # Missing values summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values Summary:**")
            st.dataframe(missing_info['summary'], use_container_width=True)
        
        with col2:
            st.markdown("**Missing Values by Column:**")
            missing_df = pd.DataFrame({
                'Column': missing_info['count'].index,
                'Missing Count': missing_info['count'].values,
                'Missing Percentage': missing_info['percentage'].values.round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            st.dataframe(missing_df, use_container_width=True)
        
        # Total missing info
        st.info(f"Total missing values in dataset: {missing_info['total_missing']}")
        
        # Visualizations
        st.subheader("Missing Values Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Values Heatmap:**")
            fig_heatmap = plot_missing_values_heatmap(data)
            st.pyplot(fig_heatmap)
        
        with col2:
            st.markdown("**Missing Values Percentage:**")
            fig_bar = plot_missing_values_bar(missing_info['percentage'])
            st.pyplot(fig_bar)
        
        # Missing values interpretation
        with st.expander("Missing Values Interpretation"):
            st.markdown("""
            **Key Observations:**
            
            • **Age**: 19.9% missing values (177 out of 891) - Significant amount requiring imputation
            • **Cabin**: 77.1% missing values (687 out of 891) - Consider dropping due to high missingness
            • **Embarked**: 0.2% missing values (2 out of 891) - Minimal impact, easily handled
            • **Other columns**: Complete data available
            
            **Recommendations:**
            1. **Drop Cabin column** - Too many missing values for reliable analysis
            2. **Impute Age** - Use mean, median, or advanced imputation techniques
            3. **Fill Embarked** - Use mode (most frequent port) for the 2 missing values
            """)
    else:
        st.success("No missing values found in the dataset!")

def show_duplicate_analysis(data):
    """Display duplicate rows analysis."""
    
    st.subheader("2.3. Duplicate Rows Analysis")
    
    # Check duplicates
    duplicate_info = check_duplicates(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(data))
    with col2:
        st.metric("Duplicate Rows", duplicate_info['count'])
    with col3:
        st.metric("Unique Rows", len(data) - duplicate_info['count'])
    
    if duplicate_info['count'] > 0:
        st.warning(f"Found {duplicate_info['count']} duplicate rows")
        
        # Show duplicate rows
        st.subheader("Duplicate Rows Details")
        st.dataframe(duplicate_info['duplicates'], use_container_width=True)
        
        # Show duplicate indices
        st.markdown("**Duplicate Row Indices:**")
        st.write(duplicate_info['indices'].tolist())
        
    else:
        st.success("No duplicate rows found in the dataset")
    
    # Additional data quality checks
    with st.expander("Additional Data Quality Checks"):
        # Check for columns with all null values
        all_null_cols = data.columns[data.isnull().all()]
        
        if len(all_null_cols) > 0:
            st.warning(f"Columns with all null values: {list(all_null_cols)}")
        else:
            st.success("No columns with all null values")
        
        # Check for constant columns
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            st.warning(f"Constant columns (single unique value): {constant_cols}")
        else:
            st.success("No constant columns found")
        
        # Check for high cardinality columns
        high_cardinality_cols = []
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() > len(data) * 0.8:  # More than 80% unique values
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            st.info(f"High cardinality columns (>80% unique): {high_cardinality_cols}")
        
        # Basic survival overview
        st.subheader("Basic Survival Statistics")
        survival_fig = plot_survival_overview(data)
        st.pyplot(survival_fig)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Select Section:",
        [
            "1. Project Overview",
            "2. Data Collection & Understanding"
        ]
    )
    
    # Display selected section
    if page == "1. Project Overview":
        show_project_overview()
    elif page == "2. Data Collection & Understanding":
        show_data_understanding()
    
    # Footer with documentation links
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Documentation Links")
    
    with st.sidebar.expander("Pandas Documentation"):
        st.markdown(f"[General Documentation]({DOCUMENTATION_LINKS['pandas']['general']})")
        st.markdown(f"[DataFrame.info()]({DOCUMENTATION_LINKS['pandas']['describe']})")
        st.markdown(f"[Missing Values]({DOCUMENTATION_LINKS['pandas']['fillna']})")
    
    with st.sidebar.expander("Visualization Libraries"):
        st.markdown(f"[Matplotlib]({DOCUMENTATION_LINKS['matplotlib']['general']})")
        st.markdown(f"[Seaborn]({DOCUMENTATION_LINKS['seaborn']['general']})")
        st.markdown(f"[Seaborn Heatmap]({DOCUMENTATION_LINKS['seaborn']['heatmap']})")
    
    with st.sidebar.expander("Data Science Concepts"):
        st.markdown(f"[EDA Guide]({DOCUMENTATION_LINKS['data_science']['eda_guide']})")
        st.markdown(f"[Data Cleaning]({DOCUMENTATION_LINKS['data_science']['data_cleaning']})")
    
    # Footer info
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Next Steps**: 
    Future sections will include:
    - Data Cleaning & Preprocessing
    - Exploratory Data Analysis
    - Machine Learning Models
    """)

if __name__ == "__main__":
    main()