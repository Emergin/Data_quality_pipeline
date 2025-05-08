import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import tempfile
import io
import sys
import os

# Add the parent directory to the path to import DataQualityPipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_quality_pipeline import DataQualityPipeline

# Set page configuration
st.set_page_config(
    page_title="Data Quality Pipeline",
    page_icon="🧹",
    layout="wide"
)

# App title and description
st.title("📊 Data Quality Pipeline")
st.markdown("""
This app helps you analyze and clean your data by:
1. Checking for quality issues like missing values, duplicates, and outliers
2. Visualizing data quality problems
3. Cleaning the data based on your choices
""")

# Sidebar for file upload and options
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV, Excel, or JSON file", 
                                    type=["csv", "xlsx", "xls", "json"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Uploaded: {uploaded_file.name}")
        
        # Create pipeline instance
        pipeline = DataQualityPipeline(str(temp_path))
        
        # Load data
        try:
            df = pipeline.load_data()
            st.success(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
            
        # Run quality check
        pipeline.check_quality()
        
    st.divider()
    
    # Only show these options if data is loaded
    if 'pipeline' in locals():
        st.header("Data Cleaning Options")
        
        # Select columns to drop
        st.subheader("Columns to Drop")
        drop_cols = st.multiselect("Select columns to drop", df.columns)
        
        # Handle missing values
        st.subheader("Handle Missing Values")
        
        cols_with_missing = [col for col, count in pipeline.quality_report["missing_values"].items() if count > 0]
        
        if cols_with_missing:
            st.write("Select strategy for columns with missing values:")
            
            fill_missing = {}
            for col in cols_with_missing:
                col_type = pipeline.df[col].dtype
                
                options = ["Don't fill"]
                if np.issubdtype(col_type, np.number):
                    options.extend(["Mean", "Median", "Zero"])
                else:
                    options.extend(["Mode", "Empty string"])
                
                strategy = st.selectbox(f"Strategy for '{col}'", options, key=f"missing_{col}")
                
                if strategy != "Don't fill":
                    if strategy == "Mean":
                        fill_missing[col] = "mean"
                    elif strategy == "Median":
                        fill_missing[col] = "median"
                    elif strategy == "Mode":
                        fill_missing[col] = "mode"
                    elif strategy == "Zero":
                        fill_missing[col] = 0
                    elif strategy == "Empty string":
                        fill_missing[col] = ""
        else:
            st.info("No columns with missing values detected")
            fill_missing = {}
        
        # Handle duplicates
        st.subheader("Handle Duplicates")
        drop_duplicates = st.checkbox("Remove duplicate rows", 
                                     value=pipeline.quality_report["duplicate_rows"] > 0)
        
        # Handle outliers
        st.subheader("Handle Outliers")
        
        cols_with_outliers = [col for col, stats in pipeline.quality_report["numeric_stats"].items() 
                             if stats["outliers"] > 0]
        
        handle_outliers = {}
        if cols_with_outliers:
            st.write("Select strategy for columns with outliers:")
            
            for col in cols_with_outliers:
                strategy = st.selectbox(
                    f"Strategy for '{col}' ({pipeline.quality_report['numeric_stats'][col]['outlier_percentage']}% outliers)",
                    ["Don't handle", "Remove rows", "Cap at boundaries"],
                    key=f"outlier_{col}"
                )
                
                if strategy != "Don't handle":
                    if strategy == "Remove rows":
                        handle_outliers[col] = "remove"
                    elif strategy == "Cap at boundaries":
                        handle_outliers[col] = "cap"
        else:
            st.info("No columns with outliers detected")
        
        # Clean data button
        clean_button = st.button("Clean Data", type="primary")

# Main content
if 'pipeline' in locals():
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "🔍 Quality Issues", "🧹 Cleaning Results"])
    
    with tab1:
        st.header("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.header("Column Information")
        col_info = pd.DataFrame({
            "Type": df.dtypes,
            "Non-Null Count": df.count(),
            "Null Count": df.isnull().sum(),
            "% Missing": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
        
    with tab2:
        st.header("Data Quality Issues")
        
        # Display key quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_missing = sum(pipeline.quality_report["missing_values"].values())
            st.metric("Missing Values", f"{total_missing:,}", 
                     f"{total_missing/len(df)/len(df.columns)*100:.2f}% of all cells")
            
        with col2:
            duplicates = pipeline.quality_report["duplicate_rows"]
            st.metric("Duplicate Rows", f"{duplicates:,}",
                     f"{pipeline.quality_report['duplicate_percentage']}% of rows")
            
        with col3:
            total_outliers = sum(stats["outliers"] for stats in pipeline.quality_report["numeric_stats"].values())
            st.metric("Outliers", f"{total_outliers:,}",
                     f"In {len([col for col, stats in pipeline.quality_report['numeric_stats'].items() if stats['outliers'] > 0])} columns")
        
        # Missing values chart
        st.subheader("Missing Values by Column")
        if total_missing > 0:
            missing_df = pd.DataFrame({
                'Column': list(pipeline.quality_report["missing_percentages"].keys()),
                'Percentage': list(pipeline.quality_report["missing_percentages"].values())
            })
            missing_df = missing_df.sort_values('Percentage', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Percentage', y='Column', data=missing_df, ax=ax)
            ax.set_title('Missing Value Percentages by Column')
            ax.set_xlabel('Percentage Missing (%)')
            st.pyplot(fig)
        else:
            st.info("No missing values detected in the dataset")
        
        # Duplicate rows info
        st.subheader("Duplicate Rows")
        if pipeline.quality_report["duplicate_rows"] > 0:
            st.write(f"Found {pipeline.quality_report['duplicate_rows']} duplicate rows ({pipeline.quality_report['duplicate_percentage']}% of data)")
            
            # Show examples of duplicates
            if "duplicate_examples" in pipeline.quality_report:
                st.write("Examples of duplicate rows:")
                st.dataframe(pipeline.quality_report["duplicate_examples"], use_container_width=True)
        else:
            st.info("No duplicate rows detected in the dataset")
        
        # Outliers visualization
        st.subheader("Outlier Detection")
        
        if cols_with_outliers:
            st.write("Distribution of numeric columns with outliers:")
            
            for col in cols_with_outliers:
                stats = pipeline.quality_report["numeric_stats"][col]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Distribution of '{col}'")
                ax.axvline(stats["lower_bound"], color='r', linestyle='--', label=f'Lower bound: {stats["lower_bound"]:.2f}')
                ax.axvline(stats["upper_bound"], color='r', linestyle='--', label=f'Upper bound: {stats["upper_bound"]:.2f}')
                ax.legend()
                st.pyplot(fig)
                
                st.write(f"**{col}**: {stats['outliers']} outliers detected ({stats['outlier_percentage']}% of values)")
                st.write(f"- Range: [{stats['min']:.2f} - {stats['max']:.2f}]")
                st.write(f"- IQR boundaries: [{stats['lower_bound']:.2f} - {stats['upper_bound']:.2f}]")
        else:
            st.info("No significant outliers detected in the dataset")
            
    with tab3:
        st.header("Data Cleaning Results")
        
        if 'clean_button' in locals() and clean_button:
            # Apply cleaning operations based on selected options
            cleaned_df = df.copy()
            
            # Record cleaning operations
            cleaning_log = []
            
            # 1. Drop selected columns
            if drop_cols:
                cleaned_df = cleaned_df.drop(columns=drop_cols)
                cleaning_log.append(f"Dropped {len(drop_cols)} columns: {', '.join(drop_cols)}")
            
            # 2. Fill missing values
            for col, strategy in fill_missing.items():
                if col in cleaned_df.columns:  # Check if column wasn't dropped
                    if strategy == "mean":
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                        cleaning_log.append(f"Filled missing values in '{col}' with mean: {cleaned_df[col].mean():.2f}")
                    elif strategy == "median":
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                        cleaning_log.append(f"Filled missing values in '{col}' with median: {cleaned_df[col].median():.2f}")
                    elif strategy == "mode":
                        mode_value = cleaned_df[col].mode()[0]
                        cleaned_df[col] = cleaned_df[col].fillna(mode_value)
                        cleaning_log.append(f"Filled missing values in '{col}' with mode: {mode_value}")
                    else:  # Constant value (0 or "")
                        cleaned_df[col] = cleaned_df[col].fillna(strategy)
                        cleaning_log.append(f"Filled missing values in '{col}' with constant: '{strategy}'")
            
            # 3. Handle duplicates
            if drop_duplicates and pipeline.quality_report["duplicate_rows"] > 0:
                original_len = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
                removed = original_len - len(cleaned_df)
                cleaning_log.append(f"Removed {removed} duplicate rows")
                
            # 4. Handle outliers
            outlier_rows_removed = 0
            
            for col, strategy in handle_outliers.items():
                if col in cleaned_df.columns:  # Check if column wasn't dropped
                    stats = pipeline.quality_report["numeric_stats"][col]
                    lower_bound = stats["lower_bound"]
                    upper_bound = stats["upper_bound"]
                    
                    if strategy == "remove":
                        # Count outliers to be removed
                        outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                        col_outliers = outlier_mask.sum()
                        
                        # Remove rows with outliers
                        cleaned_df = cleaned_df[~outlier_mask].reset_index(drop=True)
                        outlier_rows_removed += col_outliers
                        cleaning_log.append(f"Removed {col_outliers} rows with outliers in '{col}'")
                        
                    elif strategy == "cap":
                        # Cap outliers at boundaries
                        num_lower = (cleaned_df[col] < lower_bound).sum()
                        num_upper = (cleaned_df[col] > upper_bound).sum()
                        
                        cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                        cleaning_log.append(f"Capped {num_lower} low outliers and {num_upper} high outliers in '{col}'")
            
            # Show cleaning operations and results
            st.subheader("Cleaning Operations")
            if cleaning_log:
                for operation in cleaning_log:
                    st.write(f"- {operation}")
            else:
                st.info("No cleaning operations were performed")
            
            # Show before/after comparison
            st.subheader("Before vs After Cleaning")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Data**")
                st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                st.write(f"Missing values: {df.isnull().sum().sum()}")
                st.dataframe(df.head(5), use_container_width=True)
                
            with col2:
                st.write("**Cleaned Data**")
                st.write(f"Rows: {len(cleaned_df)}, Columns: {len(cleaned_df.columns)}")
                st.write(f"Missing values: {cleaned_df.isnull().sum().sum()}")
                st.dataframe(cleaned_df.head(5), use_container_width=True)
            
            # Download clean data
            st.subheader("Download Cleaned Data")
            
            # Determine original file extension
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            # Create download buttons for different formats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = cleaned_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name=f"cleaned_{Path(uploaded_file.name).stem}.csv",
                    mime="text/csv"
                )
                
            with col2:
                # Create Excel file in memory
                excel_buffer = io.BytesIO()
                cleaned_df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                
                st.download_button(
                    label="Download as Excel",
                    data=excel_buffer,
                    file_name=f"cleaned_{Path(uploaded_file.name).stem}.xlsx",
                    mime="application/vnd.ms-excel"
                )
                
            with col3:
                # Convert to JSON
                json_data = cleaned_df.to_json(orient="records")
                
                st.download_button(
                    label="Download as JSON",
                    data=json_data,
                    file_name=f"cleaned_{Path(uploaded_file.name).stem}.json",
                    mime="application/json"
                )
                
        else:
            st.info("Click the 'Clean Data' button in the sidebar to apply cleaning operations")

else:
    # Show placeholder content when no data is loaded
    st.info("👈 Please upload a data file using the sidebar to get started")
    
    # Show sample visualization as placeholder
    st.subheader("How it works")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://raw.githubusercontent.com/streamlit/demo-self-driving/master/streamlit-badge-black-white.png", 
                 caption="Upload data to begin analysis")
        
        st.markdown("""
        ### The pipeline will:
        1. Analyze your data for quality issues
        2. Visualize problems found in the data
        3. Provide options to clean the data
        4. Allow you to download the cleaned dataset
        """)

# Footer
st.divider()
st.markdown("📊 **Data Quality Pipeline** | Built with Streamlit")