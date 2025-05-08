import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataQualityPipeline')

class DataQualityPipeline:
    """A simple data quality pipeline to ingest, check, report, and clean data."""
    
    def __init__(self, file_path: str = None):
        """Initialize the data quality pipeline.
        
        Args:
            file_path: Path to the data file
        """
        self.file_path = file_path
        self.df = None
        self.quality_report = {}
        self.original_df = None
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from various file formats.
        
        Args:
            file_path: Path to the data file. If None, uses the instance's file_path.
            
        Returns:
            pd.DataFrame: The loaded dataframe
        """
        if file_path is not None:
            self.file_path = file_path
            
        if self.file_path is None:
            raise ValueError("No file path provided")
            
        file_path = Path(self.file_path)
        logger.info(f"Loading data from {file_path}")
        
        # Handle different file formats
        if file_path.suffix.lower() == '.csv':
            self.df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            self.df = pd.read_json(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            self.df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.parquet':
            self.df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        # Store original data for comparison
        self.original_df = self.df.copy()
        
        logger.info(f"Loaded data with shape {self.df.shape}")
        return self.df
    
    def check_quality(self) -> Dict:
        """Check data quality and generate a report.
        
        Returns:
            Dict: Quality report with various metrics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Checking data quality...")
        
        # Basic statistics
        self.quality_report["row_count"] = len(self.df)
        self.quality_report["column_count"] = len(self.df.columns)
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percentages = (missing_values / len(self.df) * 100).round(2)
        self.quality_report["missing_values"] = dict(zip(self.df.columns, missing_values))
        self.quality_report["missing_percentages"] = dict(zip(self.df.columns, missing_percentages))
        
        # Duplicates
        duplicates = self.df.duplicated().sum()
        self.quality_report["duplicate_rows"] = duplicates
        self.quality_report["duplicate_percentage"] = round(duplicates / len(self.df) * 100, 2)
        
        # Data types
        self.quality_report["data_types"] = dict(self.df.dtypes.astype(str))
        
        # Summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        self.quality_report["numeric_columns"] = list(numeric_cols)
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "min": self.df[col].min(),
                "max": self.df[col].max(),
                "mean": self.df[col].mean(),
                "median": self.df[col].median(),
                "std": self.df[col].std()
            }
            
            # Detect outliers (using IQR method)
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            stats[col]["outliers"] = outlier_count
            stats[col]["outlier_percentage"] = round(outlier_count / len(self.df) * 100, 2)
            
        self.quality_report["numeric_stats"] = stats
        
        # Categorical analysis
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.quality_report["categorical_columns"] = list(cat_cols)
        
        cat_stats = {}
        for col in cat_cols:
            value_counts = self.df[col].value_counts()
            unique_count = len(value_counts)
            top_categories = value_counts.head(5).to_dict()
            
            cat_stats[col] = {
                "unique_values": unique_count,
                "top_categories": top_categories,
                "empty_strings": (self.df[col] == "").sum()
            }
            
        self.quality_report["categorical_stats"] = cat_stats
        
        logger.info("Quality check completed")
        return self.quality_report
    
    def generate_visual_report(self) -> None:
        """Generate visual reports of data quality issues."""
        if self.quality_report == {}:
            self.check_quality()
            
        logger.info("Generating visual reports...")
        
        # Set style
        sns.set(style="whitegrid")
        plt.rcParams.update({'figure.figsize': (12, 8)})
        
        # Missing values chart
        plt.figure(figsize=(12, 6))
        missing_df = pd.DataFrame({
            'Column': list(self.quality_report["missing_percentages"].keys()),
            'Missing %': list(self.quality_report["missing_percentages"].values())
        }).sort_values('Missing %', ascending=False)
        
        sns.barplot(x='Missing %', y='Column', data=missing_df)
        plt.title('Missing Values by Column (%)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_values.png')
        
        # Histograms for numeric columns
        numeric_cols = self.quality_report["numeric_columns"]
        for i, col in enumerate(numeric_cols[:min(6, len(numeric_cols))]):  # First 6 columns max
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(self.output_dir / f'distribution_{col}.png')
            
        # Boxplots for outlier detection
        plt.figure(figsize=(14, 8))
        if numeric_cols:
            # Select subset of numeric columns if there are many
            plot_cols = numeric_cols[:min(6, len(numeric_cols))]
            melted_df = self.df[plot_cols].melt()
            sns.boxplot(x='variable', y='value', data=melted_df)
            plt.title('Boxplots for Numeric Columns (Outlier Detection)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'outliers_boxplot.png')
            
        # Value counts for categorical columns
        cat_cols = self.quality_report["categorical_columns"]
        for i, col in enumerate(cat_cols[:min(6, len(cat_cols))]):  # First 6 columns max
            plt.figure(figsize=(12, 6))
            top_categories = pd.Series(self.quality_report["categorical_stats"][col]["top_categories"])
            sns.barplot(x=top_categories.index, y=top_categories.values)
            plt.title(f'Top values for {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'categories_{col}.png')
            
        # Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr = self.df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                      square=True, linewidths=.5, vmin=-1, vmax=1)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_heatmap.png')
        
        logger.info(f"Visual reports saved to {self.output_dir}")
        
    def print_text_report(self) -> None:
        """Print a text summary of the data quality report."""
        if self.quality_report == {}:
            self.check_quality()
            
        print("\n" + "="*50)
        print("DATA QUALITY REPORT")
        print("="*50)
        
        print(f"\nDataset Dimensions: {self.quality_report['row_count']} rows × {self.quality_report['column_count']} columns")
        print(f"Duplicate Rows: {self.quality_report['duplicate_rows']} ({self.quality_report['duplicate_percentage']}%)")
        
        print("\nMISSING VALUES:")
        print("-"*50)
        for col, missing_pct in sorted(self.quality_report["missing_percentages"].items(), 
                                    key=lambda x: x[1], reverse=True):
            if missing_pct > 0:
                print(f"{col}: {self.quality_report['missing_values'][col]} values ({missing_pct}%)")
        
        print("\nOUTLIERS:")
        print("-"*50)
        for col, stats in self.quality_report["numeric_stats"].items():
            if stats["outliers"] > 0:
                print(f"{col}: {stats['outliers']} outliers ({stats['outlier_percentage']}%)")
        
        print("\nCATEGORICAL COLUMNS:")
        print("-"*50)
        for col, stats in self.quality_report["categorical_stats"].items():
            print(f"{col}: {stats['unique_values']} unique values")
            if stats['empty_strings'] > 0:
                print(f"  - Empty strings: {stats['empty_strings']}")
        
        print("\nRECOMMENDATIONS:")
        print("-"*50)
        
        # Missing values recommendations
        high_missing = [col for col, pct in self.quality_report["missing_percentages"].items() if pct > 50]
        medium_missing = [col for col, pct in self.quality_report["missing_percentages"].items() 
                        if 20 < pct <= 50]
        
        if high_missing:
            print(f"- Consider dropping columns with >50% missing: {', '.join(high_missing)}")
        if medium_missing:
            print(f"- Consider imputation for columns with 20-50% missing: {', '.join(medium_missing)}")
            
        # Outlier recommendations
        high_outliers = [col for col, stats in self.quality_report["numeric_stats"].items() 
                        if stats["outlier_percentage"] > 5]
        if high_outliers:
            print(f"- Investigate outliers in: {', '.join(high_outliers)}")
            
        # Duplicate recommendations
        if self.quality_report["duplicate_percentage"] > 0:
            print("- Remove duplicate rows")
            
        print("="*50)
        
    def clean_data(self, 
                  drop_cols: List[str] = None, 
                  fill_missing: Dict[str, str] = None,
                  drop_duplicates: bool = False,
                  handle_outliers: Dict[str, str] = None) -> pd.DataFrame:
        """Clean the data based on specified operations.
        
        Args:
            drop_cols: List of columns to drop
            fill_missing: Dict mapping column names to imputation strategy ('mean', 'median', 'mode', or a value)
            drop_duplicates: Whether to drop duplicate rows
            handle_outliers: Dict mapping column names to outlier strategy ('remove', 'cap')
            
        Returns:
            pd.DataFrame: The cleaned dataframe
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Cleaning data...")
        
        # Make a copy of the data
        cleaned_df = self.df.copy()
        
        # Drop columns
        if drop_cols:
            cleaned_df = cleaned_df.drop(columns=[col for col in drop_cols if col in cleaned_df.columns])
            logger.info(f"Dropped columns: {drop_cols}")
            
        # Handle missing values
        if fill_missing:
            for col, strategy in fill_missing.items():
                if col in cleaned_df.columns:
                    if strategy == 'mean':
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                    elif strategy == 'median':
                        if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    elif strategy == 'mode':
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                    else:
                        # Use the provided value
                        cleaned_df[col] = cleaned_df[col].fillna(strategy)
                        
            logger.info(f"Filled missing values in columns: {list(fill_missing.keys())}")
                    
        # Drop duplicates
        if drop_duplicates:
            original_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            dropped_rows = original_rows - len(cleaned_df)
            logger.info(f"Dropped {dropped_rows} duplicate rows")
            
        # Handle outliers
        if handle_outliers:
            for col, strategy in handle_outliers.items():
                if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # Calculate boundaries
                    Q1 = cleaned_df[col].quantile(0.25)
                    Q3 = cleaned_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if strategy == 'remove':
                        # Remove rows with outliers
                        outlier_mask = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                        original_rows = len(cleaned_df)
                        cleaned_df = cleaned_df[~outlier_mask]
                        removed_rows = original_rows - len(cleaned_df)
                        logger.info(f"Removed {removed_rows} rows with outliers in column {col}")
                        
                    elif strategy == 'cap':
                        # Cap outliers at boundaries
                        original_outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
                        cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                        logger.info(f"Capped {original_outliers} outliers in column {col}")
        
        # Store the cleaned data
        self.df = cleaned_df
        
        return cleaned_df
    
    def save_data(self, output_path: str) -> None:
        """Save the cleaned data to a file.
        
        Args:
            output_path: Path where to save the data
        """
        if self.df is None:
            raise ValueError("No data to save")
            
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save based on file extension
        if output_path.suffix.lower() == '.csv':
            self.df.to_csv(output_path, index=False)
        elif output_path.suffix.lower() == '.json':
            self.df.to_json(output_path, orient='records')
        elif output_path.suffix.lower() in ['.xlsx', '.xls']:
            self.df.to_excel(output_path, index=False)
        elif output_path.suffix.lower() == '.parquet':
            self.df.to_parquet(output_path, index=False)
        else:
            # Default to CSV
            if not output_path.suffix:
                output_path = output_path.with_suffix('.csv')
            self.df.to_csv(output_path, index=False)
            
        logger.info(f"Saved cleaned data to {output_path}")
    
    def save_quality_report(self, output_path: str = None) -> None:
        """Save the quality report as JSON.
        
        Args:
            output_path: Path where to save the report. Default is 'output/quality_report.json'
        """
        if self.quality_report == {}:
            self.check_quality()
            
        if output_path is None:
            output_path = self.output_dir / 'quality_report.json'
        else:
            output_path = Path(output_path)
            
        # Convert numpy types to Python types for JSON serialization
        def convert_np(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                              np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (dict)):
                return {key: convert_np(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_np(x) for x in obj]
            else:
                return obj
        
        # Convert the report
        serializable_report = convert_np(self.quality_report)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save the report
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
            
        logger.info(f"Saved quality report to {output_path}")
    
    def compare_before_after(self) -> Dict:
        """Compare the original and cleaned datasets.
        
        Returns:
            Dict: Comparison metrics
        """
        if self.original_df is None or self.df is None:
            raise ValueError("Both original and cleaned dataframes must exist")
            
        comparison = {
            "original_shape": self.original_df.shape,
            "cleaned_shape": self.df.shape,
            "rows_removed": self.original_df.shape[0] - self.df.shape[0],
            "columns_removed": self.original_df.shape[1] - self.df.shape[1],
        }
        
        # Columns removed
        comparison["removed_columns"] = list(set(self.original_df.columns) - set(self.df.columns))
        
        # Missing values before and after
        orig_missing = self.original_df.isnull().sum().sum()
        clean_missing = self.df.isnull().sum().sum()
        comparison["original_missing_values"] = int(orig_missing)
        comparison["cleaned_missing_values"] = int(clean_missing)
        comparison["missing_values_reduction"] = int(orig_missing - clean_missing)
        
        # Print comparison
        print("\n" + "="*50)
        print("BEFORE vs AFTER CLEANING")
        print("="*50)
        print(f"Original shape: {comparison['original_shape'][0]} rows × {comparison['original_shape'][1]} columns")
        print(f"Cleaned shape: {comparison['cleaned_shape'][0]} rows × {comparison['cleaned_shape'][1]} columns")
        print(f"Rows removed: {comparison['rows_removed']}")
        print(f"Columns removed: {comparison['columns_removed']} {comparison['removed_columns']}")
        print(f"Missing values: {comparison['original_missing_values']} → {comparison['cleaned_missing_values']} (reduction: {comparison['missing_values_reduction']})")
        print("="*50)
        
        return comparison


def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(description='Data Quality Pipeline')
    parser.add_argument('file_path', help='Path to the data file')
    parser.add_argument('--output', '-o', help='Output path for cleaned data', default='output/cleaned_data.csv')
    parser.add_argument('--report', '-r', help='Generate visual report', action='store_true')
    parser.add_argument('--clean', '-c', help='Clean the data', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = DataQualityPipeline(args.file_path)
    pipeline.load_data()
    pipeline.check_quality()
    pipeline.print_text_report()
    
    if args.report:
        pipeline.generate_visual_report()
        pipeline.save_quality_report()
    
    if args.clean:
        # You can customize this based on the actual data cleaning needs
        pipeline.clean_data(
            drop_cols=[],  # Add columns to drop
            fill_missing={'all': 'mean'},  # Default strategy
            drop_duplicates=True,
            handle_outliers={}  # Add columns and strategies
        )
        pipeline.save_data(args.output)
        pipeline.compare_before_after()
    
    return 0


if __name__ == "__main__":
    main()