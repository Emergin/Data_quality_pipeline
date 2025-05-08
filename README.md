# Mini Data Quality Pipeline

A simple yet powerful tool for data quality assessment, visualization, and cleaning. This project includes both a Python module for programmatic use and a Streamlit web application for interactive data analysis.

## Features

- **Data Ingestion**: Load data from various formats (CSV, JSON, Excel, Parquet)
- **Quality Assessment**: Check for missing values, duplicates, outliers, and data type issues
- **Visual Reporting**: Generate charts and visualizations of data quality issues
- **Data Cleaning**: Options for handling missing values, duplicates, and outliers
- **Interactive UI**: Web-based interface for exploring and cleaning data

## Project Structure

```
.
├── data_quality_pipeline.py     # Core functionality as Python module
├── streamlit_app.py             # Web application
├── README.md                    # This file
└── output/                      # Default output directory for reports and cleaned data
```

## Installation

1. Clone the repository or download the files
2. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn streamlit
```

## Usage

### Command-line Usage

You can use the data quality pipeline directly from the command line:

```bash
python data_quality_pipeline.py your_data_file.csv --report --clean
```

Options:
- `--report`, `-r`: Generate visual reports
- `--clean`, `-c`: Clean the data using default strategies
- `--output`, `-o`: Specify output path for cleaned data (default: `output/cleaned_data.csv`)

### Programmatic Usage

```python
from data_quality_pipeline import DataQualityPipeline

# Initialize and load data
pipeline = DataQualityPipeline("your_data_file.csv")
pipeline.load_data()

# Check quality
quality_report = pipeline.check_quality()
pipeline.print_text_report()

# Generate visual reports
pipeline.generate_visual_report()

# Clean data
cleaned_df = pipeline.clean_data(
    drop_cols=["unnecessary_column"],
    fill_missing={"age": "median", "name": ""},
    drop_duplicates=True,
    handle_outliers={"salary": "cap"}
)

# Compare before and after cleaning
pipeline.compare_before_after()

# Save cleaned data and report
pipeline.save_data("cleaned_data.csv")
pipeline.save_quality_report()
```

### Web Application

Run the Streamlit app for interactive data exploration and cleaning:

```bash
streamlit run streamlit_app.py
```

The web interface allows you to:
1. Upload your data file
2. Explore data quality issues with visual reports
3. Select cleaning options interactively
4. Download the cleaned data

## Data Cleaning Options

The pipeline offers several strategies for cleaning data:

### Handling Missing Values

- **Mean**: Replace missing values with the column mean (numeric only)
- **Median**: Replace missing values with the column median (numeric only)
- **Mode**: Replace missing values with the most common value
- **Custom value**: Replace missing values with a specified constant

### Handling Duplicates

- **Remove**: Delete duplicate rows

### Handling Outliers

- **Remove**: Delete rows containing outliers
- **Cap**: Replace outliers with the boundary values (1.5 × IQR method)

## Quality Metrics

The quality report includes:

- **Basic statistics**: Row and column counts
- **Missing values**: Counts and percentages by column
- **Duplicates**: Total count and percentage
- **Data types**: Type information for each column
- **Numeric columns**: Min, max, mean, median, standard deviation
- **Outliers**: Counts and percentages using IQR method
- **Categorical columns**: Unique value counts, top categories, empty strings

## Contributing

Contributions are welcome! Here are some ways you can contribute:
- Add more data cleaning strategies
- Improve visualizations
- Enhance performance for large datasets
- Add more quality checks
- Improve documentation

## License

MIT License