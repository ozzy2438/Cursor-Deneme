# 🚀 Auto Data Analysis Dashboard

A comprehensive end-to-end data analysis system that automatically performs data exploration, quality assessment, cleaning, and visualization with an interactive dashboard.

## ✨ Features

### 🔍 **Automatic Data Exploration**
- **Data Quality Assessment**: Identifies missing values, duplicates, outliers, and data type issues
- **Statistical Analysis**: Comprehensive descriptive statistics for numeric and categorical variables
- **Correlation Analysis**: Detects highly correlated features and multicollinearity
- **Target Variable Analysis**: Specialized analysis for supervised learning scenarios

### 🧹 **Intelligent Data Cleaning**
- **Missing Value Handling**: Multiple strategies (auto, drop, mean/median/mode imputation)
- **Duplicate Removal**: Automatic detection and removal of duplicate records
- **Outlier Treatment**: IQR-based outlier detection with capping or removal options
- **Categorical Encoding**: Smart encoding based on cardinality (one-hot vs label encoding)

### 📊 **Rich Visualizations**
- **Missing Value Patterns**: Heatmaps showing missing data distribution
- **Correlation Matrices**: Interactive heatmaps for feature relationships
- **Distribution Plots**: Histograms and density plots for all numeric variables
- **Box Plots**: Outlier detection and quartile visualization
- **Categorical Distributions**: Bar charts for categorical variable frequencies
- **Target Analysis**: Specialized plots for target variable exploration

### 💡 **Automated Insights**
- **Data Overview**: Summary of dataset characteristics and composition
- **Quality Issues**: Automatic detection and reporting of data quality problems
- **Statistical Insights**: Key findings from correlation and distribution analysis
- **Recommendations**: Actionable suggestions for data preprocessing and modeling

### 🎯 **Interactive Dashboard**
- **Streamlit-based Interface**: User-friendly web application
- **Real-time Analysis**: Interactive exploration with immediate feedback
- **Multiple Data Sources**: File upload or built-in sample datasets
- **Export Capabilities**: Download cleaned data and analysis reports
- **Customizable Options**: Configurable cleaning and analysis parameters

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cursor-deneme
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Dashboard

**Launch the interactive dashboard:**
```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the Python API

```python
import pandas as pd
from data_analyzer import AutoDataAnalyzer

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize the analyzer
analyzer = AutoDataAnalyzer(data, target_column='your_target')

# Perform analysis
quality_report = analyzer.analyze_data_quality()
eda_results = analyzer.perform_eda()
visualizations = analyzer.generate_visualizations()

# Clean the data
cleaned_data = analyzer.auto_clean_data()

# Get insights and recommendations
insights = analyzer.generate_insights()

# Export results
analyzer.export_report('analysis_report')
```

## 📋 Usage Examples

### Example 1: Basic Analysis
```python
from data_analyzer import AutoDataAnalyzer
import pandas as pd

# Create or load data
data = pd.read_csv('sales_data.csv')

# Initialize analyzer
analyzer = AutoDataAnalyzer(data, target_column='sales')

# Comprehensive analysis in one go
quality_report = analyzer.analyze_data_quality()
eda_results = analyzer.perform_eda()
cleaned_data = analyzer.auto_clean_data()
insights = analyzer.generate_insights()

print("Analysis complete!")
```

### Example 2: Custom Cleaning Parameters
```python
# Custom data cleaning with specific strategies
cleaned_data = analyzer.auto_clean_data(
    handle_missing='impute_median',    # Use median for missing values
    remove_duplicates=True,            # Remove duplicate rows
    handle_outliers='cap',             # Cap outliers instead of removing
    encode_categorical=True            # Encode categorical variables
)
```

### Example 3: Running the Demo
```bash
python example_usage.py
```

This will run a comprehensive demo with sample data, showing all features of the system.

## 🎯 Dashboard Features

### 📊 **Data Overview Tab**
- Dataset dimensions and memory usage
- Data type breakdown (numeric/categorical)
- Interactive data preview table
- Key statistics at a glance

### 🔍 **Quality Analysis Tab**
- Missing value analysis with percentages
- Data type information
- Duplicate detection
- Quality issue alerts and warnings

### 📈 **Exploratory Data Analysis Tab**
- Statistical summaries for numeric variables
- Categorical variable analysis
- Correlation analysis with highly correlated pairs
- Target variable insights (if specified)

### 📊 **Visualizations Tab**
Organized into sub-tabs:
- **Missing Values**: Heatmap patterns
- **Correlations**: Interactive correlation matrix
- **Distributions**: Histograms for all numeric variables
- **Outliers**: Box plots for outlier detection
- **Categorical**: Bar charts for categorical distributions
- **Target Analysis**: Target variable specific visualizations

### 💡 **Insights Tab**
- **Key Insights**: Automated data insights
- **Recommendations**: Actionable preprocessing suggestions
- Color-coded insight categories

### 🧹 **Cleaning Tab**
- Interactive data cleaning controls
- Before/after comparison
- Detailed cleaning operation log
- Download cleaned dataset
- Configurable cleaning parameters

## 🛠️ Configuration Options

### Data Input Options
- **File Upload**: CSV file upload with drag-and-drop
- **Sample Datasets**: Built-in datasets (Iris, Tips, Gapminder, etc.)

### Analysis Parameters
- **Target Column**: Optional target variable selection
- **Missing Value Strategy**: Auto, drop, mean, median, mode imputation
- **Duplicate Handling**: Enable/disable duplicate removal
- **Outlier Treatment**: Auto, remove, cap, or ignore outliers
- **Categorical Encoding**: Enable/disable automatic encoding

## 📊 Supported Data Types

### Input Formats
- **CSV files** (primary support)
- **Pandas DataFrames** (programmatic usage)

### Data Types
- **Numeric**: integers, floats, scientific notation
- **Categorical**: strings, objects, categories
- **Datetime**: automatic detection and handling
- **Mixed Types**: intelligent type inference

## 🔧 Technical Architecture

### Core Components

1. **AutoDataAnalyzer Class** (`data_analyzer.py`)
   - Main analysis engine
   - Modular design with separate methods for each analysis type
   - Comprehensive logging and error handling

2. **Streamlit Dashboard** (`dashboard.py`)
   - Interactive web interface
   - Real-time analysis updates
   - Modern UI with custom CSS styling

3. **Example Usage** (`example_usage.py`)
   - Demonstration scripts
   - Sample data generation
   - Best practices examples

### Dependencies
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: plotly, matplotlib, seaborn
- **Web Interface**: streamlit
- **Machine Learning**: scikit-learn
- **Data Quality**: missingno, ydata-profiling

## 🎨 Visualization Gallery

The system generates multiple types of visualizations:

### Missing Value Analysis
- **Heatmap**: Shows missing value patterns across columns and rows
- **Bar Charts**: Missing value counts and percentages

### Statistical Visualizations
- **Correlation Heatmap**: Feature correlation matrix with values
- **Distribution Plots**: Histograms for all numeric variables
- **Box Plots**: Outlier detection and quartile analysis

### Categorical Analysis
- **Bar Charts**: Frequency distributions for categorical variables
- **Pie Charts**: Proportion analysis for target variables

### Target Variable Analysis
- **Histograms**: Distribution for numeric targets
- **Pie Charts**: Class distribution for categorical targets

## 📈 Performance Features

### Optimization
- **Parallel Processing**: Efficient computation for large datasets
- **Memory Management**: Optimized memory usage for large files
- **Caching**: Streamlit caching for improved performance
- **Progressive Loading**: Incremental analysis for better UX

### Scalability
- **Large Dataset Support**: Handles datasets with millions of rows
- **Memory Efficient**: Smart memory management and cleanup
- **Responsive UI**: Non-blocking operations with progress indicators

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Run tests and ensure code quality

### Feature Requests
- Data format support (Excel, JSON, etc.)
- Advanced statistical tests
- Machine learning model suggestions
- Custom visualization options

## 📄 License

This project is open-source and available under the MIT License.

## 🆘 Support

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Memory Issues**: For large datasets, increase system memory or use sampling
3. **Visualization Issues**: Update plotly and streamlit to latest versions

### Getting Help
- Check the example usage in `example_usage.py`
- Review the detailed docstrings in `data_analyzer.py`
- Run the demo to see expected behavior

---

**Built with ❤️ for data scientists and analysts who want to focus on insights, not data preprocessing!**