# 📊 Data Analysis Editor

A comprehensive web-based data analysis tool built with Streamlit that provides an intuitive interface for data exploration, visualization, and machine learning.

## Features

### 📁 Data Upload
- Upload CSV and Excel files
- Load sample datasets (Iris, Tips, Flights, Titanic)
- Real-time data preview

### 📋 Data Overview
- Dataset dimensions and memory usage
- Column information and data types
- Missing values analysis
- Statistical summary

### 🧹 Data Cleaning
- Handle missing values with multiple strategies
- Remove duplicate rows
- Interactive data cleaning operations

### 📈 Data Visualization
- **Histogram**: Distribution analysis
- **Box Plot**: Outlier detection
- **Scatter Plot**: Relationship analysis
- **Line Plot**: Trend analysis
- **Bar Plot**: Categorical comparisons
- **Heatmap**: Correlation visualization
- **Pair Plot**: Multi-variable relationships

### 📊 Statistical Analysis
- Descriptive statistics
- Correlation analysis
- T-tests and ANOVA
- Strong correlation detection

### 🤖 Machine Learning
- Linear regression modeling
- Model performance metrics
- Prediction visualization
- Feature selection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data-analysis-editor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **Upload Data**: Start by uploading your dataset or selecting a sample dataset
2. **Explore**: Use the Data Overview section to understand your data structure
3. **Clean**: Apply data cleaning operations as needed
4. **Visualize**: Create various charts and plots to explore patterns
5. **Analyze**: Perform statistical analysis on your data
6. **Model**: Build machine learning models for predictions
7. **Export**: Download your processed data

## Supported File Formats

- CSV files (.csv)
- Excel files (.xlsx, .xls)

## Dependencies

- Streamlit: Web app framework
- Pandas: Data manipulation
- NumPy: Numerical computing
- Matplotlib & Seaborn: Statistical plotting
- Plotly: Interactive visualizations
- SciPy: Statistical functions
- Scikit-learn: Machine learning

## Sample Datasets

The application includes several built-in sample datasets:
- **Iris**: Classic flower classification dataset
- **Tips**: Restaurant tips dataset
- **Flights**: Airline passenger data
- **Titanic**: Historic passenger survival data

## Contributing

Feel free to submit issues and pull requests to improve the application.

## License

This project is open source and available under the MIT License.