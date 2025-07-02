import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import missingno as msno
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import io
import base64
from typing import Dict, List, Tuple, Any, Optional
import logging

warnings.filterwarnings('ignore')

class AutoDataAnalyzer:
    """
    Comprehensive automatic data analysis system for exploration, cleaning, and visualization.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: Optional[str] = None):
        """
        Initialize the analyzer with data.
        
        Args:
            data: Input DataFrame
            target_column: Target variable for supervised analysis (optional)
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.target_column = target_column
        self.analysis_results = {}
        self.cleaning_log = []
        self.visualizations = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def analyze_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis.
        """
        quality_report = {
            'basic_info': {
                'shape': self.data.shape,
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'dtypes': self.data.dtypes.to_dict()
            },
            'missing_values': {
                'total_missing': self.data.isnull().sum().to_dict(),
                'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
                'rows_with_missing': self.data.isnull().any(axis=1).sum(),
                'complete_rows': (~self.data.isnull().any(axis=1)).sum()
            },
            'duplicates': {
                'duplicate_rows': self.data.duplicated().sum(),
                'duplicate_percentage': self.data.duplicated().sum() / len(self.data) * 100
            },
            'data_types': {
                'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
                'datetime_columns': list(self.data.select_dtypes(include=['datetime64']).columns)
            }
        }
        
        # Detect potential data quality issues
        quality_issues = []
        
        # High missing value columns
        high_missing = [(col, pct) for col, pct in quality_report['missing_values']['missing_percentage'].items() 
                       if pct > 50]
        if high_missing:
            quality_issues.append(f"High missing values (>50%): {high_missing}")
            
        # Duplicate rows
        if quality_report['duplicates']['duplicate_rows'] > 0:
            quality_issues.append(f"Found {quality_report['duplicates']['duplicate_rows']} duplicate rows")
            
        # Potential outliers in numeric columns
        numeric_cols = quality_report['data_types']['numeric_columns']
        outlier_columns = []
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(self.data) * 0.05:  # More than 5% outliers
                outlier_columns.append((col, outliers))
        
        if outlier_columns:
            quality_issues.append(f"Potential outlier columns: {outlier_columns}")
            
        quality_report['quality_issues'] = quality_issues
        self.analysis_results['data_quality'] = quality_report
        
        return quality_report
    
    def perform_eda(self) -> Dict[str, Any]:
        """
        Comprehensive Exploratory Data Analysis.
        """
        eda_results = {}
        
        # Statistical summary
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            eda_results['numeric_summary'] = self.data[numeric_cols].describe().to_dict()
            
            # Correlation analysis
            correlation_matrix = self.data[numeric_cols].corr()
            eda_results['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            eda_results['high_correlations'] = high_corr_pairs
        
        if len(categorical_cols) > 0:
            categorical_summary = {}
            for col in categorical_cols:
                categorical_summary[col] = {
                    'unique_values': self.data[col].nunique(),
                    'value_counts': self.data[col].value_counts().head(10).to_dict(),
                    'most_frequent': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None
                }
            eda_results['categorical_summary'] = categorical_summary
        
        # Target variable analysis (if specified)
        if self.target_column and self.target_column in self.data.columns:
            target_analysis = {
                'type': str(self.data[self.target_column].dtype),
                'unique_values': self.data[self.target_column].nunique(),
                'value_counts': self.data[self.target_column].value_counts().to_dict()
            }
            
            if self.data[self.target_column].dtype in ['int64', 'float64']:
                target_analysis['statistics'] = self.data[self.target_column].describe().to_dict()
            
            eda_results['target_analysis'] = target_analysis
        
        self.analysis_results['eda'] = eda_results
        return eda_results
    
    def auto_clean_data(self, 
                       handle_missing: str = 'auto',
                       remove_duplicates: bool = True,
                       handle_outliers: str = 'auto',
                       encode_categorical: bool = True) -> pd.DataFrame:
        """
        Automatic data cleaning pipeline.
        
        Args:
            handle_missing: 'auto', 'drop', 'impute_mean', 'impute_median', 'impute_mode'
            remove_duplicates: Whether to remove duplicate rows
            handle_outliers: 'auto', 'remove', 'cap', 'none'
            encode_categorical: Whether to encode categorical variables
        """
        cleaned_data = self.data.copy()
        cleaning_steps = []
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_data)
            if removed_duplicates > 0:
                cleaning_steps.append(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
        
        if handle_missing == 'auto':
            # Auto strategy: impute numeric with median, categorical with mode
            for col in numeric_cols:
                missing_pct = cleaned_data[col].isnull().sum() / len(cleaned_data) * 100
                if missing_pct > 0:
                    if missing_pct > 70:
                        cleaned_data = cleaned_data.drop(columns=[col])
                        cleaning_steps.append(f"Dropped column '{col}' (>70% missing)")
                    else:
                        median_val = cleaned_data[col].median()
                        cleaned_data[col].fillna(median_val, inplace=True)
                        cleaning_steps.append(f"Imputed missing values in '{col}' with median ({median_val:.2f})")
            
            for col in categorical_cols:
                missing_pct = cleaned_data[col].isnull().sum() / len(cleaned_data) * 100
                if missing_pct > 0:
                    if missing_pct > 70:
                        cleaned_data = cleaned_data.drop(columns=[col])
                        cleaning_steps.append(f"Dropped column '{col}' (>70% missing)")
                    else:
                        mode_val = cleaned_data[col].mode().iloc[0] if not cleaned_data[col].mode().empty else 'Unknown'
                        cleaned_data[col].fillna(mode_val, inplace=True)
                        cleaning_steps.append(f"Imputed missing values in '{col}' with mode ('{mode_val}')")
        
        # Handle outliers
        if handle_outliers == 'auto' and len(numeric_cols) > 0:
            for col in numeric_cols:
                if col in cleaned_data.columns:  # Check if column still exists after missing value handling
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                    outlier_count = outliers_mask.sum()
                    
                    if outlier_count > 0 and outlier_count < len(cleaned_data) * 0.1:  # Less than 10% outliers
                        # Cap outliers instead of removing them
                        cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                        cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
                        cleaning_steps.append(f"Capped {outlier_count} outliers in '{col}'")
        
        # Encode categorical variables
        if encode_categorical and len(categorical_cols) > 0:
            for col in categorical_cols:
                if col in cleaned_data.columns and col != self.target_column:
                    unique_values = cleaned_data[col].nunique()
                    if unique_values <= 10:  # One-hot encode if <= 10 categories
                        dummies = pd.get_dummies(cleaned_data[col], prefix=col, drop_first=True)
                        cleaned_data = pd.concat([cleaned_data.drop(columns=[col]), dummies], axis=1)
                        cleaning_steps.append(f"One-hot encoded '{col}' ({unique_values} categories)")
                    else:  # Label encode if > 10 categories
                        le = LabelEncoder()
                        cleaned_data[col] = le.fit_transform(cleaned_data[col].astype(str))
                        cleaning_steps.append(f"Label encoded '{col}' ({unique_values} categories)")
        
        self.cleaned_data = cleaned_data
        self.cleaning_log = cleaning_steps
        
        return cleaned_data
    
    def generate_visualizations(self) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for the data.
        """
        visualizations = {}
        
        # Data overview visualizations
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        # 1. Missing values heatmap
        if self.data.isnull().sum().sum() > 0:
            fig_missing = go.Figure(data=go.Heatmap(
                z=self.data.isnull().values,
                x=self.data.columns,
                y=list(range(len(self.data))),
                colorscale='Viridis',
                name='Missing Values'
            ))
            fig_missing.update_layout(
                title='Missing Values Pattern',
                xaxis_title='Columns',
                yaxis_title='Rows',
                height=400
            )
            visualizations['missing_pattern'] = fig_missing
        
        # 2. Correlation heatmap for numeric variables
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_corr.update_layout(
                title='Feature Correlation Matrix',
                height=500,
                width=500
            )
            visualizations['correlation_heatmap'] = fig_corr
        
        # 3. Distribution plots for numeric variables
        if len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig_dist = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=numeric_cols[:n_rows*n_cols],
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig_dist.add_trace(
                    go.Histogram(x=self.data[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig_dist.update_layout(
                title='Distribution of Numeric Variables',
                height=300*n_rows,
                showlegend=False
            )
            visualizations['distributions'] = fig_dist
        
        # 4. Box plots for outlier detection
        if len(numeric_cols) > 0:
            fig_box = go.Figure()
            for col in numeric_cols[:6]:  # Limit to first 6 columns
                fig_box.add_trace(go.Box(y=self.data[col], name=col))
            
            fig_box.update_layout(
                title='Box Plots for Outlier Detection',
                yaxis_title='Values',
                height=400
            )
            visualizations['box_plots'] = fig_box
        
        # 5. Categorical variable distributions
        if len(categorical_cols) > 0:
            for col in categorical_cols[:4]:  # Limit to first 4 categorical columns
                value_counts = self.data[col].value_counts().head(10)
                
                fig_cat = go.Figure(data=[
                    go.Bar(x=value_counts.index, y=value_counts.values)
                ])
                fig_cat.update_layout(
                    title=f'Distribution of {col}',
                    xaxis_title=col,
                    yaxis_title='Count',
                    height=400
                )
                visualizations[f'categorical_{col}'] = fig_cat
        
        # 6. Target variable analysis (if specified)
        if self.target_column and self.target_column in self.data.columns:
            if self.data[self.target_column].dtype in ['int64', 'float64']:
                # Numeric target
                fig_target = go.Figure(data=[
                    go.Histogram(x=self.data[self.target_column], name=self.target_column)
                ])
                fig_target.update_layout(
                    title=f'Distribution of Target Variable: {self.target_column}',
                    xaxis_title=self.target_column,
                    yaxis_title='Frequency'
                )
            else:
                # Categorical target
                target_counts = self.data[self.target_column].value_counts()
                fig_target = go.Figure(data=[
                    go.Pie(labels=target_counts.index, values=target_counts.values)
                ])
                fig_target.update_layout(
                    title=f'Distribution of Target Variable: {self.target_column}'
                )
            
            visualizations['target_distribution'] = fig_target
        
        self.visualizations = visualizations
        return visualizations
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate automated insights from the analysis.
        """
        insights = {
            'data_overview': [],
            'quality_insights': [],
            'statistical_insights': [],
            'recommendations': []
        }
        
        # Data overview insights
        insights['data_overview'].append(f"Dataset contains {self.data.shape[0]} rows and {self.data.shape[1]} columns")
        
        numeric_cols = len(self.data.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(self.data.select_dtypes(include=['object']).columns)
        
        insights['data_overview'].append(f"Data types: {numeric_cols} numeric, {categorical_cols} categorical columns")
        
        # Quality insights
        missing_percentage = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        insights['quality_insights'].append(f"Overall missing data: {missing_percentage:.2f}%")
        
        duplicate_percentage = (self.data.duplicated().sum() / len(self.data)) * 100
        if duplicate_percentage > 0:
            insights['quality_insights'].append(f"Duplicate rows: {duplicate_percentage:.2f}%")
        
        # Statistical insights
        if 'eda' in self.analysis_results:
            eda_results = self.analysis_results['eda']
            
            if 'high_correlations' in eda_results and eda_results['high_correlations']:
                insights['statistical_insights'].append(
                    f"Found {len(eda_results['high_correlations'])} highly correlated feature pairs (|r| > 0.8)"
                )
            
            if 'categorical_summary' in eda_results:
                high_cardinality_cols = [
                    col for col, info in eda_results['categorical_summary'].items() 
                    if info['unique_values'] > 50
                ]
                if high_cardinality_cols:
                    insights['statistical_insights'].append(
                        f"High cardinality categorical columns: {high_cardinality_cols}"
                    )
        
        # Recommendations
        if missing_percentage > 10:
            insights['recommendations'].append("Consider advanced missing value imputation techniques")
        
        if duplicate_percentage > 5:
            insights['recommendations'].append("Remove duplicate rows to improve data quality")
        
        if numeric_cols > 0:
            insights['recommendations'].append("Consider feature scaling for machine learning models")
        
        if categorical_cols > 0:
            insights['recommendations'].append("Encode categorical variables before modeling")
        
        return insights
    
    def export_report(self, filename: str = 'data_analysis_report') -> str:
        """
        Export comprehensive analysis report.
        """
        report = {
            'data_quality': self.analysis_results.get('data_quality', {}),
            'eda_results': self.analysis_results.get('eda', {}),
            'cleaning_log': self.cleaning_log,
            'insights': self.generate_insights()
        }
        
        # Save as JSON for programmatic access
        import json
        with open(f"{filename}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return f"Report exported to {filename}.json"