#!/usr/bin/env python3
"""
Example usage of the AutoDataAnalyzer for end-to-end data analysis.
This script demonstrates how to use the system for automatic data exploration,
cleaning, and visualization.
"""

import pandas as pd
import numpy as np
from data_analyzer import AutoDataAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_dataset():
    """Create a sample dataset with various data quality issues for demonstration."""
    np.random.seed(42)
    
    # Generate sample data with intentional issues
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'score': np.random.normal(75, 15, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce data quality issues
    
    # 1. Missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'education'] = np.nan
    
    # 2. Outliers
    outlier_indices = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
    df.loc[outlier_indices, 'age'] = np.random.uniform(100, 120, len(outlier_indices))
    
    # 3. Duplicates
    duplicate_rows = df.sample(n=int(0.03 * n_samples))
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    # 4. Add a target variable for supervised analysis
    df['target'] = (df['score'] > 75).astype(int)
    
    return df

def demonstrate_auto_analysis():
    """Demonstrate the automatic data analysis capabilities."""
    print("🚀 Auto Data Analysis Demo")
    print("=" * 50)
    
    # Create sample dataset
    print("📊 Creating sample dataset with intentional data quality issues...")
    data = create_sample_dataset()
    print(f"Created dataset with shape: {data.shape}")
    
    # Initialize the analyzer
    print("\n🔧 Initializing AutoDataAnalyzer...")
    analyzer = AutoDataAnalyzer(data, target_column='target')
    
    # Step 1: Data Quality Analysis
    print("\n🔍 Step 1: Analyzing Data Quality...")
    quality_report = analyzer.analyze_data_quality()
    
    print(f"✅ Basic Info:")
    print(f"   - Shape: {quality_report['basic_info']['shape']}")
    print(f"   - Memory Usage: {quality_report['basic_info']['memory_usage'] / 1024:.2f} KB")
    
    print(f"✅ Missing Values:")
    total_missing = sum(quality_report['missing_values']['total_missing'].values())
    print(f"   - Total missing values: {total_missing}")
    print(f"   - Rows with missing data: {quality_report['missing_values']['rows_with_missing']}")
    
    print(f"✅ Duplicates:")
    print(f"   - Duplicate rows: {quality_report['duplicates']['duplicate_rows']}")
    
    if quality_report['quality_issues']:
        print(f"⚠️  Quality Issues Found:")
        for issue in quality_report['quality_issues']:
            print(f"   - {issue}")
    
    # Step 2: Exploratory Data Analysis
    print("\n📊 Step 2: Performing Exploratory Data Analysis...")
    eda_results = analyzer.perform_eda()
    
    if 'numeric_summary' in eda_results:
        print("✅ Numeric Variables Summary:")
        numeric_cols = list(eda_results['numeric_summary'].keys())
        print(f"   - Numeric columns: {numeric_cols}")
    
    if 'categorical_summary' in eda_results:
        print("✅ Categorical Variables Summary:")
        for col, info in eda_results['categorical_summary'].items():
            print(f"   - {col}: {info['unique_values']} unique values")
    
    if 'high_correlations' in eda_results and eda_results['high_correlations']:
        print(f"✅ High Correlations Found: {len(eda_results['high_correlations'])} pairs")
    
    # Step 3: Data Cleaning
    print("\n🧹 Step 3: Automatic Data Cleaning...")
    cleaned_data = analyzer.auto_clean_data()
    
    print(f"✅ Cleaning Results:")
    print(f"   - Original shape: {data.shape}")
    print(f"   - Cleaned shape: {cleaned_data.shape}")
    print(f"   - Cleaning steps performed: {len(analyzer.cleaning_log)}")
    
    for i, step in enumerate(analyzer.cleaning_log, 1):
        print(f"   {i}. {step}")
    
    # Step 4: Generate Visualizations
    print("\n📈 Step 4: Generating Visualizations...")
    visualizations = analyzer.generate_visualizations()
    print(f"✅ Generated {len(visualizations)} visualizations:")
    for viz_name in visualizations.keys():
        print(f"   - {viz_name}")
    
    # Step 5: Generate Insights
    print("\n💡 Step 5: Generating Insights and Recommendations...")
    insights = analyzer.generate_insights()
    
    print("✅ Key Insights:")
    for category, insight_list in insights.items():
        if insight_list:
            print(f"   {category.replace('_', ' ').title()}:")
            for insight in insight_list:
                print(f"     • {insight}")
    
    # Step 6: Export Report
    print("\n📋 Step 6: Exporting Analysis Report...")
    report_filename = analyzer.export_report('demo_analysis_report')
    print(f"✅ {report_filename}")
    
    print("\n🎉 Analysis Complete!")
    print("=" * 50)
    
    return analyzer, cleaned_data

def demonstrate_custom_analysis():
    """Demonstrate custom analysis options."""
    print("\n🛠️  Custom Analysis Demo")
    print("=" * 30)
    
    # Load a real dataset (using built-in sample data)
    try:
        import plotly.express as px
        data = px.data.tips()
        print(f"📊 Loaded Tips dataset: {data.shape}")
        
        # Initialize analyzer with target variable
        analyzer = AutoDataAnalyzer(data, target_column='total_bill')
        
        # Custom cleaning with specific parameters
        print("\n🧹 Custom Data Cleaning...")
        cleaned_data = analyzer.auto_clean_data(
            handle_missing='impute_median',
            remove_duplicates=True,
            handle_outliers='cap',
            encode_categorical=True
        )
        
        print(f"✅ Custom cleaning completed:")
        print(f"   - Original columns: {list(data.columns)}")
        print(f"   - Cleaned columns: {list(cleaned_data.columns)}")
        print(f"   - Shape change: {data.shape} → {cleaned_data.shape}")
        
        # Generate and display insights
        insights = analyzer.generate_insights()
        print("\n💡 Generated Insights:")
        for category, insight_list in insights.items():
            if insight_list:
                print(f"   {category.replace('_', ' ').title()}:")
                for insight in insight_list[:2]:  # Show first 2 insights
                    print(f"     • {insight}")
        
    except ImportError:
        print("⚠️  Plotly not available, using generated sample data instead")
        data = create_sample_dataset()
        analyzer = AutoDataAnalyzer(data)
        cleaned_data = analyzer.auto_clean_data()
        print(f"✅ Processed sample data: {data.shape} → {cleaned_data.shape}")

def main():
    """Main execution function."""
    print("🤖 AutoDataAnalyzer - End-to-End Data Analysis System")
    print("=" * 60)
    
    try:
        # Demonstrate automatic analysis
        analyzer, cleaned_data = demonstrate_auto_analysis()
        
        # Demonstrate custom analysis
        demonstrate_custom_analysis()
        
        print("\n📝 Summary:")
        print("The AutoDataAnalyzer provides:")
        print("✅ Automatic data quality assessment")
        print("✅ Comprehensive exploratory data analysis")
        print("✅ Intelligent data cleaning with multiple strategies")
        print("✅ Rich interactive visualizations")
        print("✅ Automated insights and recommendations")
        print("✅ Export capabilities for reports and cleaned data")
        
        print("\n🚀 To run the interactive dashboard:")
        print("   streamlit run dashboard.py")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()