import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_analyzer import AutoDataAnalyzer
import io
import json
from typing import Dict, Any
import base64

# Page configuration
st.set_page_config(
    page_title="Auto Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample datasets for demonstration."""
    datasets = {
        "Iris": px.data.iris(),
        "Tips": px.data.tips(),
        "Gapminder": px.data.gapminder(),
        "Stock Prices": px.data.stocks(),
        "Cars": px.data.cars()
    }
    return datasets

def display_data_overview(analyzer: AutoDataAnalyzer):
    """Display data overview section."""
    st.markdown('<div class="section-header">📋 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Rows</h4>
            <h2>{analyzer.data.shape[0]:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Columns</h4>
            <h2>{analyzer.data.shape[1]:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(analyzer.data.select_dtypes(include=[np.number]).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Numeric</h4>
            <h2>{numeric_cols}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        categorical_cols = len(analyzer.data.select_dtypes(include=['object']).columns)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Categorical</h4>
            <h2>{categorical_cols}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📝 Data Preview")
    st.dataframe(analyzer.data.head(10), use_container_width=True)

def display_data_quality(quality_report: Dict[str, Any]):
    """Display data quality analysis."""
    st.markdown('<div class="section-header">🔍 Data Quality Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        missing_data = quality_report['missing_values']
        
        # Missing values summary
        total_missing = sum(missing_data['total_missing'].values())
        if total_missing > 0:
            missing_df = pd.DataFrame([
                {'Column': col, 'Missing': count, 'Percentage': f"{pct:.2f}%"}
                for col, count, pct in zip(
                    missing_data['total_missing'].keys(),
                    missing_data['total_missing'].values(),
                    missing_data['missing_percentage'].values()
                ) if count > 0
            ])
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found!")
    
    with col2:
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame([
            {'Column': col, 'Type': str(dtype)}
            for col, dtype in quality_report['basic_info']['dtypes'].items()
        ])
        st.dataframe(dtypes_df, use_container_width=True)
    
    # Quality issues
    if quality_report['quality_issues']:
        st.subheader("⚠️ Quality Issues Detected")
        for issue in quality_report['quality_issues']:
            st.warning(issue)

def display_eda_results(eda_results: Dict[str, Any]):
    """Display exploratory data analysis results."""
    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Statistical summary for numeric columns
    if 'numeric_summary' in eda_results:
        st.subheader("📈 Numeric Variables Summary")
        numeric_summary_df = pd.DataFrame(eda_results['numeric_summary']).T
        st.dataframe(numeric_summary_df, use_container_width=True)
    
    # Categorical summary
    if 'categorical_summary' in eda_results:
        st.subheader("📝 Categorical Variables Summary")
        cat_summary_data = []
        for col, info in eda_results['categorical_summary'].items():
            cat_summary_data.append({
                'Column': col,
                'Unique Values': info['unique_values'],
                'Most Frequent': str(info['most_frequent'])
            })
        
        if cat_summary_data:
            cat_summary_df = pd.DataFrame(cat_summary_data)
            st.dataframe(cat_summary_df, use_container_width=True)
    
    # High correlations
    if 'high_correlations' in eda_results and eda_results['high_correlations']:
        st.subheader("🔗 High Correlations (|r| > 0.8)")
        high_corr_df = pd.DataFrame(eda_results['high_correlations'])
        st.dataframe(high_corr_df, use_container_width=True)

def display_visualizations(visualizations: Dict[str, Any]):
    """Display all generated visualizations."""
    st.markdown('<div class="section-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different types of visualizations
    viz_tabs = st.tabs([
        "Missing Values", "Correlations", "Distributions", 
        "Outliers", "Categorical", "Target Analysis"
    ])
    
    with viz_tabs[0]:
        if 'missing_pattern' in visualizations:
            st.plotly_chart(visualizations['missing_pattern'], use_container_width=True)
        else:
            st.info("No missing values to visualize!")
    
    with viz_tabs[1]:
        if 'correlation_heatmap' in visualizations:
            st.plotly_chart(visualizations['correlation_heatmap'], use_container_width=True)
        else:
            st.info("Not enough numeric variables for correlation analysis!")
    
    with viz_tabs[2]:
        if 'distributions' in visualizations:
            st.plotly_chart(visualizations['distributions'], use_container_width=True)
        else:
            st.info("No numeric variables to show distributions!")
    
    with viz_tabs[3]:
        if 'box_plots' in visualizations:
            st.plotly_chart(visualizations['box_plots'], use_container_width=True)
        else:
            st.info("No numeric variables for outlier detection!")
    
    with viz_tabs[4]:
        categorical_plots = [k for k in visualizations.keys() if k.startswith('categorical_')]
        if categorical_plots:
            for plot_key in categorical_plots:
                st.plotly_chart(visualizations[plot_key], use_container_width=True)
        else:
            st.info("No categorical variables to visualize!")
    
    with viz_tabs[5]:
        if 'target_distribution' in visualizations:
            st.plotly_chart(visualizations['target_distribution'], use_container_width=True)
        else:
            st.info("No target variable specified or found!")

def display_insights_and_recommendations(insights: Dict[str, Any]):
    """Display insights and recommendations."""
    st.markdown('<div class="section-header">💡 Insights & Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Insights")
        
        # Data overview insights
        if insights['data_overview']:
            st.markdown("**Data Overview:**")
            for insight in insights['data_overview']:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Quality insights
        if insights['quality_insights']:
            st.markdown("**Quality Insights:**")
            for insight in insights['quality_insights']:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Statistical insights
        if insights['statistical_insights']:
            st.markdown("**Statistical Insights:**")
            for insight in insights['statistical_insights']:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💡 Recommendations")
        if insights['recommendations']:
            for recommendation in insights['recommendations']:
                st.markdown(f"""
                <div class="recommendation-box">
                    ✅ {recommendation}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("Your data looks good! No specific recommendations at this time.")

def display_cleaning_log(cleaning_log: list):
    """Display data cleaning operations log."""
    st.markdown('<div class="section-header">🧹 Data Cleaning Log</div>', unsafe_allow_html=True)
    
    if cleaning_log:
        for i, step in enumerate(cleaning_log, 1):
            st.markdown(f"**Step {i}:** {step}")
    else:
        st.info("No data cleaning operations performed yet.")

def main():
    """Main dashboard application."""
    st.markdown('<div class="main-header">🚀 Auto Data Analysis Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Auto Data Analysis Dashboard**! This tool automatically performs comprehensive 
    data exploration, quality analysis, cleaning, and visualization. Simply upload your data or 
    select a sample dataset to get started.
    """)
    
    # Sidebar for data input and configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data input options
        data_source = st.radio(
            "Choose Data Source:",
            ["Upload File", "Sample Datasets"]
        )
        
        data = None
        target_column = None
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file to analyze"
            )
            
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.success(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
        
        else:
            sample_datasets = load_sample_data()
            selected_dataset = st.selectbox(
                "Choose a sample dataset:",
                list(sample_datasets.keys())
            )
            data = sample_datasets[selected_dataset]
            st.success(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Target column selection
        if data is not None:
            target_column = st.selectbox(
                "Select Target Column (optional):",
                ["None"] + list(data.columns),
                help="Select the target variable for supervised analysis"
            )
            if target_column == "None":
                target_column = None
        
        # Analysis options
        st.header("🔧 Analysis Options")
        
        auto_clean = st.checkbox("Auto Clean Data", value=True)
        handle_missing = st.selectbox(
            "Missing Values Strategy:",
            ["auto", "drop", "impute_mean", "impute_median", "impute_mode"]
        )
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        handle_outliers = st.selectbox(
            "Outlier Handling:",
            ["auto", "remove", "cap", "none"]
        )
        encode_categorical = st.checkbox("Encode Categorical Variables", value=True)
    
    # Main analysis
    if data is not None:
        # Initialize analyzer
        analyzer = AutoDataAnalyzer(data, target_column)
        
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Overview", "🔍 Quality", "📊 EDA", 
            "📈 Visualizations", "💡 Insights", "🧹 Cleaning"
        ])
        
        with tab1:
            display_data_overview(analyzer)
        
        with tab2:
            with st.spinner("Analyzing data quality..."):
                quality_report = analyzer.analyze_data_quality()
                display_data_quality(quality_report)
        
        with tab3:
            with st.spinner("Performing exploratory data analysis..."):
                eda_results = analyzer.perform_eda()
                display_eda_results(eda_results)
        
        with tab4:
            with st.spinner("Generating visualizations..."):
                visualizations = analyzer.generate_visualizations()
                display_visualizations(visualizations)
        
        with tab5:
            with st.spinner("Generating insights..."):
                insights = analyzer.generate_insights()
                display_insights_and_recommendations(insights)
        
        with tab6:
            st.subheader("🧹 Data Cleaning Operations")
            
            if auto_clean:
                if st.button("🚀 Run Auto Cleaning", type="primary"):
                    with st.spinner("Cleaning data..."):
                        cleaned_data = analyzer.auto_clean_data(
                            handle_missing=handle_missing,
                            remove_duplicates=remove_duplicates,
                            handle_outliers=handle_outliers,
                            encode_categorical=encode_categorical
                        )
                        
                        st.success("Data cleaning completed!")
                        
                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Before Cleaning")
                            st.write(f"Shape: {analyzer.data.shape}")
                            st.dataframe(analyzer.data.head(), use_container_width=True)
                        
                        with col2:
                            st.subheader("After Cleaning")
                            st.write(f"Shape: {cleaned_data.shape}")
                            st.dataframe(cleaned_data.head(), use_container_width=True)
                        
                        # Display cleaning log
                        display_cleaning_log(analyzer.cleaning_log)
                        
                        # Download cleaned data
                        csv = cleaned_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Cleaned Data",
                            data=csv,
                            file_name="cleaned_data.csv",
                            mime="text/csv"
                        )
            else:
                st.info("Enable 'Auto Clean Data' in the sidebar to perform data cleaning.")
        
        # Export functionality
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Analysis Report", type="secondary"):
                report_filename = analyzer.export_report()
                st.success(f"Analysis report exported!")
        
        with col2:
            # Download sample data template
            template_data = pd.DataFrame({
                'numeric_feature': np.random.randn(100),
                'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
                'target': np.random.choice([0, 1], 100)
            })
            template_csv = template_data.to_csv(index=False)
            st.download_button(
                label="📋 Download Sample Template",
                data=template_csv,
                file_name="sample_template.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Please upload a CSV file or select a sample dataset from the sidebar to begin analysis.")

if __name__ == "__main__":
    main()