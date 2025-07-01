import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import io

# Configure page
st.set_page_config(
    page_title="Data Analysis Editor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
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
        color: #2e86de;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">📊 Data Analysis Editor</div>', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "Data Upload", 
    "Data Overview", 
    "Data Cleaning", 
    "Visualization", 
    "Statistical Analysis", 
    "Machine Learning"
])

# Data Upload Section
if page == "Data Upload":
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)
    
    upload_option = st.radio("Choose upload method:", ["Upload File", "Sample Data"])
    
    if upload_option == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel files"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                else:
                    st.session_state.data = pd.read_excel(uploaded_file)
                
                st.success(f"Successfully loaded {uploaded_file.name}")
                st.write(f"Data shape: {st.session_state.data.shape}")
                st.dataframe(st.session_state.data.head())
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    else:
        sample_choice = st.selectbox("Choose sample dataset:", [
            "Iris", "Tips", "Flights", "Titanic"
        ])
        
        if st.button("Load Sample Data"):
            if sample_choice == "Iris":
                st.session_state.data = sns.load_dataset('iris')
            elif sample_choice == "Tips":
                st.session_state.data = sns.load_dataset('tips')
            elif sample_choice == "Flights":
                st.session_state.data = sns.load_dataset('flights')
            elif sample_choice == "Titanic":
                st.session_state.data = sns.load_dataset('titanic')
            
            st.success(f"Loaded {sample_choice} dataset")
            st.write(f"Data shape: {st.session_state.data.shape}")
            st.dataframe(st.session_state.data.head())

# Data Overview Section
elif page == "Data Overview":
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.data
        st.markdown('<div class="section-header">📋 Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", data.shape[0])
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data)
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum(),
            'Unique Values': data.nunique()
        })
        st.dataframe(col_info)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())

# Data Cleaning Section
elif page == "Data Cleaning":
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.data.copy()
        st.markdown('<div class="section-header">🧹 Data Cleaning</div>', unsafe_allow_html=True)
        
        # Missing values handling
        st.subheader("Handle Missing Values")
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            selected_col = st.selectbox("Select column to handle:", missing_cols)
            method = st.selectbox("Select method:", [
                "Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill"
            ])
            
            if st.button("Apply"):
                if method == "Drop rows":
                    data = data.dropna(subset=[selected_col])
                elif method == "Fill with mean":
                    if data[selected_col].dtype in ['int64', 'float64']:
                        data[selected_col].fillna(data[selected_col].mean(), inplace=True)
                elif method == "Fill with median":
                    if data[selected_col].dtype in ['int64', 'float64']:
                        data[selected_col].fillna(data[selected_col].median(), inplace=True)
                elif method == "Fill with mode":
                    data[selected_col].fillna(data[selected_col].mode()[0], inplace=True)
                elif method == "Forward fill":
                    data[selected_col].fillna(method='ffill', inplace=True)
                
                st.session_state.data = data
                st.success("Applied successfully!")
        else:
            st.info("No missing values found!")
        
        # Remove duplicates
        st.subheader("Remove Duplicates")
        if st.button("Remove Duplicate Rows"):
            before_count = len(data)
            data = data.drop_duplicates()
            after_count = len(data)
            st.session_state.data = data
            st.success(f"Removed {before_count - after_count} duplicate rows")

# Visualization Section
elif page == "Visualization":
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.data
        st.markdown('<div class="section-header">📈 Data Visualization</div>', unsafe_allow_html=True)
        
        viz_type = st.selectbox("Select visualization type:", [
            "Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Bar Plot", "Heatmap", "Pair Plot"
        ])
        
        if viz_type == "Histogram":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select column:", numeric_cols)
                bins = st.slider("Number of bins:", 10, 100, 30)
                
                fig = px.histogram(data, x=col, nbins=bins, title=f"Histogram of {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select column:", numeric_cols)
                
                fig = px.box(data, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X axis:", numeric_cols)
                y_col = st.selectbox("Select Y axis:", numeric_cols)
                
                fig = px.scatter(data, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Heatmap":
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                corr_matrix = numeric_data.corr()
                
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                               title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)

# Statistical Analysis Section
elif page == "Statistical Analysis":
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.data
        st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox("Select analysis type:", [
            "Descriptive Statistics", "Correlation Analysis", "T-Test", "ANOVA"
        ])
        
        if analysis_type == "Descriptive Statistics":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:3])
                
                if selected_cols:
                    desc_stats = data[selected_cols].describe()
                    st.dataframe(desc_stats)
        
        elif analysis_type == "Correlation Analysis":
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                corr_matrix = numeric_data.corr()
                st.dataframe(corr_matrix)
                
                # Strong correlations
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                corr_val
                            ))
                
                if strong_corr:
                    st.subheader("Strong Correlations (|r| > 0.7)")
                    for col1, col2, corr in strong_corr:
                        st.write(f"{col1} - {col2}: {corr:.3f}")

# Machine Learning Section
elif page == "Machine Learning":
    if st.session_state.data is None:
        st.warning("Please upload data first!")
    else:
        data = st.session_state.data
        st.markdown('<div class="section-header">🤖 Machine Learning</div>', unsafe_allow_html=True)
        
        ml_type = st.selectbox("Select ML type:", ["Linear Regression", "Classification"])
        
        if ml_type == "Linear Regression":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                target_col = st.selectbox("Select target variable:", numeric_cols)
                feature_cols = st.multiselect(
                    "Select features:", 
                    [col for col in numeric_cols if col != target_col]
                )
                
                if feature_cols and st.button("Train Model"):
                    X = data[feature_cols].dropna()
                    y = data[target_col].dropna()
                    
                    # Align X and y
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(X) > 0:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² Score", f"{r2:.3f}")
                        with col2:
                            st.metric("MSE", f"{mse:.3f}")
                        
                        # Prediction vs Actual plot
                        fig = px.scatter(x=y_test, y=y_pred, 
                                       labels={'x': 'Actual', 'y': 'Predicted'},
                                       title="Actual vs Predicted Values")
                        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                               y=[y_test.min(), y_test.max()],
                                               mode='lines', name='Perfect Prediction'))
                        st.plotly_chart(fig, use_container_width=True)

# Download processed data
if st.session_state.data is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Download Data")
    
    csv = st.session_state.data.to_csv(index=False)
    st.sidebar.download_button(
        label="Download as CSV",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    )