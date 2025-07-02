#!/usr/bin/env python3
"""
Setup script for the Auto Data Analysis Dashboard
"""

import sys
import subprocess
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("🔧 Installing required packages...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    
    print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible!")
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data for testing...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample dataset
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 12, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                        n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 
                                   n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
            'purchase_amount': np.random.gamma(2, 100, n_samples),
            'satisfaction_score': np.random.normal(7.5, 1.5, n_samples),
            'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 
                                       n_samples, p=[0.4, 0.3, 0.2, 0.1])
        }
        
        df = pd.DataFrame(sample_data)
        
        # Introduce some data quality issues
        # Missing values
        missing_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
        df.loc[missing_indices, 'income'] = np.nan
        
        missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
        df.loc[missing_indices, 'education'] = np.nan
        
        # Outliers
        outlier_indices = np.random.choice(df.index, size=int(0.02 * n_samples), replace=False)
        df.loc[outlier_indices, 'age'] = np.random.uniform(100, 120, len(outlier_indices))
        
        # Save sample data
        if not os.path.exists('sample_data'):
            os.makedirs('sample_data')
        
        df.to_csv('sample_data/customer_data.csv', index=False)
        print("✅ Sample data created: sample_data/customer_data.csv")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error creating sample data: {e}")
        print("Please install dependencies first")
        return False

def run_demo():
    """Run the demo script"""
    print("🚀 Running demo...")
    
    try:
        subprocess.run([sys.executable, "example_usage.py"])
        return True
    except FileNotFoundError:
        print("❌ Demo script not found. Please ensure example_usage.py exists.")
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🌟 Launching dashboard...")
    print("📌 The dashboard will open in your browser at http://localhost:8501")
    print("📌 Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install dependencies first.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main setup function"""
    print("🚀 Auto Data Analysis Dashboard Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    while True:
        print("\n📋 Setup Options:")
        print("1. Install dependencies")
        print("2. Create sample data")
        print("3. Run demo")
        print("4. Launch dashboard")
        print("5. Complete setup (1+2)")
        print("6. Exit")
        
        choice = input("\nSelect an option (1-6): ").strip()
        
        if choice == "1":
            install_requirements()
        
        elif choice == "2":
            create_sample_data()
        
        elif choice == "3":
            run_demo()
        
        elif choice == "4":
            launch_dashboard()
        
        elif choice == "5":
            print("🔄 Running complete setup...")
            if install_requirements():
                create_sample_data()
                print("\n✅ Setup complete!")
                print("🚀 You can now run 'streamlit run dashboard.py' to start the dashboard")
            else:
                print("❌ Setup failed during dependency installation")
        
        elif choice == "6":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-6.")

if __name__ == "__main__":
    main()