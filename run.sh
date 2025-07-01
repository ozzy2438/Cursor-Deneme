#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run the Streamlit application
echo "Starting Data Analysis Editor..."
echo "Open your browser and navigate to: http://localhost:8501"
echo "To stop the application, press Ctrl+C"

streamlit run app.py --server.port 8501 --server.address 0.0.0.0