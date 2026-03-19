#!/bin/bash

# Start Blazegraph in the background
java -server -Xmx4g -jar blazegraph.jar &

# Wait for Blazegraph to initialize (adjust as needed)
sleep 10

# Run Streamlit in the foreground
streamlit run streamlit-app.py --server.port 8501 --server.address 0.0.0.0
