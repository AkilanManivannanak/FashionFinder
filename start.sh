#!/bin/bash
# Start FastAPI backend in background
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1 &

# Wait for backend to be ready
sleep 5

# Start Streamlit frontend
KMP_DUPLICATE_LIB_OK=TRUE streamlit run ui.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
