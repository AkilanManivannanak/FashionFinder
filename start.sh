#!/bin/bash
# Start FastAPI on 8001 in background
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1 &

# Wait for backend
sleep 8

# Start Streamlit on $PORT (what Render expects)
KMP_DUPLICATE_LIB_OK=TRUE streamlit run ui.py \
  --server.port ${PORT:-10000} \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
