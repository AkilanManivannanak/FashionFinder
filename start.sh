#!/bin/bash
export PORT=${PORT:-10000}

# Start FastAPI on 8001 in background  
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --host 0.0.0.0 --port 8001 &

# Start Streamlit on PORT that Render expects
KMP_DUPLICATE_LIB_OK=TRUE streamlit run ui.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
