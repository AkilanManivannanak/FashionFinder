#!/bin/bash
KMP_DUPLICATE_LIB_OK=TRUE uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1 &
sleep 5
KMP_DUPLICATE_LIB_OK=TRUE streamlit run ui.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
