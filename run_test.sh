#!/bin/bash

echo ">>> Starting FastAPI Server..."
python3 -u -m uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

echo ">>> Waiting for Server to be ready..."

for i in $(seq 1 60); do
  if curl -s http://localhost:8000/ > /dev/null; then
    echo ">>> [SUCCESS] Server is UP!"
    break
  fi
  if [ $i -eq 60 ]; then
    echo ">>> [ERROR] Server failed to start."
    exit 1
  fi
  printf "."
  sleep 1
done

echo ">>> Starting k6 Stress Test..."
k6 run --out web-dashboard --summary-trend-stats="avg,min,med,max,p(50),p(95),p(99),p(99.9)" script.js

echo ">>> Test finished! Report generated as report.html"
echo ">>> Container is keeping alive for you to view logs or copy report."
tail -f /dev/null