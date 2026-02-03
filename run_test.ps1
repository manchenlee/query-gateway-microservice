$h = python -c "from config import settings; print(settings.SERVER_HOST)".Trim()
$p = python -c "from config import settings; print(settings.SERVER_PORT)".Trim()
Write-Host "Stress Testing: http://${h}:${p}" -ForegroundColor Cyan
k6 run -e HOST=$h -e PORT=$p --out web-dashboard --summary-trend-stats="avg,min,med,max,p(50),p(95),p(99),p(99.9)" --http-debug="full" script.js