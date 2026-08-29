#!/usr/bin/env bash
# Daily data pull for eqd-risk-engine — run automatically once per trading day
# via launchd (see docs/AUTOMATION.md). Uses absolute paths throughout since
# launchd runs jobs with a minimal environment, not a normal login shell.
set -uo pipefail

PROJECT_DIR="/Users/nikhilpandey/Project/eqd-risk-engine"
LOG_DIR="$PROJECT_DIR/logs"
TODAY="$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/daily_ingest_${TODAY}.log"
UNDERLYINGS=(SPX AAPL NVDA JPM XLE)

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR" || exit 1

{
    echo "=== Daily ingest run: $(date) ==="

    # Skip weekends/market holidays outright — running the pipeline on a
    # non-trading day produces zero usable data (a real, understood gotcha
    # from 2026-08-22), not silently corrupting anything, but there's no
    # point spending the network calls or cluttering the log.
    "$PROJECT_DIR/.venv/bin/python3" -c "
from eqdrisk.marketdata.calendar import is_trading_day
import datetime, sys
sys.exit(0 if is_trading_day(datetime.date.today()) else 1)
"
    if [ $? -ne 0 ]; then
        echo "$TODAY is not a trading day — skipping."
        echo "=== Done: $(date) ==="
        exit 0
    fi

    EQDRISK="$PROJECT_DIR/.venv/bin/eqdrisk"

    echo "--- ingest ---"
    "$EQDRISK" ingest --date "$TODAY" || echo "INGEST FAILED"

    echo "--- curves ---"
    "$EQDRISK" curves --date "$TODAY" || echo "CURVES FAILED"

    echo "--- iv ---"
    "$EQDRISK" iv --date "$TODAY" || echo "IV FAILED"

    for u in "${UNDERLYINGS[@]}"; do
        echo "--- calibrate $u ---"
        "$EQDRISK" calibrate --date "$TODAY" --underlying "$u" || echo "CALIBRATE $u FAILED"
    done

    echo "--- price ---"
    "$EQDRISK" price --date "$TODAY" || echo "PRICE FAILED"

    echo "--- varswap ---"
    "$EQDRISK" varswap --date "$TODAY" || echo "VARSWAP FAILED"

    echo "--- riskfactors ---"
    "$EQDRISK" riskfactors --date "$TODAY" || echo "RISKFACTORS FAILED"

    echo "=== Done: $(date) ==="
} >> "$LOG_FILE" 2>&1
