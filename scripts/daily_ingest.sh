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

    echo "--- portfolio ---"
    "$EQDRISK" portfolio --date "$TODAY" || echo "PORTFOLIO FAILED"

    # day0 = the latest earlier trading day with complete market data of its own,
    # not simply the previous trading day: a day whose option-chain ingest failed
    # is explained across rather than skipped twice (planning/decisions.md, 2026-10-04).
    PREV_DAY="$("$PROJECT_DIR/.venv/bin/python3" -c "
import datetime
from eqdrisk.config import BaseConfig
from eqdrisk.portfolio.schema import Portfolio
from eqdrisk.pricing.pnl_explain import latest_complete_day_before
cfg = BaseConfig.from_yaml('configs/base.yaml')
day0 = latest_complete_day_before(cfg, datetime.date.fromisoformat('$TODAY'), Portfolio.from_yaml('configs/portfolio.yaml'))
print(day0.isoformat() if day0 else '')
")"
    if [ -z "$PREV_DAY" ]; then
        echo "--- explainpnl: SKIPPED (no earlier day with complete market data in the lookback window) ---"
    else
        echo "--- explainpnl ($PREV_DAY -> $TODAY) ---"
        EXPLAINPNL_OUTPUT="$("$EQDRISK" explainpnl --day0 "$PREV_DAY" --day1 "$TODAY" 2>&1)" \
            || echo "EXPLAINPNL FAILED"
        echo "$EXPLAINPNL_OUTPUT"
        if echo "$EXPLAINPNL_OUTPUT" | grep -q "ALERT:"; then
            echo "*** RESIDUAL ALERT: today's P&L-explain run breached its threshold — see ALERT lines above ***"
        fi
        if echo "$EXPLAINPNL_OUTPUT" | grep -q "INCONCLUSIVE:"; then
            echo "*** INCONCLUSIVE: some residuals are inside Monte Carlo noise wider than the threshold — see INCONCLUSIVE lines above ***"
        fi

        # Investigate only a pair the explain actually produced (it skips, and
        # writes nothing, when either day lacks complete market data).
        if echo "$EXPLAINPNL_OUTPUT" | grep -q "SKIPPED _all_"; then
            echo "--- aiinvestigate: SKIPPED (no P&L explain for this pair) ---"
        else
            echo "--- aiinvestigate ($PREV_DAY -> $TODAY) ---"
            AIINVESTIGATE_OUTPUT="$("$EQDRISK" aiinvestigate --day0 "$PREV_DAY" --day1 "$TODAY" 2>&1)" \
                || echo "AIINVESTIGATE FAILED"
            echo "$AIINVESTIGATE_OUTPUT"
            if echo "$AIINVESTIGATE_OUTPUT" | grep -q "AI unavailable"; then
                echo "*** AI unavailable today (Ollama not reachable) — not treated as a pipeline failure ***"
            fi
        fi
    fi

    echo "=== Done: $(date) ==="
} >> "$LOG_FILE" 2>&1
