#!/bin/bash
set -euo pipefail

STAGE_ID="${STAGE_ID:-unknown}"
WORK_DURATION_MS="${WORK_DURATION_MS:-1500}"
EXEC_FORMAT="${EXEC_FORMAT:-native}"
DT_CALLBACK="${DT_CALLBACK:-}"

echo "[worker] stage=${STAGE_ID} format=${EXEC_FORMAT} duration=${WORK_DURATION_MS}ms"

ms_to_s=$(python3 - "$WORK_DURATION_MS" <<'PY'
import sys
print(max(1, int(int(sys.argv[1]) / 1000)))
PY
)

if [[ "${EXEC_FORMAT}" == wasm* ]]; then
	stress-ng --cpu 1 --timeout "${ms_to_s}s" --metrics-brief
elif [[ "${EXEC_FORMAT}" == qemu-* ]]; then
	stress-ng --cpu 1 --timeout "${ms_to_s}s" --metrics-brief
else
	stress-ng --cpu "$(nproc)" --timeout "${ms_to_s}s" --metrics-brief
fi

if [[ -n "${DT_CALLBACK}" ]]; then
	latency=$(( WORK_DURATION_MS + (RANDOM % 200) ))
	payload=$(cat <<JSON
{"stage_id":"${STAGE_ID}","actual_latency_ms":${latency},"exec_format":"${EXEC_FORMAT}","node":"$(hostname)"}
JSON
)
	curl -sS -X POST "${DT_CALLBACK}" -H "Content-Type: application/json" -d "${payload}" || true
fi

echo "[worker] done stage=${STAGE_ID}"





