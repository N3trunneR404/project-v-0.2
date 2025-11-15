#!/bin/bash
set -euo pipefail

STAGE_ID="${STAGE_ID:-unknown}"
WORK_DURATION_MS="${WORK_DURATION_MS:-1500}"
EXEC_FORMAT="${EXEC_FORMAT:-native}"
DT_CALLBACK="${DT_CALLBACK:-}"
COMPUTE_CPU="${COMPUTE_CPU:-1}"
COMPUTE_MEM_GB="${COMPUTE_MEM_GB:-1}"
RESOURCE_SCALE="${RESOURCE_SCALE:-0.01}"

echo "[worker] stage=${STAGE_ID} format=${EXEC_FORMAT} duration=${WORK_DURATION_MS}ms cpu=${COMPUTE_CPU} mem=${COMPUTE_MEM_GB}GB scale=${RESOURCE_SCALE}"

# Scale resources for simulation (default 1:100)
# Real CPU cores = simulated CPU * scale
real_cpu=$(python3 -c "import sys; print(max(1, int(float(sys.argv[1]) * float(sys.argv[2]))))" "$COMPUTE_CPU" "$RESOURCE_SCALE")
real_mem_gb=$(python3 -c "import sys; print(max(0.1, float(sys.argv[1]) * float(sys.argv[2])))" "$COMPUTE_MEM_GB" "$RESOURCE_SCALE")

# Convert duration (typically not scaled, but configurable)
ms_to_s=$(python3 - "$WORK_DURATION_MS" <<'PY'
import sys
print(max(1, int(int(sys.argv[1]) / 1000)))
PY
)

echo "[worker] scaled: cpu=${real_cpu} cores, mem=${real_mem_gb}GB"

# Generate real CPU load
# Use stress-ng to consume actual CPU resources
if [[ "${EXEC_FORMAT}" == wasm* ]]; then
	# WASM: limited to 1 CPU
	stress-ng --cpu 1 --timeout "${ms_to_s}s" --metrics-brief
elif [[ "${EXEC_FORMAT}" == qemu-* ]]; then
	# QEMU: limited to 1 CPU (emulation overhead)
	stress-ng --cpu 1 --timeout "${ms_to_s}s" --metrics-brief
else
	# Native: use scaled CPU cores
	stress-ng --cpu "${real_cpu}" --timeout "${ms_to_s}s" --metrics-brief
fi

# Generate real memory load (if memory is specified)
# Use Python for floating point comparison (more reliable than bc)
mem_check=$(python3 -c "import sys; print('1' if float(sys.argv[1]) > 0.1 else '0')" "$real_mem_gb")
if [[ "$mem_check" == "1" ]]; then
	mem_mb=$(python3 -c "import sys; print(int(float(sys.argv[1]) * 1024))" "$real_mem_gb")
	stress-ng --vm 1 --vm-bytes "${mem_mb}M" --timeout "${ms_to_s}s" --metrics-brief &
	MEM_PID=$!
	wait $MEM_PID 2>/dev/null || true
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





