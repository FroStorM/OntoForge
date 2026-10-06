#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:?Usage: scripts/run_text2kg_nohup.sh MODEL_PROFILE [LIMIT_PER_DOMAIN] [DATASET]}"
LIMIT_PER_DOMAIN="${2:-50}"
DATASET="${3:-wikidata_tekgen}"
DOMAINS=(1_movie 2_music 6_computer 8_politics)

mkdir -p logs results

RUN_NAME="text2kg-${PROFILE}-${DATASET}-wikidata4-limit${LIMIT_PER_DOMAIN}"
OUT_DIR="results/${RUN_NAME}"
LOG_FILE="logs/${RUN_NAME}.log"

nohup python scripts/text2kg_experiment.py \
  --model-profile "${PROFILE}" \
  --dataset "${DATASET}" \
  --domains "${DOMAINS[@]}" \
  --limit-per-domain "${LIMIT_PER_DOMAIN}" \
  --request-sleep 3 \
  --retry-sleep 10 \
  --out-dir "${OUT_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "Started ${RUN_NAME}"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUT_DIR}"
echo "Monitor with: tail -f ${LOG_FILE}"

