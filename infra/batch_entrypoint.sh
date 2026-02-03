#!/bin/sh
# AWS Batch entrypoint: run one chunk using AWS_BATCH_JOB_ARRAY_INDEX.
# Set env: BATCH_MANIFEST_S3, BATCH_RESULTS_BUCKET, BATCH_CHUNK_SIZE (default 100).
# Results go to s3://BUCKET/batch/results/<run-id>/chunk-<index>/ so each submitted job
# gets its own subfolder. Run ID is BATCH_RUN_PREFIX (if set) or AWS_BATCH_JOB_ID with
# ":array-index" stripped for array jobs.

CHUNK_ID="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
CHUNK_SIZE="${BATCH_CHUNK_SIZE:-100}"
OUTPUT_DIR="${BATCH_OUTPUT_DIR:-/results}"

if [ -z "$BATCH_MANIFEST_S3" ] || [ -z "$BATCH_RESULTS_BUCKET" ]; then
  echo "Error: BATCH_MANIFEST_S3 and BATCH_RESULTS_BUCKET must be set" >&2
  exit 1
fi

# Unique subfolder per job: BATCH_RUN_PREFIX (optional override) or job ID (strip :index for array jobs)
if [ -n "$BATCH_RUN_PREFIX" ]; then
  RUN_ID="$BATCH_RUN_PREFIX"
else
  RUN_ID="${AWS_BATCH_JOB_ID%%:*}"
fi
UPLOAD_S3="s3://${BATCH_RESULTS_BUCKET}/batch/results/${RUN_ID}/chunk-${CHUNK_ID}/"

exec python -m examples.batch.run_chunk \
  --manifest-s3 "$BATCH_MANIFEST_S3" \
  --chunk-id "$CHUNK_ID" \
  --chunk-size "$CHUNK_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --upload-s3 "$UPLOAD_S3"
