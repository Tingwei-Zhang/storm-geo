#!/bin/sh
# AWS Batch entrypoint: run one chunk using AWS_BATCH_JOB_ARRAY_INDEX.
# Set env: BATCH_MANIFEST_S3, BATCH_RESULTS_BUCKET, BATCH_CHUNK_SIZE (default 100).

CHUNK_ID="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
CHUNK_SIZE="${BATCH_CHUNK_SIZE:-100}"
OUTPUT_DIR="${BATCH_OUTPUT_DIR:-/results}"

if [ -z "$BATCH_MANIFEST_S3" ] || [ -z "$BATCH_RESULTS_BUCKET" ]; then
  echo "Error: BATCH_MANIFEST_S3 and BATCH_RESULTS_BUCKET must be set" >&2
  exit 1
fi

UPLOAD_S3="s3://${BATCH_RESULTS_BUCKET}/batch/results/chunk-${CHUNK_ID}/"

exec python -m examples.batch.run_chunk \
  --manifest-s3 "$BATCH_MANIFEST_S3" \
  --chunk-id "$CHUNK_ID" \
  --chunk-size "$CHUNK_SIZE" \
  --output-dir "$OUTPUT_DIR" \
  --upload-s3 "$UPLOAD_S3"
