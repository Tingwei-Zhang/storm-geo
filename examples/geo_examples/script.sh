python examples/geo_examples/build_geo_dataset.py \
  --manifest manifest.csv \
  --manual-docs examples/geo_examples/manual_document_example.json \
  --n-health 50 \
  --n-law 50 \
  --n-money 50

python examples/geo_examples/run_geo_experiment_local.py \
  --geo-dataset geo_out/geo_dataset_20260215_192829_eb83.csv

python examples/geo_examples/evaluate_geo.py geo_out/geo_dataset_20260215_192829_eb83.csv

python examples/geo_examples/run_geo_experiment_local.py \
  --geo-dataset geo_out/geo_dataset_20260215_192829_eb83.csv \
  --workers 8