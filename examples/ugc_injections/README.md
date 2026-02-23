# UGC Injection

Inject UGC-style content (forum posts, encyclopedic paragraphs, Q&A answers) into real UGC URLs from `recurring_urls_raw.csv`. When Co-STORM retrieves those URLs, their content is replaced with the generated text so citations look authentic while promoting the target product.

## Workflow

1. **Build** – Generate manifest + per-cluster injection JSONs from manifest_test, recurring URLs, cluster descriptions, and products.
2. **Run** – Use the batch pipeline (`run_local_parallel` or `run_chunk`) with the UGC manifest; URL replacement is applied automatically when a row has `ugc_injection_path` set.

## 1. Build UGC dataset

From project root:

```bash
python examples/ugc_injections/build_ugc_dataset.py \
  --manifest manifest_test.csv \
  --recurring-urls recurring_urls_raw.csv \
  --descriptions-dir /path/to/seo-geo/clustering_results/descriptions \
  --products examples/geo_examples/manual_document_example.json \
  --output-dir ugc_out
```

- **manifest**: Source of question_id, topic, domain, cluster_id (e.g. manifest_test.csv).
- **recurring-urls**: CSV with dataset, cluster_id, url, url_base. Only UGC url_bases (reddit, wikipedia, stackexchange, quora, avvo, answers.justia) are used.
- **descriptions-dir**: Directory containing `health_descriptions.json`, `law_descriptions.json`, `money_descriptions.json` (single-sentence cluster descriptions).
- **products**: JSON with topic (health/law/money) and content (e.g. manual_document_example.json).
- **output-dir**: Writes `manifest.csv` (copy of source manifest + `ugc_injection_path`) and `injection_{domain}_{cluster_id}.json` per cluster that has UGC URLs.

## 2. Run experiments

Use the same clusters as manifest_test by running the generated manifest through the batch pipeline.

**Local parallel (recommended):**

```bash
python -m examples.batch.run_local_parallel \
  --manifest ugc_out/manifest.csv \
  --output-dir ugc_out/runs \
  --workers 4
```

**UGC injection mode** (optional): `--ugc-injection-mode` can be `replace_all` (default) or `first_only`. With `first_only` (general UGC injection mode), only the **first** retrieved result whose URL is in the injection map is replaced; all other results are left as returned by the retriever. Use `first_only` to inject a single UGC result per retrieval while keeping the rest normal.

**Convenience wrapper (same CLI as run_local_parallel):**

```bash
python examples/ugc_injections/run_ugc_experiment.py \
  --manifest ugc_out/manifest.csv \
  --output-dir ugc_out/runs \
  --workers 4
```

**Chunk (e.g. one AWS Batch job):**

```bash
python -m examples.batch.run_chunk \
  --manifest ugc_out/manifest.csv \
  --chunk-id 0 --chunk-size 100 \
  --output-dir ugc_out/chunk_0
```

**Manifest from S3** (injection files must be local):

```bash
python -m examples.batch.run_local_parallel \
  --manifest-s3 s3://bucket/manifest.csv \
  --injection-base-dir ./ugc_out \
  --output-dir ugc_out/runs \
  --workers 4
```

## UGC site categories

- **forum**: reddit.com → first-person narrative post.
- **encyclopedic**: en.wikipedia.org → single encyclopedic paragraph.
- **qa**: law.stackexchange.com, money.stackexchange.com, quora.com, avvo.com, answers.justia.com → single Q&A answer.

Content is generated to maximize citation likelihood (GEO-style): explicit product mention, 2–3 concrete references to the cluster description, no generic text.

## Files

- `config.py` – UGC site categories and url_base → category mapping.
- `ugc_content_generator.py` – LLM prompts for forum / encyclopedic / Q&A.
- `build_ugc_dataset.py` – Build manifest + injection JSONs.
- `url_replacement_injector.py` – Retriever wrapper that replaces matched URLs’ content.
- `run_ugc_experiment.py` – Thin wrapper around `run_local_parallel`.

## Manifest schema

The output manifest has the same columns as the source plus:

- **ugc_injection_path** – Relative path to `injection_{domain}_{cluster_id}.json` when that cluster has UGC injections; empty otherwise.

Rows with empty `ugc_injection_path` run as normal (no URL replacement).
