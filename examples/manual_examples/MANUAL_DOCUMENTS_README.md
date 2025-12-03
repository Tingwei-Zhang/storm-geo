# Manual document quick-start

`examples/manual_examples` now contains a single workflow:

1. Describe the manual evidence you want to force into the pipeline by editing `manual_document_example.json`.
2. Run `run_costorm_demo_first_retrieval_manual.py`. The script injects all documents from the JSON file into **only the first retrieval call** and leaves the rest of the run untouched.

## 1. Author your documents

Each entry inside `manual_document_example.json` follows the same shape used by Co-STORM’s `Information` class:

```json
{
  "url": "manual://my-document",
  "title": "My Custom Document",
  "content": "Full paragraph text. Use \\n\\n to split into snippets.",
  "description": "Optional short blurb (falls back to the first snippet).",
  "question": "Optional question this doc answers.",
  "query": "Optional search query that would fetch it.",
  "snippets": ["Optional", "pre-split", "snippets"]
}
```

If `snippets` is omitted the helper automatically splits `content` into paragraphs/sentences.

## 2. Run the demo

```
python examples/manual_examples/run_costorm_demo_first_retrieval_manual.py \
  --manual-doc-path=examples/manual_examples/manual_document_example.json
```

- You will be prompted for a topic just like the regular demo.
- The script uses the cheaper `gpt-4o-mini` presets from `run_costorm_gpt_demo.py`.
- Manual documents appear at the top of the very first retrieval response, so you can guarantee they are considered when the warm start begins.

## 3. Helper utilities

`manual_document_helper.py` exposes two tiny helpers:

- `create_information_from_document(...)`
- `create_information_from_dict({...})`

Both functions return the `Information` objects expected by the runner. They are used internally by the demo script but are available if you want to build custom tooling.

## Notes

- Keep URLs unique (e.g., `manual://doc-1`) so Co-STORM can track citations.
- You can point `--manual-doc-path` to any JSON file; the default just shows the format. Outputs are saved under `./results/co-storm-demo/<timestamp>_<slug>/` like the standard demo.
- Because the injection happens only once, the remainder of the run still reflects live retrieval results, giving you fine-grained control without rewriting the pipeline.

