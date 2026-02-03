"""Shared fixtures for batch tests."""
from pathlib import Path

import pytest


@pytest.fixture
def tmp_manifest_csv(tmp_path):
    """Small manifest CSV (2 rows) on disk."""
    path = tmp_path / "manifest.csv"
    content = """question_id,original_question_id,dataset,topic,title,body,domain,cluster_id,injection_doc_path,injection_doc_s3_uri
health_1,1,health,What are these lines?,What are these lines?,"<p>Tooth cracks.</p>",health,101,,
health_2,2,health,Another topic,Another topic,"<p>Body here.</p>",health,102,,
"""
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def tmp_cluster_csv(tmp_path):
    """One *_clusters.csv in a dir (for build_manifest input_dir)."""
    cluster_dir = tmp_path / "clusters"
    cluster_dir.mkdir()
    path = cluster_dir / "health_clusters.csv"
    content = """question_id,title,body,domain,cluster_id,cluster_probability,cluster_size
1,What are these lines?,<p>Tooth cracks.</p>,health,101,0.9,5
2,Another topic,<p>Body here.</p>,health,102,0.8,3
"""
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def tmp_cluster_dir(tmp_cluster_csv):
    """Directory containing one *_clusters.csv."""
    return tmp_cluster_csv.parent


@pytest.fixture
def tmp_injection_json(tmp_path):
    """Single-doc injection JSON (manual_document_example format)."""
    path = tmp_path / "injection.json"
    doc = {
        "url": "manual://test",
        "title": "Test Doc",
        "content": "Paragraph one.\n\nParagraph two.",
        "description": "Test description",
        "question": "Test question?",
        "query": "test query",
    }
    import json
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def tmp_dir_with_files(tmp_path):
    """Temp dir with a.txt and subdir/b.txt for upload_dir_to_s3 tests."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "b.txt").write_text("world", encoding="utf-8")
    return tmp_path
