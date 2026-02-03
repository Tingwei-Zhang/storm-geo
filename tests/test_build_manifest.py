"""Unit tests for examples.batch.build_manifest."""
import csv
import json
from pathlib import Path

import pytest

from examples.batch import build_manifest as bm


class TestNormalizeRow:
    def test_lowercase_and_strip_keys_and_string_values(self):
        out = bm._normalize_row({"  Title  ": "  foo  ", "Body": "bar"})
        assert out == {"title": "foo", "body": "bar"}

    def test_non_string_values_unchanged(self):
        out = bm._normalize_row({"x": 1, "y": None})
        assert out["x"] == 1
        assert out["y"] is None


class TestTopicFromRow:
    def test_title_only(self):
        row = {"title": "My title", "body": "Some body"}
        assert bm._topic_from_row(row, "title", 500) == "My title"

    def test_title_and_body_truncated(self):
        row = {"title": "T", "body": "a" * 600}
        topic = bm._topic_from_row(row, "title_and_body", 100)
        assert topic.startswith("T ")
        assert "..." in topic
        assert len(topic) <= 100 + len("T ") + 3

    def test_empty_title_returns_unknown(self):
        row = {"title": "", "body": ""}
        assert bm._topic_from_row(row, "title", 500) == "unknown"

    def test_empty_body_uses_title_only(self):
        row = {"title": "Only", "body": ""}
        assert bm._topic_from_row(row, "title_and_body", 500) == "Only"


class TestDatasetNameFromPath:
    def test_health_clusters_returns_health(self):
        assert bm._dataset_name_from_path(Path("health_clusters.csv")) == "health"

    def test_path_without_clusters_returns_stem(self):
        assert bm._dataset_name_from_path(Path("other.csv")) == "other"


class TestDatasetNameFromS3Uri:
    def test_health_clusters_key_returns_health(self):
        assert bm._dataset_name_from_s3_uri("s3://b/path/health_clusters.csv") == "health"

    def test_key_without_clusters_returns_stem(self):
        assert bm._dataset_name_from_s3_uri("s3://b/foo.csv") == "foo"


class TestBuildManifest:
    def test_from_local_dir(self, tmp_cluster_dir):
        rows = bm.build_manifest(input_dir=tmp_cluster_dir, topic_from="title")
        assert len(rows) == 2
        assert rows[0]["question_id"] == "health_1"
        assert rows[0]["topic"] == "What are these lines?"
        assert rows[0]["dataset"] == "health"
        assert rows[1]["question_id"] == "health_2"

    def test_duplicate_fail_fast_raises(self, tmp_path):
        cluster_dir = tmp_path / "clusters"
        cluster_dir.mkdir()
        path = cluster_dir / "health_clusters.csv"
        path.write_text(
            "question_id,title,body,domain,cluster_id\n"
            "1,A,<p>x</p>,health,1\n"
            "1,B,<p>y</p>,health,1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate question_id"):
            bm.build_manifest(
                input_dir=cluster_dir,
                topic_from="title",
                duplicate_policy="fail_fast",
            )

    def test_duplicate_first_wins_keeps_first(self, tmp_path):
        cluster_dir = tmp_path / "clusters"
        cluster_dir.mkdir()
        path = cluster_dir / "health_clusters.csv"
        path.write_text(
            "question_id,title,body,domain,cluster_id\n"
            "1,First,<p>x</p>,health,1\n"
            "1,Second,<p>y</p>,health,1\n",
            encoding="utf-8",
        )
        rows = bm.build_manifest(
            input_dir=cluster_dir,
            topic_from="title",
            duplicate_policy="first_wins",
        )
        assert len(rows) == 1
        assert rows[0]["topic"] == "First"

    def test_requires_input_dir_or_s3(self):
        with pytest.raises(ValueError, match="input_dir or input_s3_prefix"):
            bm.build_manifest()


class TestWriteManifestCsv:
    def test_roundtrip(self, tmp_path):
        rows = [
            {
                "question_id": "h_1",
                "original_question_id": "1",
                "dataset": "h",
                "topic": "T",
                "title": "T",
                "body": "",
                "domain": "",
                "cluster_id": "",
                "injection_doc_path": "",
                "injection_doc_s3_uri": "",
            },
        ]
        p = tmp_path / "out.csv"
        bm.write_manifest_csv(rows, p)
        assert p.exists()
        with open(p, newline="", encoding="utf-8") as f:
            read = list(csv.DictReader(f))
        assert len(read) == 1
        assert read[0]["question_id"] == "h_1"
        assert read[0]["topic"] == "T"


class TestWriteManifestJsonl:
    def test_roundtrip(self, tmp_path):
        rows = [
            {"question_id": "h_1", "topic": "T", "dataset": "h"},
        ]
        p = tmp_path / "out.jsonl"
        bm.write_manifest_jsonl(rows, p)
        assert p.exists()
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["question_id"] == "h_1"
