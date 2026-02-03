"""Unit and S3 (moto) tests for examples.batch.run_chunk."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from examples.batch.run_chunk import (_read_manifest_csv, load_manifest_rows,
                                      run_chunk, upload_dir_to_s3)


class TestLoadManifestRowsLocal:
    def test_load_from_path(self, tmp_manifest_csv):
        rows = load_manifest_rows(manifest_path=tmp_manifest_csv, manifest_s3_uri=None)
        assert len(rows) == 2
        assert rows[0].get("question_id") == "health_1"
        assert rows[0].get("topic") == "What are these lines?"

    def test_keys_normalized(self, tmp_manifest_csv):
        rows = load_manifest_rows(manifest_path=tmp_manifest_csv, manifest_s3_uri=None)
        assert "question_id" in rows[0]
        assert "topic" in rows[0]


class TestReadManifestCsv:
    def test_read_with_extra_spaces_in_header(self, tmp_path):
        p = tmp_path / "m.csv"
        p.write_text(
            " question_id , topic \n"
            "h_1, T1 \n",
            encoding="utf-8",
        )
        rows = _read_manifest_csv(p)
        assert len(rows) == 1
        # Keys are stripped; values are left as-is by _read_manifest_csv
        assert rows[0]["question_id"] == "h_1"
        assert rows[0]["topic"] == " T1 "


class TestRunChunkWithMock:
    def test_summary_one_ok_one_fail(self, tmp_path, tmp_manifest_csv):
        rows = load_manifest_rows(manifest_path=tmp_manifest_csv, manifest_s3_uri=None)
        output_dir = tmp_path / "out"
        call_count = [0]

        def mock_run_single_query(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                qid = kwargs.get("question_id", "unknown")
                out = Path(kwargs["output_dir"]) / qid
                out.mkdir(parents=True, exist_ok=True)
                (out / "log.json").write_text("{}", encoding="utf-8")
            else:
                raise RuntimeError("simulated failure")

        with patch("examples.batch.run_chunk.run_single_query", side_effect=mock_run_single_query):
            summary = run_chunk(rows, output_dir)

        assert summary["success_count"] == 1
        assert summary["failure_count"] == 1
        assert len(summary["failed"]) == 1
        assert summary["failed"][0]["question_id"] == "health_2"
        assert "simulated failure" in summary["failed"][0]["error"]

        summary_path = output_dir / "chunk_summary.json"
        assert summary_path.exists()
        with open(summary_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["success_count"] == 1
        assert loaded["failure_count"] == 1

        assert (output_dir / "health_1").exists()
        assert (output_dir / "health_1" / "log.json").exists()

    def test_missing_question_id_or_topic_appended_to_failed(self, tmp_path):
        rows = [
            {"question_id": "h_1", "topic": "T1"},
            {"question_id": "", "topic": "T2"},
            {"question_id": "h_3", "topic": ""},
        ]
        output_dir = tmp_path / "out"

        with patch("examples.batch.run_chunk.run_single_query") as m:
            m.return_value = None
            summary = run_chunk(rows, output_dir)

        assert summary["success_count"] == 1
        assert summary["failure_count"] == 2
        failed_qids = {f["question_id"] for f in summary["failed"]}
        assert "" in failed_qids or "unknown" in failed_qids
        assert "h_3" in failed_qids


@pytest.mark.moto
class TestLoadManifestRowsS3:
    def test_load_from_s3(self, tmp_manifest_csv):
        try:
            from moto import mock_aws
        except ImportError:
            pytest.skip("moto not installed")
        with mock_aws():
            import boto3
            bucket = "test-bucket"
            key = "batch/manifest.csv"
            client = boto3.client("s3")
            client.create_bucket(Bucket=bucket)
            client.upload_file(str(tmp_manifest_csv), bucket, key)
            rows = load_manifest_rows(manifest_path=None, manifest_s3_uri=f"s3://{bucket}/{key}")
            assert len(rows) == 2
            assert rows[0].get("question_id") == "health_1"


@pytest.mark.moto
class TestUploadDirToS3:
    def test_upload_dir_tree(self, tmp_dir_with_files):
        try:
            from moto import mock_aws
        except ImportError:
            pytest.skip("moto not installed")
        with mock_aws():
            import boto3
            bucket = "test-bucket"
            prefix = "s3://test-bucket/results/chunk-0/"
            client = boto3.client("s3")
            client.create_bucket(Bucket=bucket)
            upload_dir_to_s3(tmp_dir_with_files, prefix)
            paginator = client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=bucket, Prefix="results/chunk-0/"):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            assert "results/chunk-0/a.txt" in keys
            assert "results/chunk-0/subdir/b.txt" in keys
            body_a = client.get_object(Bucket=bucket, Key="results/chunk-0/a.txt")["Body"].read()
            assert body_a.decode("utf-8") == "hello"
