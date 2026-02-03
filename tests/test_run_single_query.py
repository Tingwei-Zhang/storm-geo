"""Unit and S3 (moto) tests for examples.batch.run_single_query."""
import json

import pytest

from examples.batch.run_single_query import (load_injection_doc_from_path,
                                             load_injection_doc_from_s3,
                                             load_injection_docs,
                                             sanitize_question_id)


class TestSanitizeQuestionId:
    def test_empty_returns_unknown(self):
        assert sanitize_question_id("") == "unknown"

    def test_unsafe_chars_replaced_by_underscore(self):
        assert "/" not in sanitize_question_id("health/123")
        assert sanitize_question_id("health_123") == "health_123"

    def test_leading_trailing_underscore_stripped(self):
        out = sanitize_question_id("___x___")
        assert out == "x" or out == "unknown"

    def test_length_capped_at_200(self):
        long_id = "a" * 300
        assert len(sanitize_question_id(long_id)) <= 200


class TestLoadInjectionDocFromPath:
    def test_single_object(self, tmp_injection_json):
        docs = load_injection_doc_from_path(tmp_injection_json)
        assert len(docs) == 1
        assert docs[0].url == "manual://test"
        assert docs[0].title == "Test Doc"
        # Snippets come from content split on \n\n (e.g. "Paragraph one.", "Paragraph two.")
        assert docs[0].snippets
        assert "Paragraph one." in docs[0].snippets

    def test_list_of_one(self, tmp_path):
        path = tmp_path / "inj.json"
        path.write_text(
            json.dumps([{"url": "u", "title": "T", "content": "C"}]),
            encoding="utf-8",
        )
        docs = load_injection_doc_from_path(path)
        assert len(docs) == 1
        assert docs[0].url == "u"


class TestLoadInjectionDocs:
    def test_neither_set_returns_empty_list(self):
        assert load_injection_docs() == []

    def test_path_set_and_exists(self, tmp_injection_json):
        docs = load_injection_docs(injection_doc_path=str(tmp_injection_json))
        assert len(docs) == 1
        assert docs[0].title == "Test Doc"

    def test_path_set_and_missing_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent.json"
        assert not missing.exists()
        docs = load_injection_docs(injection_doc_path=str(missing))
        assert docs == []


@pytest.mark.moto
class TestLoadInjectionDocFromS3:
    def test_load_from_s3(self, tmp_injection_json):
        try:
            from moto import mock_aws
        except ImportError:
            pytest.skip("moto not installed")
        with mock_aws():
            import boto3
            bucket = "test-bucket"
            key = "injection/doc.json"
            client = boto3.client("s3")
            client.create_bucket(Bucket=bucket)
            client.upload_file(str(tmp_injection_json), bucket, key)
            docs = load_injection_doc_from_s3(f"s3://{bucket}/{key}")
            assert len(docs) == 1
            assert docs[0].url == "manual://test"
            assert docs[0].title == "Test Doc"
