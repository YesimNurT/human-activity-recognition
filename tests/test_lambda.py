"""Lambda handler tests with DynamoDB and CloudWatch replaced by fakes."""
import importlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aws" / "src"))


class FakeBatch:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def put_item(self, Item):
        self.store.append(Item)


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setenv("TABLE_NAME", "test-table")
    stored, metrics = [], []
    table = mock.Mock()
    table.batch_writer.side_effect = lambda **_: FakeBatch(stored)
    cloudwatch = mock.Mock()
    cloudwatch.put_metric_data.side_effect = lambda **kw: metrics.extend(kw["MetricData"])
    with mock.patch("boto3.resource") as res, mock.patch("boto3.client", return_value=cloudwatch):
        res.return_value.Table.return_value = table
        sys.modules.pop("lambda_function", None)
        module = importlib.import_module("lambda_function")
    module.stored, module.metrics = stored, metrics
    return module


def post(handler, body):
    return handler.lambda_handler({"httpMethod": "POST", "body": json.dumps(body)}, None)


PRED = {"device_id": "edge-1", "ts": 1, "label": "WALKING", "confidence": 0.93,
        "latency_ms": 0.2, "model_version": "har-cnn-int8-v1", "true_label": "WALKING"}


def test_batch_is_stored_and_metrics_emitted(handler):
    res = post(handler, {"predictions": [PRED, {**PRED, "ts": 2, "label": "SITTING"}]})
    assert res["statusCode"] == 200
    assert json.loads(res["body"]) == {"stored": 2}
    assert [i["label"] for i in handler.stored] == ["WALKING", "SITTING"]
    names = [m["MetricName"] for m in handler.metrics]
    assert names.count("Confidence") == 2 and names.count("Correct") == 2


def test_single_prediction_body_is_accepted(handler):
    assert post(handler, PRED)["statusCode"] == 200


@pytest.mark.parametrize("bad", [
    {**PRED, "label": "RUNNING"},
    {**PRED, "confidence": 1.5},
    {k: v for k, v in PRED.items() if k != "device_id"},
])
def test_invalid_predictions_rejected(handler, bad):
    assert post(handler, {"predictions": [bad]})["statusCode"] == 400
    assert handler.stored == []


def test_batch_size_limit(handler):
    assert post(handler, {"predictions": [PRED] * 101})["statusCode"] == 400
