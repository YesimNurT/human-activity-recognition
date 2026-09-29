"""Monitoring API for edge HAR predictions.

POST /predictions  body: {"predictions": [ {device_id, ts, label, confidence,
                         latency_ms, model_version, true_label?}, ... ]}  (max 100)
GET  /predictions?device_id=...&limit=50   most recent predictions of a device

Each prediction is stored in DynamoDB (30-day TTL) and summarized as
CloudWatch metrics in the EdgeHAR namespace.
"""
import json
import os
import time
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

LABELS = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"}
MAX_BATCH = 100
TTL_SECONDS = 30 * 24 * 3600

table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
cloudwatch = boto3.client("cloudwatch")


def response(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, default=str)}


def validate(p):
    if not isinstance(p, dict):
        return "prediction must be an object"
    for field in ("device_id", "label", "confidence", "latency_ms"):
        if field not in p:
            return f"missing field '{field}'"
    if p["label"] not in LABELS:
        return f"unknown label '{p['label']}'"
    if not 0 <= float(p["confidence"]) <= 1:
        return "confidence must be in [0, 1]"
    return None


def to_item(p):
    now_ms = int(time.time() * 1000)
    item = {
        "device_id": str(p["device_id"])[:64],
        "ts": int(p.get("ts", now_ms)),
        "label": p["label"],
        "confidence": Decimal(str(p["confidence"])),
        "latency_ms": Decimal(str(p["latency_ms"])),
        "model_version": str(p.get("model_version", "unknown"))[:64],
        "received_at": now_ms,
        "expires_at": now_ms // 1000 + TTL_SECONDS,
    }
    if p.get("true_label") in LABELS:
        item["true_label"] = p["true_label"]
    return item


def put_metrics(items):
    data = []
    for it in items:
        dims = [{"Name": "ModelVersion", "Value": it["model_version"]}]
        data += [
            {"MetricName": "Confidence", "Dimensions": dims, "Value": float(it["confidence"])},
            {"MetricName": "LatencyMs", "Dimensions": dims, "Value": float(it["latency_ms"]),
             "Unit": "Milliseconds"},
            {"MetricName": "Predictions", "Dimensions": [{"Name": "Activity", "Value": it["label"]}],
             "Value": 1, "Unit": "Count"},
        ]
        if "true_label" in it:
            data.append({"MetricName": "Correct", "Dimensions": dims,
                         "Value": float(it["label"] == it["true_label"])})
    for i in range(0, len(data), 1000):  # PutMetricData limit per call
        cloudwatch.put_metric_data(Namespace="EdgeHAR", MetricData=data[i:i + 1000])


def handle_post(event):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return response(400, {"error": "body must be JSON"})
    preds = body["predictions"] if "predictions" in body else [body]
    if not preds or len(preds) > MAX_BATCH:
        return response(400, {"error": f"send between 1 and {MAX_BATCH} predictions"})
    for i, p in enumerate(preds):
        err = validate(p)
        if err:
            return response(400, {"error": f"prediction {i}: {err}"})

    items = [to_item(p) for p in preds]
    with table.batch_writer(overwrite_by_pkeys=["device_id", "ts"]) as batch:
        for it in items:
            batch.put_item(Item=it)
    put_metrics(items)
    return response(200, {"stored": len(items)})


def handle_get(event):
    params = event.get("queryStringParameters") or {}
    device_id = params.get("device_id")
    if not device_id:
        return response(400, {"error": "device_id query parameter is required"})
    limit = min(int(params.get("limit", 50)), 500)
    res = table.query(KeyConditionExpression=Key("device_id").eq(device_id),
                      ScanIndexForward=False, Limit=limit)
    return response(200, {"items": res["Items"]})


def lambda_handler(event, context):
    method = event.get("httpMethod")
    if method == "POST":
        return handle_post(event)
    if method == "GET":
        return handle_get(event)
    return response(405, {"error": "method not allowed"})
