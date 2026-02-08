#!/usr/bin/env python3
"""
Comprehensive performance test for ToxTransformer service.
Tests both predict_all and predict (single property) endpoints.
"""
import requests
import time
import sqlite3
import random
import concurrent.futures
from urllib.parse import quote
import statistics
import sys
import json

API_URL = "http://136.111.102.10:6515"
DB_PATH = "cache/build_sqlite/cvae.sqlite"

def get_test_samples(n=50):
    """Get random samples of InChIs with known property values."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get samples with their property tokens and expected values
    cursor.execute("""
        SELECT DISTINCT a.inchi, a.property_token, a.value, p.title
        FROM activity a
        JOIN property p ON a.property_id = p.property_id
        WHERE a.inchi IS NOT NULL
          AND length(a.inchi) < 300
          AND a.property_token IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (n * 3,))  # Get more samples than needed

    samples = []
    seen_inchis = set()
    for row in cursor.fetchall():
        inchi, prop_token, expected_value, title = row
        if inchi not in seen_inchis and len(samples) < n:
            samples.append({
                "inchi": inchi,
                "property_token": int(prop_token),
                "expected_value": expected_value,
                "title": title
            })
            seen_inchis.add(inchi)

    conn.close()
    return samples

def predict_all(inchi, timeout=120):
    """Test predict_all endpoint."""
    start = time.time()
    try:
        url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "predictions": len(data),
                "time": elapsed,
                "data": data
            }
        elif response.status_code == 503:
            return {"success": False, "error": "Server busy (503)", "time": elapsed}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "time": elapsed}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout", "time": time.time() - start}
    except Exception as e:
        return {"success": False, "error": str(e), "time": time.time() - start}

def predict_single(inchi, property_token, timeout=30):
    """Test predict endpoint (single property)."""
    start = time.time()
    try:
        url = f"{API_URL}/predict?inchi={quote(inchi, safe='')}&property_token={property_token}"
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start

        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "predicted_value": data.get("value"),
                "time": elapsed,
                "data": data
            }
        elif response.status_code == 503:
            return {"success": False, "error": "Server busy (503)", "time": elapsed}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}", "time": elapsed}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Timeout", "time": time.time() - start}
    except Exception as e:
        return {"success": False, "error": str(e), "time": time.time() - start}

def test_predict_all_accuracy(samples):
    """Test predict_all endpoint and check if predictions match expected values."""
    print("\n=== Predict All - Accuracy Test ===")

    results = []
    correct = 0
    total_checked = 0

    for i, sample in enumerate(samples[:20]):
        print(f"  Testing {i+1}/20...", end=" ", flush=True)
        result = predict_all(sample["inchi"])

        if result["success"]:
            # Find the prediction for this property token
            pred_value = None
            for pred in result["data"]:
                if pred.get("property_token") == sample["property_token"]:
                    pred_value = pred.get("value")
                    break

            if pred_value is not None:
                # Check if prediction matches expected (binary classification)
                pred_binary = 1 if pred_value >= 0.5 else 0
                expected_binary = sample["expected_value"]
                is_correct = pred_binary == expected_binary
                if is_correct:
                    correct += 1
                total_checked += 1

                results.append({
                    "success": True,
                    "time": result["time"],
                    "predicted": pred_value,
                    "expected": expected_binary,
                    "correct": is_correct
                })
                status = "MATCH" if is_correct else "MISMATCH"
                print(f"OK ({result['time']:.2f}s) - pred={pred_value:.3f}, exp={expected_binary}, {status}")
            else:
                results.append({"success": True, "time": result["time"], "no_match": True})
                print(f"OK ({result['time']:.2f}s) - property not found")
        else:
            results.append({"success": False, "error": result["error"], "time": result["time"]})
            print(f"FAILED - {result['error']}")

    if total_checked > 0:
        accuracy = correct / total_checked * 100
        print(f"\n  Accuracy: {correct}/{total_checked} ({accuracy:.1f}%)")

    return results

def test_predict_single_performance(samples):
    """Test single property prediction performance."""
    print("\n=== Single Property Prediction Performance ===")

    results = []
    for i, sample in enumerate(samples[:30]):
        print(f"  Testing {i+1}/30...", end=" ", flush=True)
        result = predict_single(sample["inchi"], sample["property_token"])

        if result["success"]:
            results.append(result)
            print(f"OK ({result['time']:.3f}s) - value={result['predicted_value']:.3f}")
        else:
            results.append(result)
            print(f"FAILED - {result['error']}")

    # Summary
    successful = [r for r in results if r["success"]]
    if successful:
        times = [r["time"] for r in successful]
        print(f"\n  Summary:")
        print(f"    Successful: {len(successful)}/{len(results)}")
        print(f"    Response times: min={min(times):.3f}s, max={max(times):.3f}s, mean={statistics.mean(times):.3f}s")

    return results

def test_concurrent_load(samples, workers=3, requests_per_worker=5):
    """Test concurrent load handling."""
    print(f"\n=== Concurrent Load Test ({workers} workers, {requests_per_worker} each) ===")

    total_requests = workers * requests_per_worker
    test_samples = samples[:total_requests]

    start = time.time()
    results = []

    def worker_task(sample):
        return predict_all(sample["inchi"], timeout=180)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_task, s) for s in test_samples]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result["success"] else f"FAILED: {result.get('error', 'unknown')}"
            print(f"  {i+1}/{total_requests}: {status} ({result['time']:.2f}s)")

    total_time = time.time() - start

    # Summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n  Summary:")
    print(f"    Total wall time: {total_time:.2f}s")
    print(f"    Successful: {len(successful)}/{len(results)}")
    print(f"    Failed: {len(failed)}")

    if successful:
        times = [r["time"] for r in successful]
        preds = [r["predictions"] for r in successful]
        print(f"    Response times: min={min(times):.2f}s, max={max(times):.2f}s, mean={statistics.mean(times):.2f}s")
        print(f"    Throughput: {sum(preds)/total_time:.1f} predictions/sec")

    if failed:
        error_types = {}
        for r in failed:
            err = r.get("error", "unknown")
            error_types[err] = error_types.get(err, 0) + 1
        print(f"    Error breakdown: {error_types}")

    return results

def main():
    print("ToxTransformer Comprehensive Performance Test")
    print("=" * 60)

    # Check service health
    print("\nChecking service health...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code == 200:
            print(f"  Service healthy: {resp.json()}")
        else:
            print(f"  Service unhealthy: HTTP {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"  Service unreachable: {e}")
        sys.exit(1)

    # Get test data
    print("\nLoading test samples from training data...")
    samples = get_test_samples(50)
    print(f"  Loaded {len(samples)} test samples with known values")

    # Test 1: Accuracy
    test_predict_all_accuracy(samples)

    # Test 2: Single property performance
    test_predict_single_performance(samples)

    # Test 3: Concurrent load
    test_concurrent_load(samples, workers=3, requests_per_worker=5)

    print("\n" + "=" * 60)
    print("Comprehensive test complete!")

if __name__ == "__main__":
    main()
