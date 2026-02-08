#!/usr/bin/env python3
"""
Performance test for ToxTransformer service using training data samples.
"""
import requests
import time
import sqlite3
import random
import concurrent.futures
from urllib.parse import quote
import statistics
import sys

API_URL = "http://136.111.102.10:6515"
DB_PATH = "cache/build_sqlite/cvae.sqlite"

def get_sample_inchis(n=100):
    """Get random sample of InChIs from training data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get distinct InChIs
    cursor.execute("""
        SELECT DISTINCT inchi FROM activity
        WHERE inchi IS NOT NULL AND length(inchi) < 500
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,))

    inchis = [row[0] for row in cursor.fetchall()]
    conn.close()
    return inchis

def predict_single(inchi, timeout=120):
    """Make a single prediction request."""
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
                "inchi": inchi[:50]
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "time": elapsed,
                "inchi": inchi[:50]
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start,
            "inchi": inchi[:50]
        }

def run_sequential_test(inchis, max_requests=20):
    """Run sequential requests."""
    print(f"\n=== Sequential Test ({min(len(inchis), max_requests)} requests) ===")

    results = []
    for i, inchi in enumerate(inchis[:max_requests]):
        print(f"  Request {i+1}/{max_requests}...", end=" ", flush=True)
        result = predict_single(inchi)
        results.append(result)

        if result["success"]:
            print(f"OK - {result['predictions']} predictions in {result['time']:.2f}s")
        else:
            print(f"FAILED - {result['error']}")

    return results

def run_concurrent_test(inchis, max_workers=5, max_requests=20):
    """Run concurrent requests."""
    print(f"\n=== Concurrent Test ({min(len(inchis), max_requests)} requests, {max_workers} workers) ===")

    start = time.time()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict_single, inchi): inchi
                   for inchi in inchis[:max_requests]}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result["success"] else "FAILED"
            print(f"  Completed {i+1}/{max_requests}: {status}")

    total_time = time.time() - start
    print(f"  Total wall time: {total_time:.2f}s")

    return results

def analyze_results(results, test_name="Test"):
    """Analyze and print results summary."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n=== {test_name} Summary ===")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        times = [r["time"] for r in successful]
        predictions = [r["predictions"] for r in successful]

        print(f"\n  Response times:")
        print(f"    Min: {min(times):.3f}s")
        print(f"    Max: {max(times):.3f}s")
        print(f"    Mean: {statistics.mean(times):.3f}s")
        print(f"    Median: {statistics.median(times):.3f}s")
        if len(times) > 1:
            print(f"    Std Dev: {statistics.stdev(times):.3f}s")

        print(f"\n  Predictions per request:")
        print(f"    Min: {min(predictions)}")
        print(f"    Max: {max(predictions)}")
        print(f"    Mean: {statistics.mean(predictions):.1f}")

        # Throughput
        total_predictions = sum(predictions)
        total_time = sum(times)
        print(f"\n  Throughput:")
        print(f"    Total predictions: {total_predictions:,}")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Predictions/sec: {total_predictions/total_time:.1f}")

    if failed:
        print(f"\n  Failures:")
        for r in failed[:5]:
            print(f"    {r['error']}")

def main():
    print("ToxTransformer Performance Test")
    print("=" * 50)

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

    # Get sample data
    print("\nLoading sample InChIs from training data...")
    inchis = get_sample_inchis(50)
    print(f"  Loaded {len(inchis)} unique InChIs")

    # Sequential test
    seq_results = run_sequential_test(inchis, max_requests=10)
    analyze_results(seq_results, "Sequential")

    # Concurrent test (low concurrency - service has lock)
    conc_results = run_concurrent_test(inchis, max_workers=2, max_requests=10)
    analyze_results(conc_results, "Concurrent (2 workers)")

    print("\n" + "=" * 50)
    print("Performance test complete!")

if __name__ == "__main__":
    main()
