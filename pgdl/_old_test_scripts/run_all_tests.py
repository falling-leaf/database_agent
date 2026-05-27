#!/usr/bin/env python3
"""
Comprehensive Test Runner for PGDL/MorphingDB.

Runs all test scripts in pgdl/test/ with timeouts, captures pass/fail,
and produces a summary report.

Usage:
    uv run run_all_tests.py [--quick]

Modes:
    --quick   Only run fast tests (sample_query, test_single with reduced scope)
    (default) Run all tests with reasonable timeouts
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all test scripts with their timeout and description
ALL_TESTS = [
    {
        "script": "sample_query.py",
        "timeout": 300,
        "description": "Sample query execution (series/slice_test)",
        "heavy": False,
    },
    {
        "script": "test_single.py",
        "timeout": 600,
        "description": "Single-thread latency benchmark (predict_batch_float8 + db_agent_single, series only)",
        "heavy": True,
    },
    {
        "script": "test_skew.py",
        "timeout": 900,
        "description": "Skew dataset performance test",
        "heavy": True,
    },
    {
        "script": "test_concurrency.py",
        "timeout": 1800,
        "description": "Concurrency stress test (predict_batch_float8, 9 datasets)",
        "heavy": True,
    },
    {
        "script": "test_db_agent_concurrency.py",
        "timeout": 1800,
        "description": "DB agent concurrency test",
        "heavy": True,
    },
    {
        "script": "test_evadb_concurrency.py",
        "timeout": 1800,
        "description": "EVA-DB concurrency test",
        "heavy": True,
    },
    {
        "script": "run_tests.py",
        "timeout": 300,
        "description": "SST2 all test (run_tests.py)",
        "heavy": False,
    },
    {
        "script": "benchmark_musique.py",
        "timeout": 600,
        "description": "Musique benchmark",
        "heavy": True,
    },
]

QUICK_TESTS = [
    {
        "script": "sample_query.py",
        "timeout": 300,
        "description": "Sample query execution (series/slice_test)",
        "heavy": False,
    },
]


def run_test(test_info, quick_mode=False):
    """Run a single test script with timeout. Returns result dict."""
    script = test_info["script"]
    script_path = os.path.join(TEST_DIR, script)
    
    if not os.path.exists(script_path):
        return {
            "script": script,
            "description": test_info["description"],
            "status": "skipped",
            "reason": "File not found",
        }
    
    if quick_mode and test_info.get("heavy", False):
        return {
            "script": script,
            "description": test_info["description"],
            "status": "skipped",
            "reason": "Skipped in quick mode (heavy test)",
        }
    
    print(f"\n{'='*80}")
    print(f"TEST: {script}")
    print(f"  Description: {test_info['description']}")
    print(f"  Timeout: {test_info['timeout']}s")
    print(f"  Heavy: {test_info.get('heavy', False)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ["uv", "run", "python", script],
            cwd=TEST_DIR,
            capture_output=True,
            text=True,
            timeout=test_info["timeout"],
        )
        elapsed = time.time() - start_time
        
        stdout_tail = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        stderr_tail = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr
        
        if result.returncode == 0:
            status = "PASSED"
        else:
            status = "FAILED"
        
        return {
            "script": script,
            "description": test_info["description"],
            "status": status,
            "returncode": result.returncode,
            "elapsed": round(elapsed, 2),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "script": script,
            "description": test_info["description"],
            "status": "TIMEOUT",
            "elapsed": round(elapsed, 2),
            "reason": f"Exceeded {test_info['timeout']}s timeout",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "script": script,
            "description": test_info["description"],
            "status": "ERROR",
            "elapsed": round(elapsed, 2),
            "reason": str(e),
        }


def main():
    quick_mode = "--quick" in sys.argv
    
    print("="*80)
    print("PGDL/MorphingDB Comprehensive Test Runner")
    print(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Working directory: {TEST_DIR}")
    print("="*80)
    
    tests = QUICK_TESTS if quick_mode else ALL_TESTS
    
    results = []
    for test_info in tests:
        result = run_test(test_info, quick_mode)
        results.append(result)
    
    # Print summary
    print("\n\n")
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    timeout = 0
    skipped = 0
    error = 0
    
    for r in results:
        status = r["status"]
        elapsed = r.get("elapsed", "N/A")
        script = r["script"]
        desc = r["description"]
        
        if status == "PASSED":
            passed += 1
            print(f"  [PASS]   {elapsed:>8}s  {script} - {desc}")
        elif status == "FAILED":
            failed += 1
            print(f"  [FAIL]   {elapsed:>8}s  {script} - {desc}")
            if r.get("stderr_tail"):
                print(f"           stderr: {r['stderr_tail'][:200]}")
        elif status == "TIMEOUT":
            timeout += 1
            print(f"  [TIMEOUT]{elapsed:>8}s  {script} - {desc}")
        elif status == "SKIPPED":
            skipped += 1
            print(f"  [SKIP]   {'N/A':>8}s  {script} - {r.get('reason', '')}")
        else:
            error += 1
            print(f"  [ERROR]  {elapsed:>8}s  {script} - {desc}")
            if r.get("reason"):
                print(f"           {r['reason']}")
    
    total = len(results)
    print("-"*80)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Timeout: {timeout} | Skipped: {skipped} | Error: {error}")
    print("-"*80)
    
    # Save results
    results_dir = os.path.join(TEST_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"test_report_{ts}.json")
    
    # Clean up large outputs for JSON
    clean_results = []
    for r in results:
        cr = dict(r)
        cr.pop("stdout_tail", None)
        cr.pop("stderr_tail", None)
        clean_results.append(cr)
    
    with open(report_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "mode": "quick" if quick_mode else "full", "results": clean_results}, f, indent=2)
    
    print(f"\nFull report saved to: {report_path}")
    
    if failed > 0 or error > 0:
        print("\nSome tests FAILED or had ERRORS. Review the report above.")
        sys.exit(1)
    elif timeout > 0:
        print("\nSome tests TIMED OUT. They may need more time or indicate performance issues.")
        sys.exit(0)
    else:
        print("\nAll tests PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
