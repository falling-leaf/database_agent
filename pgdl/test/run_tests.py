#!/usr/bin/env python3
"""
Overall Test Runner for PGDL/MorphingDB.

Runs all non-concurrency test scripts with timeouts, captures pass/fail,
and produces a summary report. Concurrency tests are excluded by design
-- use dedicated concurrency scripts for those.

Usage:
    uv run python run_tests.py                        # run all non-concurrency tests
    uv run python run_tests.py --quick                 # run only fast tests
    uv run python run_tests.py --list                  # list all available tests
    uv run python run_tests.py sample_query test_skew  # run specific tests

Categories:
    fast      sample_query
    medium    test_reasoning_sequential, test_reasoning_samples, test_sql_timing,
              test_evadb, verify_dbagent_extra_models, verify_inference_time
    heavy     test_single, test_skew, benchmark_musique, benchmark_musique_linear,
              benchmark_musique_fair, benchmark_reasoning, test_reasoning_independent,
              test_reasoning_python_bench
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Test definitions
# =============================================================================
ALL_TESTS = [
    # --- Fast tests ---
    {
        "name": "sample_query",
        "script": "sample_query.py",
        "timeout": 300,
        "category": "fast",
        "description": "Sample query execution (series/slice_test)",
    },
    # --- Medium tests ---
    {
        "name": "test_reasoning_sequential",
        "script": "test_reasoning_sequential.py",
        "timeout": 600,
        "category": "medium",
        "description": "Sequential reasoning benchmark (1/5/10/20 queries)",
    },
    {
        "name": "test_reasoning_samples",
        "script": "test_reasoning_samples.py",
        "timeout": 600,
        "category": "medium",
        "description": "Multi-sample reasoning benchmark (n_samples=1/5/10/20)",
    },
    {
        "name": "test_sql_timing",
        "script": "test_sql_timing.py",
        "timeout": 600,
        "category": "medium",
        "description": "SQL timing test",
    },
    {
        "name": "test_evadb",
        "script": "test_evadb.py",
        "timeout": 600,
        "category": "medium",
        "description": "EVA-DB basic test",
    },
    {
        "name": "verify_dbagent_extra_models",
        "script": "verify_dbagent_extra_models.py",
        "timeout": 600,
        "category": "medium",
        "description": "Verify dbagent extra model calls",
    },
    {
        "name": "verify_inference_time",
        "script": "verify_inference_time.py",
        "timeout": 600,
        "category": "medium",
        "description": "Verify inference time measurements",
    },
    # --- Heavy tests ---
    {
        "name": "test_single",
        "script": "test_single.py",
        "timeout": 1200,
        "category": "heavy",
        "description": "Single-thread latency benchmark (predict_batch_float8 + db_agent_single)",
    },
    {
        "name": "test_skew",
        "script": "test_skew.py",
        "timeout": 1800,
        "category": "heavy",
        "description": "Skew dataset performance test (3 tables x 6 ratios x 3 query_times)",
    },
    {
        "name": "benchmark_musique",
        "script": "benchmark_musique.py",
        "timeout": 1200,
        "category": "heavy",
        "description": "Musique benchmark: db_agent_single vs pure Python pipeline",
    },
    {
        "name": "benchmark_musique_linear",
        "script": "benchmark_musique_linear.py",
        "timeout": 1800,
        "category": "heavy",
        "description": "Fair musique benchmark with linear scaling",
    },
    {
        "name": "benchmark_musique_fair",
        "script": "benchmark_musique_fair.py",
        "timeout": 1800,
        "category": "heavy",
        "description": "Fair musique benchmark: dbagent vs Python baselines",
    },
    {
        "name": "benchmark_reasoning",
        "script": "benchmark_reasoning.py",
        "timeout": 3600,
        "category": "heavy",
        "description": "Fair reasoning benchmark: dbagent vs Python/Hermes/NeurDB/GenDB",
    },
    {
        "name": "test_reasoning_independent",
        "script": "test_reasoning_independent.py",
        "timeout": 1800,
        "category": "heavy",
        "description": "Independent reasoning benchmark (n queries sequential)",
    },
    {
        "name": "test_reasoning_python_bench",
        "script": "test_reasoning_python_bench.py",
        "timeout": 1200,
        "category": "heavy",
        "description": "Python reasoning benchmark matching dbagent batch sizes",
    },
]


def get_test_by_name(name):
    """Find a test by its name (exact match or partial match on script name)."""
    for t in ALL_TESTS:
        if t["name"] == name or t["script"] == name or t["script"].replace(".py", "") == name:
            return t
    return None


def list_tests():
    """Print all available tests grouped by category."""
    categories = {"fast": [], "medium": [], "heavy": []}
    for t in ALL_TESTS:
        categories[t["category"]].append(t)

    print("\nAvailable tests (non-concurrency):")
    print("=" * 80)
    for cat in ["fast", "medium", "heavy"]:
        print(f"\n  [{cat.upper()}]")
        for t in categories[cat]:
            print(f"    {t['name']:<40} {t['description']}")
            print(f"    {'':<40} timeout={t['timeout']}s, script={t['script']}")
    print()


def run_single_test(test_info):
    """Run a single test script with timeout. Returns result dict."""
    script = test_info["script"]
    script_path = os.path.join(TEST_DIR, script)

    if not os.path.exists(script_path):
        return {
            "name": test_info["name"],
            "script": script,
            "description": test_info["description"],
            "category": test_info["category"],
            "status": "skipped",
            "reason": "File not found",
        }

    print(f"\n{'=' * 80}")
    print(f"TEST: {test_info['name']}")
    print(f"  Description: {test_info['description']}")
    print(f"  Category:    {test_info['category']}")
    print(f"  Timeout:     {test_info['timeout']}s")
    print(f"  Script:      {script}")
    print(f"{'=' * 80}")

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

        # Capture last 1500 chars of output for debugging
        stdout_tail = result.stdout[-1500:] if result.stdout else ""
        stderr_tail = result.stderr[-1500:] if result.stderr else ""

        status = "passed" if result.returncode == 0 else "failed"

        return {
            "name": test_info["name"],
            "script": script,
            "description": test_info["description"],
            "category": test_info["category"],
            "status": status,
            "returncode": result.returncode,
            "elapsed": round(elapsed, 2),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "name": test_info["name"],
            "script": script,
            "description": test_info["description"],
            "category": test_info["category"],
            "status": "timeout",
            "elapsed": round(elapsed, 2),
            "reason": f"Exceeded {test_info['timeout']}s timeout",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "name": test_info["name"],
            "script": script,
            "description": test_info["description"],
            "category": test_info["category"],
            "status": "error",
            "elapsed": round(elapsed, 2),
            "reason": str(e),
        }


def main():
    # --- Parse arguments ---
    args = sys.argv[1:]

    if "--list" in args:
        list_tests()
        sys.exit(0)

    quick_mode = "--quick" in args
    if "--quick" in args:
        args.remove("--quick")

    # Specific tests requested?
    if args:
        selected_tests = []
        for arg in args:
            t = get_test_by_name(arg)
            if t:
                selected_tests.append(t)
            else:
                print(f"WARNING: Unknown test '{arg}', skipping.")
        if not selected_tests:
            print("No valid tests specified. Use --list to see available tests.")
            sys.exit(1)
    elif quick_mode:
        selected_tests = [t for t in ALL_TESTS if t["category"] == "fast"]
    else:
        selected_tests = ALL_TESTS

    # --- Run tests ---
    print("=" * 80)
    print("PGDL/MorphingDB Overall Test Runner")
    print(f"Mode:  {'QUICK (fast only)' if quick_mode else 'FULL (all non-concurrency)'}")
    print(f"Tests: {len(selected_tests)} selected")
    print(f"Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dir:   {TEST_DIR}")
    print("=" * 80)

    results = []
    for test_info in selected_tests:
        result = run_single_test(test_info)
        results.append(result)

    # --- Print summary ---
    print("\n\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = failed = timeout = skipped = error = 0
    for r in results:
        status = r["status"]
        elapsed = r.get("elapsed", "N/A")
        name = r["name"]
        desc = r["description"]
        cat = r.get("category", "")

        if status == "passed":
            passed += 1
            print(f"  [PASS]   {elapsed:>8}s  [{cat:<7}] {name} - {desc}")
        elif status == "failed":
            failed += 1
            print(f"  [FAIL]   {elapsed:>8}s  [{cat:<7}] {name} - {desc}")
            if r.get("stderr_tail"):
                print(f"           stderr: {r['stderr_tail'][:200]}")
        elif status == "timeout":
            timeout += 1
            print(f"  [TIMEOUT]{elapsed:>8}s  [{cat:<7}] {name} - {desc}")
        elif status == "skipped":
            skipped += 1
            print(f"  [SKIP]   {'N/A':>8}s  [{cat:<7}] {name} - {r.get('reason', '')}")
        else:
            error += 1
            print(f"  [ERROR]  {elapsed:>8}s  [{cat:<7}] {name} - {desc}")
            if r.get("reason"):
                print(f"           {r['reason']}")

    total = len(results)
    print("-" * 80)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | "
          f"Timeout: {timeout} | Skipped: {skipped} | Error: {error}")
    print("-" * 80)

    # --- Save report ---
    results_dir = os.path.join(TEST_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"test_report_{ts}.json")

    clean_results = []
    for r in results:
        cr = dict(r)
        cr.pop("stdout_tail", None)
        cr.pop("stderr_tail", None)
        clean_results.append(cr)

    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "quick" if quick_mode else "full",
            "total": total,
            "passed": passed,
            "failed": failed,
            "timeout": timeout,
            "skipped": skipped,
            "error": error,
            "results": clean_results,
        }, f, indent=2)

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
