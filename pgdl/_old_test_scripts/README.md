# Old Scripts (Moved for consolidation)

These scripts were consolidated into the two main entry points:

| Old Script | Consolidated Into |
|---|---|
| `run_all_tests.py` | `run_tests.py` (overall test runner) |
| `setup_musique_data.py` | `setup_data.py` (data initialization) |
| `setup_musique_20.py` | `setup_data.py` (data initialization, --musique-20 flag) |
| `setup_reasoning_data.py` | `setup_data.py` (data initialization) |

The individual test files (test_*.py, benchmark_*.py, etc.) are preserved
as they contain the actual test logic called by `run_tests.py`.
