# Changelog

## 2026-01-16

### Fixed: Worker hang on completion

**Problem:** The 200k grid runner would hang indefinitely when worker processes completed their metrics but crashed before writing `summary.json`. The main process waited forever on `as_completed(futures)` with no timeout.

**Solution:**
1. Added 10-minute timeout to `future.result()` calls
2. Added final sweep that detects orphaned cells (complete metrics, missing summary) and regenerates summaries from the raw data

**Files changed:**
- `CURSOR_run_surface_full_200k.py`

**Patch:**
```python
# Before (line 1265):
status_msg, is_complete, n_done = future.result()

# After:
status_msg, is_complete, n_done = future.result(timeout=600)
```

Plus added ~40 lines of orphan sweep logic after the main executor loop.
