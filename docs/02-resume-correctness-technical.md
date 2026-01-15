# Resume Correctness: Technical Deep Dive

This document explains the resume correctness system in detail — why it exists, how it works, and what guarantees it provides.

---

## The Problem

### What Went Wrong

The original Monte Carlo runner had a critical bug that manifested during resume after crashes:

**Symptom**: Cell showed `n_perms_done: 290000` when `n_perms_target: 200000`.

**Root cause**: The resume logic read `n_done` from `progress.json`, which could get out of sync with the actual data in `metrics_compact.csv`.

**Scenario**:
1. Run starts, processes permutations 0-110,000
2. Crash occurs
3. `progress.json` says `n_done: 110000`
4. Resume starts, begins at `start_idx = 110000`
5. But `metrics_compact.csv` only has 100,000 rows (some writes didn't flush)
6. Permutations 100,000-110,000 are now **duplicated**
7. Run continues, finishes at 200,000
8. CSV has 210,000 rows, but only 200,000 unique perm_index values

**Impact**: Summary statistics are computed from 210,000 rows, including 10,000 duplicates. This biases the estimates and violates the "exactly N permutations" contract.

---

## The Solution

### Source of Truth Principle

**The `metrics_compact.csv` file is the ONLY source of truth.**

`progress.json` is advisory only — it's useful for quick status checks but never trusted for resume decisions.

### Resume Algorithm

```python
def read_metrics_and_dedupe(metrics_path):
    # Step 1: Load CSV
    df = pd.read_csv(metrics_path)
    
    # Step 2: Dedupe by perm_index, keep first occurrence
    df_deduped = df.drop_duplicates(subset=["perm_index"], keep="first")
    
    # Step 3: Count unique indices
    n_done = len(df_deduped["perm_index"].unique())
    
    # Step 4: Find max perm_index for resume position
    max_perm_index = df_deduped["perm_index"].max()
    
    # Step 5: If dedupe was needed, rewrite CSV atomically
    if n_duplicates > 0:
        atomic_write_csv(metrics_path, df_deduped)
    
    return df_deduped, n_done, max_perm_index
```

### Key Insight: Start Position

**Resume starts at `max(perm_index) + 1`, not at `n_done`.**

Why? Consider this scenario:
- CSV has perm_index values: [0, 1, 2, 5, 6, 7, 8, 9] (indices 3, 4 missing)
- `n_done` = 8 (count of unique values)
- `max(perm_index)` = 9
- If we resume at `n_done=8`, we'd generate perm_index 8, which already exists!
- Correct resume position: `max + 1 = 10`

---

## Atomic Write Pattern

All file writes use the atomic replace pattern to prevent corruption:

```python
def atomic_write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)  # Atomic on most filesystems
    except Exception:
        if tmp.exists():
            tmp.unlink()  # Clean up on failure
        raise
```

### Why This Works

On Windows (NTFS) and Linux (ext4), `Path.replace()` is atomic at the filesystem level:
- Either the entire new file appears, or it doesn't
- No partial writes visible
- Safe against power loss/crash

### What About CSV Appends?

For metrics_compact.csv, we append in chunks:

```python
df_chunk.to_csv(metrics_path, mode="a", header=write_header, index=False)
```

This isn't atomic, but it's okay because:
1. We dedupe on resume anyway
2. Partial writes just mean some rows missing
3. We'll regenerate those permutations on resume

---

## Dedupe Rules

### Rule 1: Keep First Occurrence

When duplicates exist, we keep the **first** occurrence and discard later ones.

Why first?
- First write happened before the crash/issue
- Later writes are the duplicates from incorrect resume
- First is more likely to be "correct" in a temporal sense

### Rule 2: Sort by perm_index for Verification

When writing summary, we verify ordering:

```python
df_final = df.drop_duplicates(subset=["perm_index"], keep="first")
df_final = df_final.sort_values("perm_index")
```

This ensures the CSV is in canonical order, making debugging easier.

### Rule 3: Truncate if Exceeds Target

If we somehow have more than `n_target` unique permutations:

```python
if n_done > n_target:
    df_final = df_final.sort_values("perm_index").head(n_target)
    atomic_write_csv(metrics_path, df_final)
```

We keep the first `n_target` by perm_index order.

---

## Seeding Scheme

Deterministic seeding is critical for reproducibility:

### Cell Seed Derivation

```python
def seed_base_for_cell(global_seed, cell_id, seed_stride):
    # Hash cell_id to get offset
    payload = cell_id.encode("utf-8")
    offset = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    
    # Mod by stride to keep in reasonable range
    if seed_stride > 0:
        offset = offset % seed_stride
    
    # Combine with global seed
    return (global_seed + offset) & 0xFFFFFFFF
```

### Per-Permutation Seed

```python
def seed_for_sim(base_seed, perm_index):
    return (base_seed + perm_index * 1000003) & 0xFFFFFFFF
```

The multiplier 1000003 is a prime that spreads seeds well.

### Why This Matters for Resume

Same cell_id + same perm_index = same seed = same random draws = same simulation result.

This means:
- If a permutation is accidentally re-run, it produces identical data
- Deduping by perm_index catches this correctly
- Results are reproducible across runs

---

## Summary Statistics Verification

### What We Track

In `summary.json`:

```json
{
  "n_perms_target": 200000,
  "n_perms_done": 200000,
  "n_duplicates_dropped": 1523,
  "n_rows_raw": 201523,
  "n_rows_deduped": 200000,
  "resume_detected": true
}
```

### Invariants

1. `n_perms_done == n_perms_target` when complete
2. `n_rows_deduped == n_perms_done` always
3. `n_rows_raw >= n_rows_deduped` always
4. `n_duplicates_dropped == n_rows_raw - n_rows_deduped`

### Verification in Analysis

The analysis script verifies these:

```python
def verify_cell(cell_dir, n_target):
    df = pd.read_csv(cell_dir / "metrics_compact.csv")
    df_deduped = df.drop_duplicates(subset=["perm_index"], keep="first")
    
    assert len(df_deduped) == n_target, "Wrong count"
    assert df_deduped["perm_index"].min() == 0, "Missing start"
    assert df_deduped["perm_index"].max() == n_target - 1, "Missing end"
```

---

## Progress File Format

### progress.json (Advisory)

```json
{
  "cell_id": "3_2_0_1_1",
  "cell_index": 847,
  "phase": "full_200k",
  "n_done": 150000,
  "n_target": 200000,
  "pct_done": 75.0,
  "runtime_seconds": 1234.5,
  "updated_at": "2026-01-15T14:30:00",
  "params": {
    "p_skip": 0.03,
    "slip_dollars": 50.0,
    "delay_bars_max": 1,
    "shuffle_mode": "permute",
    "bootstrap_mode": "trade_bootstrap",
    "block_len": 10
  },
  "resume_info": {
    "resume_detected": true,
    "n_rows_raw": 152000,
    "n_duplicates_dropped": 2000
  }
}
```

### heartbeat.json (Aggregated)

```json
{
  "timestamp": "2026-01-15T14:30:00",
  "run_name": "mc_surface_full_200k_20260115",
  "current_cell_id": "3_2_0_1_1",
  "cells_complete": 847,
  "cells_total": 1512,
  "sims_done_total_estimate": 169400000,
  "uptime_seconds": 43200
}
```

---

## Error Recovery Scenarios

### Scenario 1: Crash Mid-Chunk

**State**: CSV has 100,000 rows, progress.json says 102,000

**Recovery**:
1. Resume reads CSV: n_done = 100,000, max_perm = 99,999
2. Resume starts at perm_index = 100,000
3. Correct behavior ✓

### Scenario 2: Crash After Write, Before Progress Update

**State**: CSV has 102,000 rows, progress.json says 100,000

**Recovery**:
1. Resume reads CSV: n_done = 102,000, max_perm = 101,999
2. Resume starts at perm_index = 102,000
3. Correct behavior ✓

### Scenario 3: Duplicate Perm Indices

**State**: CSV has duplicate rows for perm_index 50,000-51,999

**Recovery**:
1. Resume dedupes: keeps first occurrence of each
2. Rewrites CSV atomically
3. Continues from max_perm + 1
4. Correct behavior ✓

### Scenario 4: Corrupted CSV

**State**: CSV is truncated/corrupted, can't be read

**Recovery**:
1. pd.read_csv raises exception
2. Caught, returns n_done=0, max_perm=-1
3. Resume starts fresh from perm_index = 0
4. Loses progress but data integrity maintained ✓

---

## Testing Resume Correctness

### Manual Test

1. Start a run
2. Wait for some progress
3. Kill the process (Ctrl+C or Task Manager)
4. Restart with same run_name
5. Verify:
   - No "0/N complete" restart messages
   - n_done continues from where it left off
   - Final n_perms_done == n_perms_target exactly

### Automated Verification

```python
def verify_run_integrity(run_dir):
    per_cell_dir = run_dir / "per_cell"
    
    for cell_dir in per_cell_dir.iterdir():
        metrics_path = cell_dir / "metrics_compact.csv"
        summary_path = cell_dir / "summary.json"
        
        if not summary_path.exists():
            continue  # Incomplete cell
            
        df = pd.read_csv(metrics_path)
        summary = json.loads(summary_path.read_text())
        
        # Check count
        assert len(df) == summary["n_perms_done"]
        
        # Check no duplicates remain
        assert df["perm_index"].is_unique
        
        # Check range
        assert df["perm_index"].min() == 0
        assert df["perm_index"].max() == summary["n_perms_done"] - 1
```

---

## Summary

| Guarantee | Implementation |
|-----------|----------------|
| Source of truth | metrics_compact.csv only |
| Dedupe policy | Keep first occurrence |
| Resume position | max(perm_index) + 1 |
| Atomic writes | tmp + replace pattern |
| Deterministic | SHA256 seeding |
| Verified | Count + range + uniqueness checks |

The system is designed to be **crash-safe** and **self-healing**. Any inconsistency introduced by crashes is automatically detected and corrected on the next run.
