---
phase: 14-housekeeping-tech-debt
plan: "01"
subsystem: graph
tags: [migration, schema, dbt-01, legacy, round-trip]
dependency_graph:
  requires: []
  provides: [DBT-01]
  affects: [src/automil/graph.py, tests/test_graph.py]
tech_stack:
  added: []
  patterns: [migrate-on-read, schema-version-gate, idempotent-migration]
key_files:
  created: []
  modified:
    - src/automil/graph.py
    - tests/test_graph.py
decisions:
  - "Capture _on_disk_schema_version before setdefault fills defaults — gate uses pre-setdefault value"
  - "Keep flat metric keys after migration (back-compat, no data loss risk)"
  - "Migration is in-memory only — no save() in __init__"
  - "Warning fires only when nodes actually migrated (_migrated > 0), silent for idempotent loads"
metrics:
  duration: "~10 minutes"
  completed: "2026-06-12T16:00:46Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 14 Plan 01: DBT-01 Legacy Schema Round-Trip Summary

Pre-D-200 graph.json files (flat val_auc/val_bacc/test_auc/test_bacc metric keys, no `metrics` dict) now migrate on-read through `ExperimentGraph.__init__` with no KeyError, yielding fully-populated `node["metrics"]` dicts and bumping `schema_version` to 2 in memory.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add DBT-01 RED tests (3 legacy round-trip tests) | 702cc4a | tests/test_graph.py |
| 2 | Implement DBT-01 migrate-on-read in graph.py | d7738f4 | src/automil/graph.py |

## What Was Built

### Migration block in `ExperimentGraph.__init__` (src/automil/graph.py)

Added immediately after the existing top-level/meta defaults block:

1. **`_on_disk_schema_version` capture** — reads `self._data.get("schema_version", 1)` BEFORE `setdefault` fills in the new default of 2. This is required because absent-key legacy graphs would otherwise be patched to `schema_version=2` by `setdefault` before the migration gate checks, causing them to silently skip migration.

2. **Schema version default bump** — changed `"schema_version": 1` to `"schema_version": 2` in the `defaults` dict so fresh-init graphs and newly-written graphs never re-trigger the migration gate on future loads.

3. **Per-node migration loop** — gated on `_on_disk_schema_version < 2`. For each node lacking a `"metrics"` key, spreads the 4 consumer metric keys (`val_auc`, `val_bacc`, `test_auc`, `test_bacc`) into a fresh `node["metrics"]` dict via `node.get(k, 0.0)`. Flat keys are preserved (keep-flat strategy). Sets `self._data["schema_version"] = 2` inside the gate so the in-memory graph is marked as migrated.

4. **Conditional warning** — logs at WARNING level only when `_migrated > 0` and `loaded_from_disk`, prompting the operator to re-save. Silent for already-D-200 graphs and fresh-init paths.

### 3 new tests in `tests/test_graph.py`

All 3 are top-level pytest functions (not inside a class) using real `graph.json` fixtures:

- `test_legacy_schema_round_trip` — `schema_version=1`, flat keys, no `metrics` dict; asserts migration populates all 4 keys + preserves flat keys + bumps schema_version=2
- `test_legacy_schema_absent_version` — no `schema_version` key at all; asserts migration still fires
- `test_post_d200_graph_not_remigrated` — `schema_version=2` with `metrics` dict present; asserts no re-migration, no extra keys, schema_version=2 unchanged

## Test Results

```
tests/test_graph.py: 33 passed (30 pre-existing + 3 new DBT-01)
tests/ full suite: 1 failed (pre-existing clause_11), 1057 passed, 53 skipped
```

The clause_11 failure is the pre-existing baseline failure, addressed by plan 14-02.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] setdefault ordering defeated absent-version gate**

- **Found during:** Task 2 GREEN verification
- **Issue:** After bumping the `schema_version` default from 1 to 2, the `setdefault` loop fills in `schema_version=2` for any graph file that lacks the key — BEFORE the migration gate `self._data.get("schema_version", 1) < 2` runs. This caused `test_legacy_schema_absent_version` to fail (gate saw 2, skipped migration).
- **Fix:** Captured `_on_disk_schema_version = self._data.get("schema_version", 1)` immediately before the `defaults` block. The migration gate now uses this pre-setdefault value instead of re-reading from `self._data`. The fix is correct for all three cases: absent key (reads 1, gate fires), key=1 (reads 1, gate fires), key=2 (reads 2, gate skips).
- **Files modified:** src/automil/graph.py
- **Commit:** d7738f4

## Known Stubs

None. The migration is fully wired; no placeholder data flows to any consumer.

## Threat Flags

No new security-relevant surface beyond what the plan's threat model covers. The migration is read-only in-memory; `node.get(k, 0.0)` defaults prevent any exception on malformed nodes (T-14-01-02 accepted).

## Self-Check: PASSED
