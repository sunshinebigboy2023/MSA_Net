# MSA Service Structure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the local MoMKE standalone service into a cleaner backend-style layout without breaking the currently working CMUMOSI inference path.

**Architecture:** Keep `MoMKE` as the model core, move service code under `msa_service/controller`, `msa_service/service`, `msa_service/dao`, and `msa_service/domain`, then prune unrelated research assets after copying required metadata into stable top-level folders.

**Tech Stack:** Python, PyTorch, Transformers, unittest, simple WSGI server

---

### Task 1: Finalize service package structure

**Files:**
- Create: `msa_service/dao/__init__.py`
- Create: `msa_service/dao/task_dao.py`
- Modify: `msa_service/service/analysis_service.py`
- Modify: `msa_service/controller/http_server.py`
- Modify: `msa_service/__init__.py`

**Step 1: Write the failing test**

Use existing tests to verify imports fail until references are updated.

**Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL with imports still pointing at the old package.

**Step 3: Write minimal implementation**

Add the `dao` package, update imports, and make `AnalysisService` use the new task DAO naming.

**Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: import errors are resolved.

**Step 5: Commit**

```bash
git add msa_service tests docs/plans
git commit -m "refactor: reorganize standalone service package"
```

### Task 2: Move stable runtime assets into service-friendly locations

**Files:**
- Create: `models/checkpoints/`
- Modify: `tests/test_*.py`
- Move: root `*CMUMOSI*.pth` to `models/checkpoints/`

**Step 1: Write the failing test**

Use current checkpoint-discovery tests that still expect the file in repo root.

**Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL after moving the checkpoint unless discovery logic is updated.

**Step 3: Write minimal implementation**

Update test helpers and runtime discovery to search `models/checkpoints/` first, then move the checkpoint.

**Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: checkpoint discovery succeeds.

**Step 5: Commit**

```bash
git add models tests
git commit -m "chore: organize standalone model assets"
```

### Task 3: Preserve required dataset metadata before cleanup

**Files:**
- Create: `dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl`
- Create: `dataset/CMUMOSEI/CMUMOSEI_features_raw_2way.pkl`
- Create: `dataset/IEMOCAP/IEMOCAP_features_raw_4way.pkl`
- Create: `dataset/IEMOCAP/IEMOCAP_features_raw_6way.pkl`

**Step 1: Write the failing test**

Manual verification is enough here because runtime path lookup depends on file existence.

**Step 2: Run test to verify it fails**

Check: `Get-ChildItem dataset -Recurse`
Expected: metadata files are missing before copy.

**Step 3: Write minimal implementation**

Copy the required `.pkl` files from `GCNet/dataset` into top-level `dataset`.

**Step 4: Run test to verify it passes**

Check: `Get-ChildItem dataset -Recurse`
Expected: metadata files exist in top-level `dataset`.

**Step 5: Commit**

```bash
git add dataset
git commit -m "chore: copy required dataset metadata for standalone service"
```

### Task 4: Remove duplicate and research-only assets

**Files:**
- Delete: `momke_service/`
- Delete: `__pycache__/`
- Delete: `GCNet/baseline-*`
- Delete: `GCNet/gcnet/`
- Delete: unneeded `GCNet/run*.sh`, docs, and env files

**Step 1: Write the failing test**

Use the existing service tests and smoke commands as the regression suite.

**Step 2: Run test to verify it fails**

Only if a deletion accidentally removed a required runtime asset.

**Step 3: Write minimal implementation**

Delete obsolete packages and research-only directories after required assets are safely copied.

**Step 4: Run test to verify it passes**

Run:
- `python -m unittest discover -s tests -v`
- `python -m msa_service.cli --text "this movie was unexpectedly good"`

Expected: all tests pass and CLI returns a structured result.

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: prune standalone workspace for service deployment"
```
