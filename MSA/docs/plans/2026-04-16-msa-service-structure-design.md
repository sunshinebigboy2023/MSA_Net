# MSA Service Structure Design

**Goal:** Reorganize the standalone MoMKE inference code under `MSA` into a cleaner service-oriented layout that is easier to maintain and extend locally.

**Architecture**

- Keep `MoMKE/` as the model core and keep `config.py` / `cross_attn_encoder.py` at the root because the model imports depend on them.
- Move the standalone inference layer into `msa_service/` and organize it by backend-like responsibilities:
  - `controller/`: HTTP entrypoints
  - `service/`: orchestration, feature extraction, inference
  - `dao/`: task persistence abstraction
  - `domain/`: runtime metadata, schemas, task models
- Create stable top-level asset folders:
  - `models/checkpoints/` for inference checkpoints
  - `outputs/` for result JSON
  - `temp/` for intermediate files
- Keep only the GCNet assets still useful for future raw-modality preprocessing, and remove unrelated research/baseline code.

**Data Flow**

1. CLI or HTTP request enters through `msa_service.cli` or `msa_service.controller.http_server`.
2. `AnalysisService` creates a task through `TaskDao`.
3. `AnalysisService` resolves features from raw text or precomputed feature files.
4. `MoMKEPredictor` loads the checkpoint and returns a structured sentiment result.
5. Outputs can be written into `outputs/`.

**Cleanup Rules**

- Preserve currently working CMUMOSI inference.
- Copy any dataset metadata still required by `config.py` into top-level `dataset/` before deleting duplicates from `GCNet/`.
- Remove old duplicate service package `momke_service/` after tests pass against `msa_service/`.
- Remove caches and experimental/baseline directories that are unrelated to the standalone service target.

**Validation**

- Unit tests must pass against `msa_service`.
- Text-only CLI inference must still run locally.
- Feature-file CLI inference must still run locally.
